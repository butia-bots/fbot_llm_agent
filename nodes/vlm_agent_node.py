#!/usr/bin/env python3

from robot_interface.robot_interface import RobotInterface
from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from typing import List, Optional, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.runnables import chain as chain_decorator
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from butia_vision_msgs.msg import Recognitions3D
from fbot_llm_agent.srv import ExecuteTasks, ExecuteTasksRequest, ExecuteTasksResponse
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import numpy as np
from PIL import Image
import rospy
import cv2
import base64
from io import BytesIO
import re
import os
import math

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Fbot-VLM-Agent"


class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]

class AgentState(TypedDict):
    robot_interface: RobotInterface
    available_waypoints: List[str]
    input: str
    img: str
    recognitions: Recognitions3D
    prediction: Prediction
    scratchpad: List[BaseMessage]
    observation: str

def grasp(state: AgentState):
    robot_interface = state['robot_interface']
    action_args = state['prediction']['args']
    if action_args is None or len(action_args) != 1:
        return f"Failed to grasp object at bounding box labeled as number {action_args}"
    description_id = action_args[0]
    description_id = int(description_id)
    try:
        description = state['recognitions'].descriptions[description_id]
    except:
        return f"Error: no bbox for : {description_id}"
    quat = [
        description.bbox.center.orientation.x,
        description.bbox.center.orientation.y,
        description.bbox.center.orientation.z,
        description.bbox.center.orientation.w,
    ]
    pose = np.array([
        description.bbox.center.position.x,
        description.bbox.center.position.y,
        description.bbox.center.position.z,
        *euler_from_quaternion(quat),
    ])
    pose = robot_interface.transform_pose(pose, description.header.frame_id, robot_interface.get_arm_reference_frame())
    pose[3] = 0.0
    pose[4] = 0.0
    robot_interface.grasp(grasp_pose=pose)
    robot_interface.retreat_arm()
    return f"Grasped {description_id}"

def place_on(state: AgentState):
    robot_interface = state['robot_interface']
    action_args = state['prediction']['args']
    if action_args is None or len(action_args) != 1:
        return f"Failed to place object at bounding box labeled as number {action_args}"
    description_id = action_args[0]
    description_id = int(description_id)
    try:
        description = state['recognitions'].descriptions[description_id]
    except:
        return f"Error: no bbox for : {description_id}"
    quat = [
        description.bbox.center.orientation.x,
        description.bbox.center.orientation.y,
        description.bbox.center.orientation.z,
        description.bbox.center.orientation.w,
    ]
    pose = np.array([
        description.bbox.center.position.x,
        description.bbox.center.position.y,
        description.bbox.center.position.z,
        *euler_from_quaternion(quat),
    ])
    pose = robot_interface.transform_pose(pose, description.header.frame_id, robot_interface.get_arm_reference_frame())
    pose[3] = 0.0
    pose[4] = 0.0
    robot_interface.place(place_pose=pose)
    robot_interface.retreat_arm()
    return f"Placed on {description_id}"

def navigate_to_waypoint(state: AgentState):
    robot_interface = state['robot_interface']
    action_args = state['prediction']['args']
    if action_args is None or len(action_args) != 1:
        return f"Failed to navigate to {action_args}"
    waypoint_name = action_args[0]
    waypoint_name = str(waypoint_name).strip("'").strip('"')
    waypoint_pose = robot_interface.get_waypoint_pose(waypoint_name=waypoint_name)
    robot_interface.move_mobile_base(pose=waypoint_pose)
    return f"Navigated to waypoint {waypoint_name}"

def follow_person(state: AgentState):
    robot_interface = state['robot_interface']
    action_args = state['prediction']['args']
    if action_args is None or len(action_args) != 1:
        return f"Failed to follow person {action_args}"
    description_id = action_args[0]
    description_id = int(description_id)
    try:
        description = state['recognitions'].descriptions[description_id]
    except:
        return f"Error: no bbox for : {description_id}"
    last_pose = np.array([
        description.bbox.center.position.x,
        description.bbox.center.position.y,
        description.bbox.center.position.z,
        0.0,
        0.0,
        0.0
    ])
    last_pose = robot_interface.transform_pose(last_pose, description.header.frame_id, robot_interface.get_map_reference_frame())
    robot_pose = robot_interface.get_mobile_base_pose()
    rate = rospy.Rate(10)
    while np.linalg.norm(last_pose[:2] - robot_pose[:2]) > 1.0:
        last_pose[5] = math.atan2(last_pose[1] - robot_pose[1], last_pose[0] - robot_pose[0])
        robot_interface.move_mobile_base(last_pose, blocking=False)
        positions, clouds = robot_interface.object_detection_3d(class_name=description.label)
        poses = [robot_interface.transform_pose(np.array([*position, 0.0, 0.0, 0.0]), robot_interface.get_camera_reference_frame(), robot_interface.get_map_reference_frame()) for position in positions]
        poses.sort(key=lambda p: np.linalg.norm(p[:2] - last_pose[:2]))
        last_pose = poses[0]
        robot_pose = robot_interface.get_mobile_base_pose()
        rate.sleep()
    return f"Followed: {description_id}"


@chain_decorator
def mark_view(robot_interface: RobotInterface):
    recognitions, annotated_image_arr = robot_interface.annotate_camera_view()
    annotated_image = Image.fromarray(cv2.cvtColor(annotated_image_arr, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    annotated_image.save(buffered, format="PNG")
    return {
        'img': base64.b64encode(buffered.getvalue()).decode(),
        'recognitions': recognitions,
    }

def annotate(state: AgentState):
    marked_view = mark_view.with_retry().invoke(state['robot_interface'])
    return {**state, **marked_view}

def parse(text: str):
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = text.strip().split("\n")[-1]
    action_str = action_block[len(action_prefix) :]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip()
    if action_input is not None:
        action_input = [
            inp.strip().strip("[]") for inp in action_input.strip().split(";")
        ]
    return {"action": action, "args": action_input}

def format_descriptions(state: AgentState):
    labels = []
    for i, description in enumerate(state['recognitions'].descriptions):
        text = description.label
        labels.append(f'{i}: "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}

def select_tool(state: AgentState):
    action = state['prediction']['action']
    if action == 'ANSWER':
        return END
    if action == 'retry':
        return 'agent'
    return action

class VLMAgentNode:
    def __init__(self):
        self.read_parameters()
        if self.llm_api_type == 'openai':
            self.chat_model = ChatOpenAI(model_name=self.vlm_model, openai_api_base=self.llm_api_base_url)
        elif self.llm_api_type == 'google':
            self.chat_model = ChatGoogleGenerativeAI(model=self.vlm_model)
        self.robot_interface = RobotInterface(manipulator_model=self.manipulator_model)
        self.agent_prompt = hub.pull(self.prompt_repo)
        self.agent = annotate | RunnablePassthrough.assign(
            prediction=format_descriptions | self.agent_prompt | self.chat_model | StrOutputParser() | parse
        )
        self.make_graph()
        self.execute_tasks_server = rospy.Service('/fbot_llm_agent/execute_tasks', ExecuteTasks, handler=self.handle_execute_tasks)

    def handle_execute_tasks(self, req: ExecuteTasksRequest):
        res = ExecuteTasksResponse()
        response = ''
        for task in req.task_list:
            result = self.agent_executor.invoke(
                {
                    'robot_interface': self.robot_interface,
                    'input': task.description,
                    'scratchpad': [],
                    'available_waypoints': self.robot_interface.get_available_waypoints()
                },
                {
                    'recursion_limit': self.max_steps
                }
            )
            pred = result.get('prediction') or {}
            action = pred.get('action')
            action_input = pred.get('args')
            if 'ANSWER' in action:
                final_answer = action_input[0]
            response = final_answer
        res.response = response
        return res

    def make_graph(self):
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("agent", self.agent)
        graph_builder.set_entry_point("agent")
        
        graph_builder.add_node("update_scratchpad", self.update_scratchpad)
        graph_builder.add_edge("update_scratchpad", "agent")

        tools = {
            'NavigateToWaypoint': navigate_to_waypoint,
            'FollowPerson': follow_person,
        }

        if self.manipulator_model is not None:
            manipulation_tools = {
                'Grasp': grasp,
                'PlaceOn': place_on,
            }
            tools = {**tools, **manipulation_tools}

        for node_name, tool in tools.items():
            graph_builder.add_node(node_name, RunnableLambda(tool) | (lambda observation: {"observation": observation}))
            graph_builder.add_edge(node_name, "update_scratchpad")
        
        graph_builder.add_conditional_edges("agent", select_tool)
        self.agent_executor = graph_builder.compile()

    def update_scratchpad(self, state: AgentState):
        old = state.get("scratchpad")
        if old:
            txt: str = old[0].content
            last_line = txt.rsplit("\n", 1)[-1]
            step = int(re.match(r"\d+", last_line).group()) + 1
        else:
            txt = "Previous action observations:\n"
            step = 1
        txt += f"\n{step}. {state['observation']}"
        if self.llm_api_type in ('openai',):
            return {**state, "scratchpad": [SystemMessage(content=txt)]}
        elif self.llm_api_type in ('google',):
            return {**state, "scratchpad": [HumanMessage(content=txt)]}

    def read_parameters(self):
        self.llm_api_base_url = rospy.get_param("~llm_api_base_url", 'http://localhost:11434')
        self.vlm_model = rospy.get_param('~vlm_model', 'llava-llama3')
        self.manipulator_model = rospy.get_param('~manipulator_model', 'doris_arm')
        self.prompt_repo = rospy.get_param("~prompt_repo", "crislmfroes/fbot-vlm-agent")
        self.max_steps = rospy.get_param("~max_steps", 25)
        self.llm_api_type = rospy.get_param("~llm_api_type", "google")


if __name__ == '__main__':
    rospy.init_node('vlm_agent_node', anonymous=True)
    node = VLMAgentNode()
    rospy.spin()