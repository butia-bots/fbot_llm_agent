#!/usr/bin/env python3

from ai2thor.controller import Controller
from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from typing import List, Optional, TypedDict, Dict
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
from maestro.visualizers import MarkVisualizer
import supervision as sv
import random
from langgraph.errors import GraphRecursionError

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Fbot-VLM-Agent"

OBJECT_IN_HAND = 'None'
OBJECT_ID2NAME = {}
CURRENT_LOCATION = 'None'
robot_interface = None

class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]

class AgentState(TypedDict):
    robot_interface: Controller
    available_waypoints: List[str]
    input: str
    img: str
    recognitions: Dict[str, np.ndarray]
    prediction: Prediction
    scratchpad: List[BaseMessage]
    object_in_hand: str
    current_location: str
    observation: str

def grasp(state: AgentState):
    global OBJECT_IN_HAND
    global OBJECT_ID2NAME
    global robot_interface
    object_name2id = dict([(v,k) for k,v in OBJECT_ID2NAME.items()])
    #robot_interface = state['robot_interface']
    action_args = state['prediction']['args']
    if action_args is None or len(action_args) != 1:
        return f"Failed to grasp object at bounding box labeled as number {action_args}"
    description_id = action_args[0].split(',')[0]
    object_name = action_args[0].split(',')[1]
    description_id = int(description_id)
    recognitions = list(state['recognitions'].items())
    try:
        description = recognitions[description_id]
    except:
        return f"Error: no bbox for : {description_id}"
    if description[0] == OBJECT_IN_HAND:
        return f"Error: {description_id} is already grasped, you must choose another action!"
    robot_interface.step(
        action="PickupObject",
        objectId=description[0],
        forceAction=False,
        manualInteract=False
    )
    OBJECT_IN_HAND = OBJECT_ID2NAME[description[0]]
    return f"Grasped {description_id}, {object_name}"

def place_on(state: AgentState):
    global OBJECT_IN_HAND
    global robot_interface
    global OBJECT_ID2NAME
    object_name2id = dict([(v,k) for k,v in OBJECT_ID2NAME.items()])
    #robot_interface = state['robot_interface']
    action_args = state['prediction']['args']
    if action_args is None or len(action_args) != 1:
        return f"Failed to place object at bounding box labeled as number {action_args}"
    description_id = action_args[0].split(',')[0]
    object_name = action_args[0].split(',')[1]
    description_id = int(description_id)
    recognitions = list(state['recognitions'].items())
    try:
        description = recognitions[description_id]
    except:
        return f"Error: no bbox for : {description_id}"
    robot_interface.step(
        action="PutObject",
        objectId=description[0],
        forceAction=True,
        placeStationary=False
    )
    return f"Placed on {description_id}, {object_name}"

def navigate_to_waypoint(state: AgentState):
    global OBJECT_ID2NAME
    global robot_interface
    global CURRENT_LOCATION
    object_name2id = dict([(v,k) for k,v in OBJECT_ID2NAME.items()])
    #robot_interface = state['robot_interface']
    action_args = state['prediction']['args']
    if action_args is None or len(action_args) != 1:
        return f"Failed to navigate to {action_args}"
    waypoint_name = action_args[0]
    waypoint_name = str(waypoint_name).strip("'").strip('"')
    object_id = None
    if waypoint_name not in object_name2id:
        recognition_keys = list(state['recognitions'].keys())
        for recognition_key in recognition_keys:
            if recognition_key.startswith(waypoint_name):
                object_id = recognition_key
                break
    elif waypoint_name not in object_name2id:
        return f"Failed to navigate to {action_args}. Make sure the waypoint name contains the numbers after the underline, as in `WaypointName_0`"
    else:
        object_id = object_name2id[waypoint_name]
    if object_id is None:
        return f"Failed to navigate to {action_args}. Make sure the waypoint name contains the numbers after the underline, as in `WaypointName_0`"
    poses = robot_interface.step(
        action="GetInteractablePoses",
        objectId=object_id,
    ).metadata["actionReturn"]
    pose = random.choice(poses)
    robot_interface.step("TeleportFull", **pose)
    CURRENT_LOCATION = waypoint_name
    return f"Navigated to waypoint {waypoint_name}"

def follow_person(state: AgentState):
    global OBJECT_ID2NAME
    global robot_interface
    object_name2id = dict([(v,k) for k,v in OBJECT_ID2NAME.items()])
    #robot_interface = state['robot_interface']
    action_args = state['prediction']['args']
    if action_args is None or len(action_args) != 1:
        return f"Failed to follow person {action_args}"
    description_id = action_args[0]
    description_id = int(description_id)
    recognitions = list(state['recognitions'].items())
    try:
        description = recognitions[description_id]
    except:
        return f"Error: no bbox for : {description_id}"
    waypoint_name = description[0]
    poses = robot_interface.step(
        action="GetInteractablePoses",
        objectId=object_name2id[waypoint_name],
    ).metadata["actionReturn"]
    pose = random.choice(poses)
    robot_interface.step("TeleportFull", **pose)
    return f"Followed: {description_id}"


@chain_decorator
def mark_view(robot_interface: Controller):
    global OBJECT_IN_HAND
    global CURRENT_LOCATION
    image_arr = robot_interface.last_event.frame
    detections2d = robot_interface.last_event.instance_detections2D
    visualizer = MarkVisualizer()
    xyxy = np.array([v for k, v in detections2d.items()])
    masks = []
    for bbox in xyxy:
        mask = np.zeros(shape=(*image_arr.shape[:2],), dtype=bool)
        mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = True
        masks.append(mask)
    mask = np.array(masks)
    marks = sv.Detections(xyxy=xyxy, mask=mask)
    annotated_image_arr = visualizer.visualize(image=image_arr, marks=marks, with_box=True)
    annotated_image = Image.fromarray(annotated_image_arr)
    buffered = BytesIO()
    annotated_image.save(buffered, format="PNG")
    return {
        'img': base64.b64encode(buffered.getvalue()).decode(),
        'recognitions': detections2d,
        'object_in_hand': OBJECT_IN_HAND,
        'current_location': CURRENT_LOCATION,
    }

def annotate(state: AgentState):
    global robot_interface
    marked_view = mark_view.with_retry().invoke(robot_interface)
    return {**state, **marked_view}

def format_descriptions(state: AgentState):
    global OBJECT_ID2NAME
    object_name2id = dict([(v,k) for k,v in OBJECT_ID2NAME.items()])
    labels = []
    for i, description in enumerate(state['recognitions'].items()):
        if description[0] in OBJECT_ID2NAME:
            text = OBJECT_ID2NAME[description[0]].split('_')[0]
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
        global OBJECT_ID2NAME
        global CURRENT_LOCATION
        global robot_interface
        self.read_parameters()
        if self.llm_api_type == 'openai':
            self.chat_model = ChatOpenAI(model_name=self.vlm_model, openai_api_base=self.llm_api_base_url, temperature=0)
        elif self.llm_api_type == 'google':
            self.chat_model = ChatGoogleGenerativeAI(model=self.vlm_model, convert_system_message_to_human=True)
        self.robot_interface = Controller(width=640, height=480, renderInstanceSegmentation=True)
        robot_interface = self.robot_interface
        self.robot_interface.step(action='Done')
        waypoints = self.get_waypoints()
        object_name2id = dict([(v,k) for k,v in OBJECT_ID2NAME.items()])
        waypoint_name = waypoints[0]
        poses = robot_interface.step(
            action="GetInteractablePoses",
            objectId=object_name2id[waypoint_name],
        ).metadata["actionReturn"]
        pose = random.choice(poses)
        robot_interface.step("TeleportFull", **pose)
        CURRENT_LOCATION = waypoint_name
        self.agent_prompt = hub.pull(self.prompt_repo)
        self.agent = annotate | RunnablePassthrough.assign(
            prediction=format_descriptions | self.agent_prompt | self.chat_model | StrOutputParser() | self.parse
        )
        self.make_graph()
        self.execute_tasks_server = rospy.Service('/fbot_llm_agent/execute_tasks', ExecuteTasks, handler=self.handle_execute_tasks)

    def get_waypoints(self):
        global OBJECT_ID2NAME
        obj_names = []
        for obj in self.robot_interface.last_event.metadata["objects"]:
            name = obj["objectId"].split('|')[0]+f"_{len(obj_names)}"
            OBJECT_ID2NAME[obj["objectId"]] = name
            obj_names.append(name)
        return obj_names
    
    def handle_execute_tasks(self, req: ExecuteTasksRequest):
        res = ExecuteTasksResponse()
        response = ''
        for task in req.task_list:
            try:
                result = self.agent_executor.invoke(
                    {
                        'robot_interface': self.robot_interface,
                        'input': task.description,
                        'scratchpad': [],
                        'available_waypoints': self.get_waypoints()
                    },
                    {
                        'recursion_limit': self.max_steps,
                    }
                )
                pred = result.get('prediction') or {}
                action = pred.get('action')
                action_input = pred.get('args')
                if 'ANSWER' in action:
                    final_answer = action_input[0]
                response = final_answer
            except (GraphRecursionError, UnboundLocalError) as e:
                response = ''
        res.response = response
        self.robot_interface.step(action='Done')
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

    def parse(self, text: str):
        
        action_prefix = "Action: "
        if not text.strip().split("\n")[-1].startswith(action_prefix):
            return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
        thought_prefix = "Thought: "
        thought = '\n'.join(text.strip().split('\n')[:-1]).strip(thought_prefix)
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

    def read_parameters(self):
        self.llm_api_base_url = rospy.get_param("~llm_api_base_url", 'http://localhost:11434')
        self.vlm_model = rospy.get_param('~vlm_model', 'llava-llama3')
        self.prompt_repo = rospy.get_param("~prompt_repo", "crislmfroes/fbot-vlm-agent")
        self.max_steps = rospy.get_param("~max_steps", 25)
        self.llm_api_type = rospy.get_param("~llm_api_type", "google")


if __name__ == '__main__':
    rospy.init_node('vlm_agent_node', anonymous=True)
    node = VLMAgentNode()
    rospy.spin()