#!/usr/bin/env python3

from typing import Any
from fbot_llm_agent.srv import ExecuteTasks, ExecuteTasksRequest, ExecuteTasksResponse
from fbot_llm_agent.msg import Task as TaskMsg
from robot_interface.butia_behavior_interface import ButiaBehaviorInterface
import rospy
from transformers import ReactCodeAgent, HfEngine, CodeAgent
from openai import OpenAI

class OllamaEngine:
    def __init__(self, model: str) -> None:
        self.client = OpenAI(api_key='ollama', base_url='http://localhost:11434/v1')
        self.model = model

    def __call__(self, messages, stop_sequences=['Task',]) -> Any:
        return self.client.chat.completions.create(messages=messages, model=self.model, stop=stop_sequences, temperature=0).choices[0].message.content

def handle_execute_tasks(req: ExecuteTasksRequest):
    for task_msg in req.task_list:
        answer = agent.run(task=task_msg.description)
    res = ExecuteTasksResponse()
    res.response = answer
    return res

if __name__ == '__main__':
    rospy.init_node('hf_react_code_agent_node', anonymous=True)
    #robot_interface = RobotInterface(manipulator_model="wx200", arm_group_name="interbotix_arm", arm_controller_ns="interbotix_arm_controller", gripper_controller_ns="interbotix_gripper_controller")
    robot_interface = ButiaBehaviorInterface(fake_execution=True)
    llm_engine = OllamaEngine(model="llama3.1")
    agent = ReactCodeAgent(tools=robot_interface.get_code_tools_hf(), llm_engine=llm_engine)
    execute_tasks_server = rospy.Service('/fbot_llm_agent/execute_tasks', ExecuteTasks, handler=handle_execute_tasks)
    rospy.spin()