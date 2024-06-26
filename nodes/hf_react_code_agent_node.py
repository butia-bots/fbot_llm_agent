#!/usr/bin/env python3

from fbot_llm_agent.srv import ExecuteTasks, ExecuteTasksRequest, ExecuteTasksResponse
from fbot_llm_agent.msg import Task as TaskMsg
from robot_interface.robot_interface import RobotInterface
import rospy
from transformers import ReactCodeAgent, HfEngine


def handle_execute_tasks(req: ExecuteTasksRequest):
    for task_msg in req.task_list:
        agent.run(task=task_msg.description)
    res = ExecuteTasksResponse()
    return res

if __name__ == '__main__':
    rospy.init_node('crewai_agent_node', anonymous=True)
    robot_interface = RobotInterface(manipulator_model="wx200")
    llm_engine = HfEngine(model="meta-llama/Meta-Llama-3-8B-Instruct")
    agent = ReactCodeAgent(tools=robot_interface.get_code_tools_hf(), llm_engine=llm_engine)
    execute_tasks_server = rospy.Service('/fbot_llm_agent/execute_tasks', ExecuteTasks, handler=handle_execute_tasks)
    rospy.spin()