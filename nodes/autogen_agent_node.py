#!/usr/bin/env python3

from fbot_llm_agent.srv import ExecuteTasks, ExecuteTasksRequest, ExecuteTasksResponse
from fbot_llm_agent.msg import Task as TaskMsg
from agents.autogen_agent import robot_agent, robot_operator, robot_groupchat_manager, robot_interface
import rospy

def handle_execute_tasks(req: ExecuteTasksRequest):
    for task_msg in req.task_list:
        answer = robot_operator.initiate_chat(recipient=robot_agent, max_turns=1, message=task_msg.description + f"\nobservation: {robot_interface.caption()}").summary
    res = ExecuteTasksResponse()
    res.response = answer
    return res

if __name__ == '__main__':
    rospy.init_node('autogen_agent_node', anonymous=True)
    execute_tasks_server = rospy.Service('/fbot_llm_agent/execute_tasks', ExecuteTasks, handler=handle_execute_tasks)
    rospy.spin()