#!/usr/bin/env python3

from crewai import Agent, Crew, Process, Task
from fbot_robot_learning.msg import PolicyInfo
from fbot_robot_learning.srv import GetAvailablePolicies, GetAvailablePoliciesResponse
from fbot_llm_agent_tools.imitation.imitation_tool import make_imitation_learning_tool
from fbot_llm_agent.srv import ExecuteTasks, ExecuteTasksRequest, ExecuteTasksResponse
from langchain_community.chat_models.openai import ChatOpenAI
import rospy

def handle_execute_tasks(req: ExecuteTasksRequest):
    tasks = []
    for task_msg in req.task_list:
        task = Task(
            description=task_msg.description,
            agent=agent,
            expected_output=task_msg.expected_output
        )
        tasks.append(task)
    crew = Crew(
        tasks=tasks,
        agents=[agent,],
        process=Process.sequential,
        verbose=True
    )
    result = crew.kickoff()
    res = ExecuteTasksResponse()
    res.response = result
    return res

if __name__ == '__main__':
    rospy.init_node('crewai_agent_node', anonymous=True)
    robot_name = rospy.get_param('~robot_name', "DoRIS")
    llm_api_model = rospy.get_param('~llm_api_model', 'gpt-3.5-turbo')
    llm_api_base_url = rospy.get_param('~llm_api_base_url', None)
    get_available_policies = rospy.ServiceProxy('/fbot_robot_learning/get_available_policies', GetAvailablePolicies)
    execute_tasks_server = rospy.Service('/fbot_llm_agent/execute_tasks', ExecuteTasks, handler=handle_execute_tasks)
    available_policies_res: GetAvailablePoliciesResponse = get_available_policies.call()
    
    tools = []
    
    for policy_info in available_policies_res.available_policies_info:
        tools.append(make_imitation_learning_tool(policy_info=policy_info))

    llm = ChatOpenAI(model=llm_api_model, base_url=llm_api_base_url)

    agent = Agent(
        role='Service Robot',
        goal='Help with daily tasks inside the house',
        backstory=f'You are {robot_name}, a domestic service robot developed by team Fbot@Home.',
        llm=llm,
        tools=tools,
        memory=True
    )

    rospy.spin()