from autogen.agentchat import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from robot_interface.butia_behavior_interface import ButiaBehaviorInterface
from typing import Optional, Literal
import yaml
import rospkg

with open(rospkg.RosPack().get_path('fbot_llm_agent')+'/config/autogen_agent.yaml') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

robot_interface = ButiaBehaviorInterface(fake_execution=config['fake_execution'])

config_list = [
  {
    "api_type": "ollama",
    "model": "llama3",
    #"client_host": "http://jetson:11434",
    "client_host": "http://0.0.0.0:11434",
    "native_tool_calls": False,
    #"api_key": "ollama",
  }
]

robot_agent = AssistantAgent(
    name="General Purpose Service Robot",
    #system_message="You are a general purpose service robot. When given a step-by-step plan for executing a task, you execute each step sequentially. Reply with TERMINATE once the task is completed",
    system_message="You are a general purpose service robot. Reply with TERMINATE once the task is completed",
    llm_config={'config_list': config_list}
)

robot_operator = UserProxyAgent(
    name="Robot Operator",
    is_termination_msg=lambda msg: 'TERMINATE' in msg['content'],
    human_input_mode='NEVER',
    code_execution_config=False,
    llm_config=False
)

function_executor = AssistantAgent(
    name="Function Executor",
    llm_config=False,
    human_input_mode='NEVER',
    is_termination_msg=lambda msg: 'tool_calls' not in msg
)

planner = AssistantAgent(
    name="Task Planner",
    system_message=f"""You are the task-level planning system of a service robot with the following capabilities:\n- navigation\n- person following\n- visual question answering\n- textual question answering\n- speech synthesis\nAfter taking a request for a task, you must reply with a step-by-step plan to be executed by the robot. The plan should only include the previously mentioned capabilities.\n\nExamples: {config['planner_examples']}\n\nAvailable destinations for navigation: {config['locations']}""",
    llm_config={'config_list': config_list},
)

@function_executor.register_for_execution()
@robot_agent.register_for_llm(name='query_spatial_memory', description="Query a static spatial memory with descriptions and poses of locations in the environment.")
def query_spatial_memory(query: str)->str:
    return {
        'locations': robot_interface.query_spatial_memory(query=query)
    }

@function_executor.register_for_execution()
@robot_agent.register_for_llm(name='move_to_pose', description='Navigates to a pose in the environment, and return execution status')
def move_to_pose(x: float, y: float, yaw: float)->str:
    return {
        'outcome': robot_interface.move_to_pose(x=x, y=y, yaw=yaw),
        'observation': robot_interface.caption()
    }

@function_executor.register_for_execution()
@robot_agent.register_for_llm(name='move_to', description='Navigate to a room, furniture or waypoint and return execution status')
def move_to(destination: Literal[tuple(config['locations'])])->str:
    return {
        'outcome': robot_interface.move_to(destination=destination),
        'observation': robot_interface.caption()
    }

@function_executor.register_for_execution()
@robot_agent.register_for_llm(name='follow', description='Follow the person in front of you and return execution status')
def follow()->str:
    return {
        'outcome': robot_interface.follow(),
        'observation': robot_interface.caption()
    }

@function_executor.register_for_execution()
@robot_agent.register_for_llm(name='answer', description='Speak the answer to a question from the person in front of you, and return execution status')
def answer()->str:
    return {
        'outcome': robot_interface.answer(),
        'observation': robot_interface.caption()
    }

@function_executor.register_for_execution()
@robot_agent.register_for_llm(name='visual_question_answering', description='Speak the answer to a visual question about the destination where you are, and return execution status')
def visual_question_answering(question: str)->str:
    return {
        'outcome': robot_interface.visual_question_answering(question=question),
        'observation': robot_interface.caption()
    }

@function_executor.register_for_execution()
@robot_agent.register_for_llm(name='speak', description='Speak an utterance and return execution status')
def speak(utterance: str)->str:
    return {
        'outcome': robot_interface.speak(utterance=utterance),
        'observation': robot_interface.caption()
    }

@function_executor.register_for_execution()
@robot_agent.register_for_llm(name='pick', description='Pick an object and return execution status')
def pick(object: str)->str:
    return {
        'outcome': robot_interface.pick(object=object),
        'observation': robot_interface.caption()
    }

@function_executor.register_for_execution()
@robot_agent.register_for_llm(name='place', description='Place previously picked object down, and return execution status')
def place(placement_zone: str)->str:
    return {
        'outcome': robot_interface.place(placement_zone=placement_zone),
        'observation': robot_interface.caption()
    }

robot_agent.register_nested_chats(
    trigger=robot_operator,
    chat_queue=[
        {
            'sender': function_executor,
            'recipient': robot_agent,
            'summary_method': 'last_msg'
        }
    ]
)

robot_groupchat = GroupChat(agents=[robot_operator, planner, robot_agent], messages=[], max_round=4, speaker_selection_method='round_robin')
robot_groupchat_manager = GroupChatManager(groupchat=robot_groupchat)

robot_agent.register_nested_chats(
    trigger=robot_groupchat_manager,
    chat_queue=[
        {
            'sender': function_executor,
            'recipient': robot_agent,
            'summary_method': 'last_msg'
        }
    ]
)
