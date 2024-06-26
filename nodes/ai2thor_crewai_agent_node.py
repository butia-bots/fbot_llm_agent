#!/usr/bin/env python3

from crewai import Agent, Crew, Process, Task
from fbot_llm_agent.srv import ExecuteTasks, ExecuteTasksRequest, ExecuteTasksResponse
from fbot_llm_agent.msg import Task as TaskMsg
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.chat_models.ollama import ChatOllama
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq.chat_models import ChatGroq
from langchain_core.tools import StructuredTool
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_experimental.tools import PythonREPLTool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from ai2thor.controller import Controller, BFSController, BFSSearchPoint
from ai2thor.util.metrics import get_shortest_path_to_object
import rospy
from typing import List
import Levenshtein
import random
random.seed(123)
import math
import networkx
import numpy as np
import tqdm
import wandb
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
import prior

OBJECT_IN_HAND = None
CURRENT_LOCATION = None

def feedback(msg: str):
    global OBJECT_IN_HAND
    global CURRENT_LOCATION
    #return f"Feedback: {msg}"
    #return f"Feedback: {msg}\nCurrent location: next to {CURRENT_LOCATION}"
    return f"Feedback: {msg}\nObject in hand: {OBJECT_IN_HAND}\nCurrent location: {CURRENT_LOCATION}"
    #return f"Feedback: {msg}\nObject in hand: {OBJECT_IN_HAND}\nCurrent location: next to {CURRENT_LOCATION}\nCurrent view description: {query_perception_system('Generate a short 1-phrase description of the scene.')}"

def check_equal_with_llm(a: str, b: str):
    return llm.invoke(f"You must disambiguate similar tags for an object recognition system. Is a {a.lower()} similar to a {b.lower()}? Answer with only `Yes` or `No`.").content.lower().startswith('yes')

def grasp(object_name: str, navigate_to_object=False)->str:
    """Grasp the object with given object_name. You must call `navigate_to_waypoint` with the object name before calling this.
Effects: the object becomes attached to the robot hand
Preconditions: the robot must be at the same location as the object, the robot hand must be empty"""
    global OBJECT_IN_HAND
    object_id = object_name
    if use_smart_tool or navigate_to_object:
        for i in range(1):
            _, success = navigate_to_waypoint(object_id)
            if success == True:
                break
    #if OBJECT_IN_HAND == 'None':
    #    navigate_to_waypoint(object_id)
    #elif check_equal_with_llm(OBJECT_IN_HAND, object_id):
    #    return feedback(f"Hand is already occupied holding {OBJECT_IN_HAND}! Grasp {object_id} failed!")
    global robot_interface
    #if not llm.invoke(f"Is a {object_id} a type of household object smaller than 40 cm? answer with only `Yes` or `No`.").content.lower().startswith('yes'):
    #    return feedback(f"{object_id} is not a type of small manipulable household object! Grasp failed!")
    #object_store = Chroma.from_documents([Document(page_content=o.split('|')[0], metadata=dict(objId=o)) for o in observe()], embeddings)
    object_store = waypoint_store
    doc = object_store.as_retriever().invoke(object_id)[0]
    object_id = doc.metadata['objId']
    object_name = doc.page_content
    obj = [o for o in robot_interface.last_event.metadata["objects"] if o['objectId'].split('|')[0] == object_id.split('|')[0]][0]
    #robot_interface.step(
    #    action="MoveArmBase",
    #    y=0.8,
    #    returnToStart=False,
    #)
    op = [o for o in robot_interface.last_event.metadata['objects'] if o['objectId'].split('|')[0] == object_id.split('|')[0]][0]['position']
    rp = robot_interface.last_event.metadata['agent']['position']
    theta = math.atan2(op['x']-rp['x'],op['z']-rp['z'])+np.pi
    #gp = dict(x=op['x']+math.cos(theta)*0.1, y=op['y'], z=op['z']+math.sin(theta)*0.1)
    gp = op.copy()
    robot_interface.step(
            action="MoveArm",
            position=dict(x=gp['x'], y=gp['y'], z=gp['z']),
            coordinateSpace='world',
            returnToStart=False,
    )
    robot_interface.step(
            action="PickupObject",
            #objectIdCandidates=[object_id,],
    )
    robot_interface.step(
            action="MoveArm",
            position=dict(x=0.0, y=0.0, z=0.0),
            coordinateSpace="armBase",
            returnToStart=False,
    )
    OBJECT_IN_HAND = object_name
    return feedback(f"Grasped {object_name}")

def place_on(object_name: str, container_name: str, grasp_object=False, navigate_to_container=False)->str:
    """Place object_name on container_name, where container_name can be another object, some container, or some support surface. You must call `grasp` with the object_name and `navigate_to_waypoint` with the container_name before calling this.
Effects: the object becomes detached from the robot hand, the robot hand becomes empty
Preconditions: the object is attached to the robot hand, the robot is at the same location as the container"""
    global OBJECT_IN_HAND
    container_id = container_name
    if use_smart_tool:
        if OBJECT_IN_HAND == None:
            grasp(object_name)
        for i in range(1):
            _, success = navigate_to_waypoint(container_id)
            if success == True:
                break
    #if OBJECT_IN_HAND == 'None':
    #   grasp(object_name)
    #elif not check_equal_with_llm(OBJECT_IN_HAND, object_name):
    #    return feedback(f"{object_name} is not currently in hand! Place {object_name} on {container_id} failed!")
    #navigate_to_waypoint(container_id)
    #open_object(container_id)
    global robot_interface
    #object_store = Chroma.from_documents([Document(page_content=o.split('|')[0], metadata=dict(objId=o)) for o in observe()], embeddings)
    object_store = waypoint_store
    doc = object_store.as_retriever().invoke(container_id)[0]
    container_id = doc.metadata['objId']
    container_name = doc.page_content
    container = [o for o in robot_interface.last_event.metadata["objects"] if o['objectId'].split('|')[0] == container_id.split('|')[0]][0]
    #robot_interface.step(
    #    action="MoveArmBase",
    #    y=0.8
    #)
    robot_interface.step(
            action="MoveArm",
            position=dict(x=container['position']['x'], y=container['position']['y'], z=container['position']['z']),
            coordinateSpace='world',
            returnToStart=False,
        )
    robot_interface.step(
            action="ReleaseObject"
        )
    robot_interface.step(
            action="MoveArm",
            position=dict(x=0.0, y=0.0, z=0.0),
            coordinateSpace="armBase",
            returnToStart=False,
        )
    object_name = OBJECT_IN_HAND
    OBJECT_IN_HAND = None
    return feedback(f"Placed {object_name} on {container_name}")

def navigate_to_waypoint(waypoint_name: str)->str:
    """Navigate to waypoint with given waypoint_name
Effects: the robot ends up at the same location as the waypoint"""
    global robot_interface
    global CURRENT_LOCATION
    #global OBJECT_IN_HAND
    #if OBJECT_IN_HAND.lower() == waypoint_name.lower():
    #    return f"Cannot navigate to object: {waypoint_name}, it already is attached to the hand."
    #if check_equal_with_llm(OBJECT_IN_HAND, waypoint_name):
    #    return feedback(f"Navigated to {OBJECT_IN_HAND}")
    waypoint_id = waypoint_name
    doc = waypoint_store.as_retriever().invoke(waypoint_id)[0]
    waypoint_id = doc.metadata['objId']
    waypoint_name = doc.page_content
    op = [o for o in robot_interface.last_event.metadata['objects'] if o['objectId'].split('|')[0] == waypoint_id.split('|')[0]][0]['position']
    rp = robot_interface.last_event.metadata['agent']['position']
    reachable_positions = robot_interface.step(
        action="GetReachablePositions"
    ).metadata["actionReturn"]
    min_x = min(position['x'] for position in reachable_positions)
    max_x = max(position['x'] for position in reachable_positions)
    min_z = min(position['z'] for position in reachable_positions)
    max_z = max(position['z'] for position in reachable_positions)
    grid_size = 0.25
    shape = (int((max_z-min_z)//grid_size),int((max_x-min_x)//grid_size))
    grid = np.zeros(shape=shape, dtype=int)
    grid[:,:] = -1
    graph = networkx.Graph()
    for i, p in enumerate(reachable_positions):
        grid[int((p['z']-max_z)//grid_size),int((p['x']-max_x)//grid_size)] = i
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == -1:
                continue
            if j > 0 and grid[i][j-1] != -1:
                graph.add_edge(grid[i][j], grid[i][j-1])
            if i > 0 and grid[i-1][j] != -1:
                graph.add_edge(grid[i][j], grid[i-1][j])
            if i > 0 and j > 0 and grid[i-1][j-1] != -1:
                graph.add_edge(grid[i][j], grid[i-1][j-1])

            if j < len(grid[i])-1 and grid[i][j+1] != -1:
                graph.add_edge(grid[i][j], grid[i][j+1])
            if i < len(grid)-1 and grid[i+1][j] != -1:
                graph.add_edge(grid[i][j], grid[i+1][j])
            if i < len(grid)-1 and j < len(grid[i])-1 and grid[i+1][j+1] != -1:
                graph.add_edge(grid[i][j], grid[i+1][j+1])

            if j > 0 and i < len(grid)-1 and grid[i+1][j-1] != -1:
                graph.add_edge(grid[i][j], grid[i+1][j-1])
            if i > 0 and j < len(grid[i])-1 and grid[i-1][j+1] != -1:
                graph.add_edge(grid[i][j], grid[i-1][j+1])
    start = sorted(reachable_positions, key=lambda p: math.sqrt(math.pow(p['x']-rp['x'],2)+math.pow(p['z']-rp['z'],2)))[0]
    start = reachable_positions.index(start)
    destination = sorted(reachable_positions, key=lambda p: math.sqrt(math.pow(p['x']-op['x'],2)+math.pow(p['z']-op['z'],2)))[0]
    destination = reachable_positions.index(destination)
    path = networkx.astar_path(graph, start, destination)
    path = [reachable_positions[i] for i in path]
    path = path[-3:]
    #print(path)
    #if len(path) == 1:
    #    CURRENT_LOCATION = waypoint_name
    #    return f"Navigated to waypoint {waypoint_name}"
    start_p = rp
    start_r = robot_interface.last_event.metadata['agent']['rotation']
    if len(path) > 1:
        for p in path:
            #print(p)
            theta = math.atan2(start_p['x']-p['x'], start_p['z']-p['z'])+np.pi
            #theta = math.atan2(p['z']-start_p['z'], p['x']-start_p['x'])
            for i in range(1):
                robot_interface.step(action="Teleport", position=p, rotation=dict(x=0, y=math.degrees(theta), z=0))
            #rospy.Rate(3.0).sleep()
            #robot_interface.step(action="RotateAgent", degrees=-(start_r['y']-math.degrees(theta)), returnToStart=False, fixedDeltaTime=0.02)
            #rospy.Rate(1.0).sleep()
            #start_p = robot_interface.last_event.metadata['agent']['position']
            #start_r = robot_interface.last_event.metadata['agent']['rotation']
            #distance = math.sqrt(math.pow(start_p['x']-p['x'],2)+math.pow(start_p['z']-p['z'],2))
            #robot_interface.step(action="MoveAgent", ahead=distance+0.05, right=0, returnToStart=False, fixedDeltaTime=0.02, speed=0.25)
            #rospy.Rate(1.0).sleep()
            #start_p = robot_interface.last_event.metadata['agent']['position']
            #start_r = robot_interface.last_event.metadata['agent']['rotation']
            start_p = p
            rospy.Rate(1/grid_size).sleep()
            #start_r = dict(x=0, y=theta, z=0)
    op = [o for o in robot_interface.last_event.metadata['objects'] if o['objectId'].split('|')[0] == waypoint_id.split('|')[0]][0]['position']
    rp = robot_interface.last_event.metadata['agent']['position']
    yaw = math.atan2(rp['x']-op['x'], rp['z']-op['z'])+np.pi
    robot_interface.step(action="Teleport", position=rp, rotation=dict(x=0, y=math.degrees(yaw), z=0))
    """poses = robot_interface.step(
        action="GetInteractablePoses",
        objectId=waypoint_id,
    ).metadata["actionReturn"]
    if len(poses) > 0:
        pose = poses[0]
        pose['standing'] = True
        pose['horizon'] = -10.0
        for i in range(10):
            robot_interface.step("TeleportFull", **pose)"""
    #for i in range(10):
    #    robot_interface.step(action="MoveArmBase", y=0.8)
    '''direction = 1
    seen = False
    for i in range(360):
        if waypoint_id in robot_interface.last_event.instance_detections2D:
            bbox = robot_interface.last_event.instance_detections2D[waypoint_id]
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2)//2
            frame = robot_interface.last_event.frame
            if abs(cx - frame.shape[1]//2) < 10:
                seen = True
                break
            if cx > frame.shape[1]//2:
                direction = 1
        else:
            direction = -1
        robot_interface.step(
            action="RotateAgent",
            degrees=direction,
            returnToStart=True
        )
    for i in range(100):
        op = [o for o in robot_interface.last_event.metadata['objects'] if o['objectId'] == waypoint_id][0]['position']
        rp = robot_interface.last_event.metadata['agent']['position']
        if math.sqrt(math.pow(op['x']-rp['x'],2)+math.pow(op['z']-rp['z'],2)) < 1.5:
            break
        robot_interface.step(
            action="MoveAgent",
            ahead=0.01,
            right=0.0,
            returnToStart=True
        )
    if seen:
        CURRENT_LOCATION = waypoint_name'''
    CURRENT_LOCATION = waypoint_name
    seen = True
    return feedback(f"Navigated to waypoint {waypoint_name}"), seen

def query_semantic_memory(query: str)->str:
    """Queries the semantic memory about objects in the house"""
    return llm.invoke(f"Given the following list of objects in the environment of a service robot: {get_waypoints()}\nAnswer the following query: {query}").content

def query_perception_system(query: str)->str:
    """Queries the perception system about objects in view"""
    return llm.invoke(f"Given the following list of objects visible to the camera of a service robot: {observe()}\nAnswer the following query: {query}").content

def get_waypoints()->List[str]:
    """Returns all available waypoint_ids"""
    global robot_interface
    obj_names = []
    for obj in robot_interface.last_event.metadata["objects"]:
        obj_names.append(obj["objectId"])
    return obj_names

def observe()->List[str]:
    """Returns all object_ids and container_ids currently visible"""
    global robot_interface
    return list(robot_interface.last_event.instance_detections2D.keys())

def check_step_is_successfull(step: str, feedback: str)->bool:
    return llm.invoke(f"Given the following feedback message: {feedback}\nGivend the following step of a plan: {step}\nAnswer: was the plan step successfully executed? Answer with Yes or No.").content.lower().startswith('yes')

def handle_execute_tasks(req: ExecuteTasksRequest):
    for task_msg in req.task_list:
        run_custom_agent(task_msg.description, max_depth=0)
    res = ExecuteTasksResponse()
    context = ""
    return res

def run_custom_agent(command: str, max_depth: int):
    global environment_description_cp
    history = ''
    tools_prompt = '\n'.join([f"{t.name}: {t.description} {t.input_schema}" for t in tools])
    if use_planner:
        planner_prompt = f"Imagine you are a service robot in a domestic environment. You can navigate, pick, and place objects. You must answer by providing a step-by-step list for performing tasks. Each step must be one of: navigate, pick, place.\nThe following objects are in the house: {get_waypoints()}\nYou are currently at the following waypoint: {CURRENT_LOCATION}\nAnswer in the following format:\n\nThought: ...(a brief summary that will help you acomplish the task)\nPlan:\n- ...\n\nYour task is: {command}"
        planner_response = llm.invoke(planner_prompt).content
        thought = planner_response.split('\n')[0].strip('Thought:').strip(' ')
        plan = planner_response.split('Plan:')[-1].split('\n')
        plan = [step.strip('-').strip('\n') for step in plan]
        plan = [step for step in plan if step != '']
        plan = [step for step in plan if llm.invoke(f"Can the following step of a plan: '{step}' be decomposed into a combination of the following actions: {[t.name for t in tools]}? Answer with `Yes` or `No`.").content.lower().startswith('yes')]
        print(f"Thought: {thought}\nPlan:\n"+"\n".join([f"-{step}" for step in plan]))
    else:
        plan = [command,]
    for step in plan:
        position = robot_interface.last_event.metadata['agent']['position']
        if use_planner and max_depth > 0 and not llm.invoke(f"Can the following step of a plan: '{step}' be implemented using only one of the following actions: {[t.name for t in tools]}? Answer with `Yes` or `No`.").content.lower().startswith('yes'):
            print(run_custom_agent(command=step, max_depth=max_depth-1))
        else:
            '''waypoints = get_waypoints()
            environment_description = {}
            wp_locations = []
            for i, wp in enumerate(waypoints):
                split_wp = wp.split('|')
                wp_name = split_wp[0]
                wp_x = float(split_wp[1].replace(',','.'))
                wp_y = float(split_wp[2].replace(',','.'))
                wp_z = float(split_wp[3].replace(',','.'))
                wp_locations.append({
                    'name': wp_name,
                    'position': {
                        'x': wp_x,
                        'y': wp_y,
                        'z': wp_z
                    }
                })
            next_to = []
            for wp1 in wp_locations:
                for wp2 in wp_locations:
                    if wp2 != wp1 and math.sqrt(math.pow(wp1['position']['x']-wp2['position']['x'], 2)+math.pow(wp1['position']['z']-wp2['position']['z'],2)) < 0.25:
                        if (wp2['name'],wp1['name']) not in next_to:
                            next_to.append((wp1['name'], wp2['name']))'''
            #waypoints = list(set([wp['name'] for wp in wp_locations]))
            #relevant_waypoints = [d.page_content for d in waypoint_store.as_retriever().invoke(step)]
            #waypoints = [wp for wp in waypoints if wp.split('_')[0] in relevant_waypoints]
            #next_to = list(set([nt for nt in next_to if nt[0].split('_')[0] in waypoints or nt[1].split('_')[0] in waypoints]))
            environment_description = {
                'waypoints': list(set([w.split('|')[0] for w in get_waypoints()])),
                #'reachable_from': next_to,
                'current_location': CURRENT_LOCATION,
                'object_in_hand': OBJECT_IN_HAND
            }
            print(environment_description)
            environment_description_cp = environment_description.copy()
            def mock_grasp(object_name: str):
                global environment_description_cp
                if not use_smart_tool:
                    if environment_description_cp['current_location'] != object_name:
                        raise ValueError(f"To grasp object {object_name} you must first navigate to waypoint {object_name}")
                    if environment_description_cp['object_in_hand'] != None:
                        raise ValueError(f"You cannot grasp object {object_name} because you already have an {environment_description_cp['object_in_hand']} in your hand")                
                environment_description_cp['object_in_hand'] = environment_description_cp['waypoints'].pop(environment_description_cp['waypoints'].index(object_name))
            def mock_place(object_name: str, container_name: str):
                global environment_description_cp
                if not use_smart_tool:
                    if environment_description_cp['current_location'] != container_name:
                        raise ValueError(f"To place object {object_name} on container {container_name} you must first navigate to waypoint {container_name}")
                    if environment_description_cp['object_in_hand'] != object_name:
                        raise ValueError(f"You cannot place object {object_name} on container {container_name} because your object in hand is: {environment_description_cp['object_in_hand']}")                
                environment_description_cp['waypoints'].append(environment_description_cp['object_in_hand'])
                environment_description_cp['object_in_hand'] = None
            def mock_navigate(waypoint_name: str):
                global environment_description_cp
                if not use_smart_tool:
                    if waypoint_name not in environment_description_cp['waypoints']:
                        raise ValueError(f"You cannot navigate to waypoint {waypoint_name}. Valid waypoints are: {environment_description_cp['waypoints']}")
                environment_description_cp['current_location'] = waypoint_name
            world_model = PythonREPLTool()
            funcs = {
                'grasp': mock_grasp,
                'place_on': mock_place,
                'navigate_to_waypoint': mock_navigate
            }
            world_model.python_repl.locals = funcs
            world_model.python_repl.globals = funcs
            messages_history = [dict(role='user', content=step),]
            #exit()
            code = ''
            use_code = False
            for i in range(world_model_iterations):
                environment_description_cp = environment_description.copy()
                if use_context == False:
                    if not 'fbot' in llm_api_model:
                        executor_prompt = f"Imagine you are a service robot in a domestic environment. You must answer by providing python code for performing tasks.\nAfter the code is executed, everything printed by it will be visible to you, so you should print relevant information to gather feedback.\nConsider the following python functions available inside the Python REPL that you can use: {tools_prompt}\nThe following objects are in the house: {environment_description}\nYou are currently at the following position: {CURRENT_LOCATION}\nYou must be within 1m of reach of any object in order to manipulate it\nAnswer in the following format:\n\nThought: ...(a brief summary that will help you accomplish the task)\nCode: ```python\n...\n```\n\n"
                    else:
                        executor_prompt = f"You are a domestic service robot that can execute the following actions: {[{'name': tool.name, 'description': tool.description} for tool in tools]}.\nEnvironment description: {environment_description}\nAnswer in the following format:\n\nThought: ...(a brief summary that will help you accomplish the task)\nCode: ```python\n...\n```"
                elif use_context == True:
                    executor_prompt = f"Imagine you are a service robot in a domestic environment. You must answer by providing python code for performing tasks.\nAfter the code is executed, everything printed by it will be visible to you, so you should print relevant information to gather feedback.\nConsider the following python functions available inside the Python REPL that you can use: {tools_prompt}\nThe following objects are in the house: {environment_description}\nYou are currently at the following position: {CURRENT_LOCATION}\nYou must be within 1m of reach of any object in order to manipulate it\nAnswer in the following format:\n\nThought: ...(a brief summary that will help you accomplish the task)\nTask Complete: ...(must be either `Yes` or `No`)\nCode: ```python\n...\n```\n\nConsider the following context about the task: {command}"
                if use_image == True:
                    image = Image.fromarray(robot_interface.last_event.frame)
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    image = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()
                    vlm_prompt = {
                        'role': 'user',
                        'content': [
                            {
                                "type": "text",
                                "text": "Describe the objects in detail"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image
                                }
                            }
                        ]
                    }
                    response = vlm.invoke([vlm_prompt,]).content
                    executor_prompt += f"\nDescription of the objects in your camera view: {response}"
                response = llm.invoke([dict(role='system', content=executor_prompt), *messages_history])
                messages_history.append(response)
                response = response.content
                thought = response.split('\n')[0].strip('Thought:').strip(' ')
                if 'Code:' in response:
                    code = response.split('```python\n')[-1].split('\n```')[0]
                    iteration = f"\nThought: {thought}\nCode: ```python\n{code}\n```\n"
                    print(iteration)
                    observation = world_model(tool_input=code)
                    if 'Error' not in observation:
                        use_code = True
                        break
                    iteration_observation = f'Observation: {observation}'
                    print(iteration_observation)
                    iteration += iteration_observation
                    history += iteration
                    messages_history.append(dict(role='user', content=iteration_observation))
                else:
                    iteration = f"Error! Wrong format for response: '{response}'!"
                    print(iteration)
                    history += iteration
                    messages_history.append(dict(role='user', content=iteration))
            if use_code and not save_dataset:
                print("Real observation: " + python_tool(tool_input=code))
    return environment_description, thought, code, use_code

if __name__ == '__main__':
    rospy.init_node('crewai_agent_node', anonymous=True)
    environment_description_cp = {}
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
    robot_name = rospy.get_param('~robot_name', "DoRIS")
    llm_api_model = rospy.get_param('~llm_api_model', 'gpt-3.5-turbo')
    llm_api_base_url = rospy.get_param('~llm_api_base_url', None)
    use_context = rospy.get_param('~use_context', True)
    use_planner = rospy.get_param('~use_planner', True)
    use_smart_tool = rospy.get_param('~use_smart_tool', True)
    use_image = rospy.get_param('~use_image', True)
    benchmark = rospy.get_param('~benchmark', False)
    benchmark_oracle = rospy.get_param('~benchmark_oracle', False)
    benchmark_iterations = rospy.get_param('~benchmark_iterations', 10)
    world_model_iterations = rospy.get_param('~world_model_iterations', 10)
    save_dataset = rospy.get_param('~save_dataset', False)
    subsplit = rospy.get_param('~subsplit', 'train')
    llm_api_type = rospy.get_param('~llm_api_type', 'google')
    if save_dataset == True:
        dataset = []
    scene_dataset = prior.load_dataset('procthor-10k')
    context = None
    execute_tasks_server = rospy.Service('/fbot_llm_agent/execute_tasks', ExecuteTasks, handler=handle_execute_tasks)
    robot_interface = Controller(agentMode="arm", width=640, height=480, renderInstanceSegmentation=True)
    robot_interface.step(action='Done')
    robot_interface.step(action="SetHandSphereRadius", radius=0.1)
    robot_interface.step(
                action="MoveArmBase",
                y=0.8,
                returnToStart=False,
            )
    waypoint_store = Chroma.from_documents([Document(page_content=o.split('|')[0], metadata=dict(objId=o)) for o in get_waypoints()], embeddings, collection_name="waypoints", persist_directory=False)
    tools = [
        StructuredTool.from_function(grasp),
        StructuredTool.from_function(place_on),
        StructuredTool.from_function(navigate_to_waypoint),
        #StructuredTool.from_function(query_perception_system),
        #StructuredTool.from_function(query_semantic_memory),
    ]
    python_tool = PythonREPLTool()
    python_tool.python_repl.locals = dict([(t.name, t.func) for t in tools])
    python_tool.python_repl.globals = dict([(t.name, t.func) for t in tools])
    python_tool.description += f"The following functions are available inside the Python REPL:\n" + '\n'.join([f"{t.name}: {t.description}" for t in tools])
    #python_tool.description += f"\nThe input to this tool must be a JSON with a field `code` with the python code to be executed."
    tool_store = Chroma.from_documents([Document(page_content=tool.description, metadata={'name': tool.name}) for tool in tools], embeddings, collection_name="tools")
    print(tool_store)
    if llm_api_type == 'openai':
        llm = ChatOpenAI(model=llm_api_model, base_url=llm_api_base_url, temperature=0.0)
    elif llm_api_type == 'groq':
        llm = ChatGroq(model_name=llm_api_model, temperature=0)
    elif llm_api_type == 'ollama':
        llm = ChatOllama(model=llm_api_model, temperature=0)
    elif llm_api_type == 'google':
        llm = ChatGoogleGenerativeAI(model=llm_api_model, temperature=0.0, convert_system_message_to_human=True)
        vlm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
    #navigate_to_waypoint(waypoint_name='stove burner')
    #navigate_to_waypoint(waypoint_name='stove')
    #print(navigate_to_waypoint('fridge'))
    
    if benchmark == True:
        n_success = 0
        #print([o['objectId'] for o in robot_interface.last_event.metadata['objects']])
        #furnitures = ['stove burner', 'sink']
        #objects = ['apple', 'tomato', 'salt', 'pan', 'pot', 'statue']
        
        random.seed(123)
        #random.shuffle(furnitures)
        #random.shuffle(objects)
        for i in tqdm.trange(benchmark_iterations):
            try:
                house = scene_dataset[subsplit][i]
                waypoint_store = Chroma.from_documents([Document(page_content=o.split('|')[0], metadata=dict(objId=o)) for o in get_waypoints() if not o.split('|')[0].islower()], embeddings, collection_name=f"waypoints_{i}", persist_directory=False)
                robot_interface.reset(scene=house)
                robot_interface.step(action='Done')
                robot_interface.step(
                    action="MoveArmBase",
                    y=0.8,
                    returnToStart=False,
                )
                robot_interface.step(
                    action="MoveArm",
                    position=dict(x=0.0, y=0.0, z=0.0),
                    coordinateSpace="armBase",
                )
                robot_interface.step(
                    action="SetHandSphereRadius",
                    radius=0.5,
                )
                CURRENT_LOCATION = None
                OBJECT_IN_HAND = None
                obj = random.choice([obj['objectId'].split('|')[0] for obj in robot_interface.last_event.metadata['objects'] if obj['pickupable'] == True and not obj['objectId'].startswith('room')])
                destination = random.choice([obj['objectId'].split('|')[0] for obj in robot_interface.last_event.metadata['objects'] if obj['receptacle'] == True and not obj['objectId'].startswith('room')])
                #print(robot_interface.last_event.metadata['objects'][0]['objectId'])
                #exit()
                if benchmark_oracle == False:
                    instruction = f"put the {obj} on the {destination}"
                    obj = [(d.page_content, d.metadata) for d in waypoint_store.as_retriever().invoke(obj)][0]
                    destination = [(d.page_content, d.metadata) for d in waypoint_store.as_retriever().invoke(destination)][0]
                    #req = ExecuteTasksRequest()
                    #task_msg = TaskMsg()
                    #task_msg.description = instruction
                    #req.task_list.append(task_msg)
                    #res = handle_execute_tasks(req)
                    #print(res.response)
                    try:
                        environment_description, thought, code, use_code = run_custom_agent(instruction, max_depth=0)
                    except BaseException as e:
                        #raise e
                        continue
                else:
                    try:
                        print(place_on(object_name=obj, container_name=destination))
                    except:
                        continue
                    obj = [(d.page_content, d.metadata) for d in waypoint_store.as_retriever().invoke(obj)][0]
                    destination = [(d.page_content, d.metadata) for d in waypoint_store.as_retriever().invoke(destination)][0]
                #print(obj)
                #print(destination)
                all_objects = robot_interface.last_event.metadata['objects']
                #print(obj[1]['objId'])
                #print([o for o in all_objects if o['objectId'].split('|')[0] == obj[1]['objId'].split('|')[0]])
                #print(all_objects)
                obj_pos = [o for o in all_objects if o['objectId'].split('|')[0] == obj[1]['objId'].split('|')[0]][0]['position']
                destination_pos = [o for o in all_objects if o['objectId'].split('|')[0] == destination[1]['objId'].split('|')[0]][0]['position']
                distance = math.sqrt(math.pow(obj_pos['x']-destination_pos['x'],2)+math.pow(obj_pos['z']-destination_pos['z'],2))
                if distance < 0.5:
                    success = True
                else:
                    success = False
                if success:
                    n_success += 1
                if save_dataset and use_code == True:
                    dataset.append({
                            'conversations': [
                                {
                                    'from': 'system',
                                    'value': f"You are a domestic service robot that can execute the following actions: {[{'name': tool.name, 'description': tool.description} for tool in tools]}.\nEnvironment description: {environment_description}"
                                },
                                {
                                    'from': 'human',
                                    'value': instruction
                                },
                                {
                                    'from': 'gpt',
                                    'value': f"Thought: {thought}\nCode: ```python\n{code}\n```"
                                }
                            ]
                        })
                print(f"Success rate: {n_success/(i+1)}")
                #wandb.log({'success_rate': n_success/(i+1)})
            except:
                pass
    if save_dataset == True:
        df = pd.DataFrame.from_records(dataset)
        df.to_json('/home/cris/catkin_ws/dataset.jsonl', orient='records', lines=True)
    rospy.spin()