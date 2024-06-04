#!/usr/bin/env python3

from crewai import Agent, Crew, Process, Task
from fbot_llm_agent.srv import ExecuteTasks, ExecuteTasksRequest, ExecuteTasksResponse
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from ai2thor.controller import Controller
import rospy
from typing import List
from Levenshtein import distance
import random

OBJECT_IN_HAND = 'None'
CURRENT_LOCATION = 'None'

def open_object(object_name: str)->str:
    """Open the object with given object_name"""
    object_id = object_name
    navigate_to_waypoint(object_id)
    global robot_interface
    object_store = Chroma.from_documents([Document(page_content=o.split('|')[0], metadata=dict(objId=o)) for o in observe()], embeddings)
    doc = object_store.as_retriever().invoke(object_id)[0]
    object_id = doc.metadata['objId']
    object_name = doc.page_content
    for i in range(10):
        robot_interface.step(
            action="OpenObject",
            objectId=object_id,
            forceAction=False,
            openness=1,
        )
    return f"Opened {object_name}"

def close_object(object_name: str)->str:
    """Close the object with given object_name"""
    object_id = object_name
    navigate_to_waypoint(object_id)
    global robot_interface
    object_store = Chroma.from_documents([Document(page_content=o.split('|')[0], metadata=dict(objId=o)) for o in observe()], embeddings)
    doc = object_store.as_retriever().invoke(object_id)[0]
    object_id = doc.metadata['objId']
    object_name = doc.page_content
    for i in range(10):
        robot_interface.step(
            action="OpenObject",
            objectId=object_id,
            forceAction=False,
            openness=0,
        )
    return f"Closed {object_name}"

def cook(object_name: str)->str:
    """Cook the object with given object_name"""
    object_id = object_name
    navigate_to_waypoint(object_id)
    global robot_interface
    object_store = Chroma.from_documents([Document(page_content=o.split('|')[0], metadata=dict(objId=o)) for o in observe()], embeddings)
    doc = object_store.as_retriever().invoke(object_id)[0]
    object_id = doc.metadata['objId']
    object_name = doc.page_content
    for i in range(10):
        robot_interface.step(
            action="CookObject",
            objectId=object_id,
            forceAction=False,
        )
    return f"Cooked {object_name}"

def slice_object(object_name: str)->str:
    """Slice the object with given object_name"""
    object_id = object_name
    navigate_to_waypoint(object_id)
    global robot_interface
    object_store = Chroma.from_documents([Document(page_content=o.split('|')[0], metadata=dict(objId=o)) for o in observe()], embeddings)
    doc = object_store.as_retriever().invoke(object_id)[0]
    object_id = doc.metadata['objId']
    object_name = doc.page_content
    for i in range(10):
        robot_interface.step(
            action="SliceObject",
            objectId=object_id,
            forceAction=False,
        )
    return f"Sliced {object_name}"

def toggle_on(object_name: str)->str:
    """Toggle on the object with given object_name"""
    object_id = object_name
    navigate_to_waypoint(object_id)
    global robot_interface
    object_store = Chroma.from_documents([Document(page_content=o.split('|')[0], metadata=dict(objId=o)) for o in observe()], embeddings)
    doc = object_store.as_retriever().invoke(object_id)[0]
    object_id = doc.metadata['objId']
    object_name = doc.page_content
    for i in range(10):
        robot_interface.step(
            action="ToggleObjectOn",
            objectId=object_id,
            forceAction=False,
        )
    return f"Toggled on {object_name}"

def toggle_off(object_name: str)->str:
    """Toggle off the object with given object_name"""
    object_id = object_name
    navigate_to_waypoint(object_id)
    global robot_interface
    object_store = Chroma.from_documents([Document(page_content=o.split('|')[0], metadata=dict(objId=o)) for o in observe()], embeddings)
    doc = object_store.as_retriever().invoke(object_id)[0]
    object_id = doc.metadata['objId']
    object_name = doc.page_content
    for i in range(10):
        robot_interface.step(
            action="ToggleObjectOff",
            objectId=object_id,
            forceAction=False,
        )
    return f"Toggled off {object_name}"

def clean(object_name: str)->str:
    """Clean the object with given object_name"""
    object_id = object_name
    navigate_to_waypoint(object_id)
    global robot_interface
    object_store = Chroma.from_documents([Document(page_content=o.split('|')[0], metadata=dict(objId=o)) for o in observe()], embeddings)
    doc = object_store.as_retriever().invoke(object_id)[0]
    object_id = doc.metadata['objId']
    object_name = doc.page_content
    for i in range(10):
        robot_interface.step(
            action="CleanObject",
            objectId=object_id,
            forceAction=False,
        )
    return f"Cleaned {object_name}"

def fill_with_liquid(object_name: str, liquid_name: str)->str:
    """Fill with liquid given as liquid_name the object with given object_name"""
    object_id = object_name
    navigate_to_waypoint(object_id)
    global robot_interface
    object_store = Chroma.from_documents([Document(page_content=o.split('|')[0], metadata=dict(objId=o)) for o in observe()], embeddings)
    doc = object_store.as_retriever().invoke(object_id)[0]
    object_id = doc.metadata['objId']
    object_name = doc.page_content
    for i in range(10):
        robot_interface.step(
            action="FillObjectWithLiquid",
            objectId=object_id,
            fillLiquid=liquid_name,
            forceAction=False,
        )
    return f"Filled {object_name} with {liquid_name}"

def empty_liquid(object_name: str)->str:
    """Empty the liquid of the object with given object_name"""
    object_id = object_name
    navigate_to_waypoint(object_id)
    global robot_interface
    object_store = Chroma.from_documents([Document(page_content=o.split('|')[0], metadata=dict(objId=o)) for o in observe()], embeddings)
    doc = object_store.as_retriever().invoke(object_id)[0]
    object_id = doc.metadata['objId']
    object_name = doc.page_content
    for i in range(10):
        robot_interface.step(
            action="EmptyLiquidFromObject",
            objectId=object_id,
            forceAction=False,
        )
    return f"Emptied liquid from {object_name}"

def use_object(object_name: str)->str:
    """Uses the object with given object_name"""
    object_id = object_name
    navigate_to_waypoint(object_id)
    global robot_interface
    object_store = Chroma.from_documents([Document(page_content=o.split('|')[0], metadata=dict(objId=o)) for o in observe()], embeddings)
    doc = object_store.as_retriever().invoke(object_id)[0]
    object_id = doc.metadata['objId']
    object_name = doc.page_content
    for i in range(10):
        robot_interface.step(
            action="UseUpObject",
            objectId=object_id,
            forceAction=False,
        )
    return f"Used up {object_name}"

def grasp(object_name: str)->str:
    """Grasp the object with given object_name"""
    global OBJECT_IN_HAND
    object_id = object_name
    if OBJECT_IN_HAND != object_id:
        navigate_to_waypoint(object_id)
    else:
        return f"Grasped {object_id}"
    global robot_interface
    object_store = Chroma.from_documents([Document(page_content=o.split('|')[0], metadata=dict(objId=o)) for o in observe()], embeddings)
    doc = object_store.as_retriever().invoke(object_id)[0]
    object_id = doc.metadata['objId']
    object_name = doc.page_content
    for i in range(10):
        robot_interface.step(
            action="PickupObject",
            objectId=object_id,
            forceAction=False,
            manualInteract=False
        )
    OBJECT_IN_HAND = object_name
    return f"Grasped {object_name}"

def place_on(object_name: str, container_name: str)->str:
    """Place previously grasped object with object_name on container with given container_name"""
    global OBJECT_IN_HAND
    if object_name != OBJECT_IN_HAND:
        grasp(object_name)
    container_id = container_name
    navigate_to_waypoint(container_id)
    global robot_interface
    object_store = Chroma.from_documents([Document(page_content=o.split('|')[0], metadata=dict(objId=o)) for o in observe()], embeddings)
    doc = object_store.as_retriever().invoke(container_id)[0]
    container_id = doc.metadata['objId']
    container_name = doc.page_content
    for i in range(10):
        robot_interface.step(
            action="PutObject",
            objectId=container_id,
            forceAction=True,
            placeStationary=False
        )
    object_name = OBJECT_IN_HAND
    OBJECT_IN_HAND = 'None'
    return f"Placed {object_name} on {container_name}"

def navigate_to_waypoint(waypoint_name: str)->str:
    """Navigate to waypoint with given waypoint_name"""
    global robot_interface
    global CURRENT_LOCATION
    global OBJECT_IN_HAND
    if waypoint_name == OBJECT_IN_HAND:
        return f"Navigated to {OBJECT_IN_HAND}"
    waypoint_id = waypoint_name
    doc = waypoint_store.as_retriever().invoke(waypoint_id)[0]
    waypoint_id = doc.metadata['objId']
    waypoint_name = doc.page_content
    poses = robot_interface.step(
        action="GetInteractablePoses",
        objectId=waypoint_id,
    ).metadata["actionReturn"]
    if len(poses) > 0:
        pose = poses[0]
        for i in range(10):
            robot_interface.step("TeleportFull", **pose)
    CURRENT_LOCATION = waypoint_name
    return f"Navigated to waypoint {waypoint_name}"

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

def handle_execute_tasks(req: ExecuteTasksRequest):
    global CURRENT_LOCATION
    global OBJECT_IN_HAND
    tasks = []
    for task_msg in req.task_list:
        task = Task(
            description=task_msg.description+f"\nYour current location is: {CURRENT_LOCATION}\nYour current object in hand is: {OBJECT_IN_HAND}",
            agent=robot,
            expected_output='A message indicating the task has been successfully executed'
        )
        tasks.append(task)
    crew = Crew(
        tasks=tasks,
        agents=[robot,],
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff()
    res = ExecuteTasksResponse()
    res.response = result
    return res

if __name__ == '__main__':
    rospy.init_node('crewai_agent_node', anonymous=True)
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
    robot_name = rospy.get_param('~robot_name', "DoRIS")
    llm_api_model = rospy.get_param('~llm_api_model', 'gpt-3.5-turbo')
    llm_api_base_url = rospy.get_param('~llm_api_base_url', None)
    execute_tasks_server = rospy.Service('/fbot_llm_agent/execute_tasks', ExecuteTasks, handler=handle_execute_tasks)
    robot_interface = Controller(width=640, height=480, renderInstanceSegmentation=True)
    robot_interface.step(action='Done')
    waypoint_store = Chroma.from_documents([Document(page_content=o.split('|')[0], metadata=dict(objId=o)) for o in get_waypoints()], embeddings)

    tools = [
        StructuredTool.from_function(open_object),
        StructuredTool.from_function(close_object),
        StructuredTool.from_function(cook),
        StructuredTool.from_function(slice_object),
        StructuredTool.from_function(toggle_on),
        StructuredTool.from_function(toggle_off),
        StructuredTool.from_function(fill_with_liquid),
        StructuredTool.from_function(empty_liquid),
        StructuredTool.from_function(use_object),
        StructuredTool.from_function(grasp),
        StructuredTool.from_function(place_on),
        StructuredTool.from_function(navigate_to_waypoint),
        StructuredTool.from_function(query_perception_system),
        StructuredTool.from_function(query_semantic_memory),
    ]

    llm = ChatOpenAI(model=llm_api_model, base_url=llm_api_base_url)

    robot = Agent(
        role='Service Robot',
        goal='Help with daily tasks inside the house',
        backstory=f'You are {robot_name}, a domestic service robot developed by team Fbot@Home.',
        llm=llm,
        tools=tools,
        memory=False,
        verbose=True
    )

    rospy.spin()