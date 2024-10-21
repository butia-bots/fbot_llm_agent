from butia_behavior.machines.gpsr_move_to import getGPSRMoveToMachine
from butia_behavior.machines.gpsr_follow import getGPSRFollowMachine
from butia_behavior.machines.gpsr_answer import getGPSRAnswerMachine
from butia_behavior.machines.gpsr_visual_question_answering import getGPSRVisualQuestionAnsweringMachine
from butia_behavior.machines.gpsr_speak import getGPSRSpeakMachine
from butia_behavior.machines.gpsr_pick import getGPSRPickMachine
from butia_behavior.machines.gpsr_place import getGPSRPlaceMachine
from butia_behavior.machines.gpsr_caption import getGPSRCaptionMachine
from butia_behavior.machines.gpsr_query_spatial_memory import getGPSRQuerySpatialMemoryMachine
from butia_behavior.machines.gpsr_move_to_pose import getGPSRMoveToPoseMachine
from butia_behavior.machines.goto_fixed import getGoToFixedMachine
from robot_interface.robot_tool import RobotTool
import smach
from typing import List, Callable
import json
from collections import Counter
from butia_vision_msgs.msg import Recognitions2D

class ButiaBehaviorInterface:
    def __init__(self, fake_execution=False):
        self.fake_execution = fake_execution

    def query_spatial_memory(self, query: str):
        """Query a static spacial memory with information about locations in the environment"""
        sm = getGPSRQuerySpatialMemoryMachine(query=query)
        outcome = sm.execute()
        results = sm.userdata['results']
        return [r.properties for r in results]

    def move_to_pose(self, x: float, y: float, yaw: float):
        """Navigate to a pose in the environment and return mission status"""
        sm = getGPSRMoveToPoseMachine(x=x, y=y, yaw=yaw)
        outcome = sm.execute()
        return outcome

    def move_to(self, destination: str)->str:
        """Navigate to a destination and return mission status"""
        if self.fake_execution:
            return 'succeeded'
        sm = getGPSRMoveToMachine(target=destination)
        outcome = sm.execute()
        return outcome
    
    def follow(self)->str:
        """Follow a person and return mission status"""
        if self.fake_execution:
            return 'succeeded'
        sm = getGPSRFollowMachine()
        outcome = sm.execute()
        return outcome
    
    def answer(self)->str:
        """Answer a question and return mission status"""
        if self.fake_execution:
            return 'I do not know'
        sm = getGPSRAnswerMachine()
        outcome = sm.execute()
        return outcome
    
    def visual_question_answering(self, question: str)->str:
        """Answer a visual question and return mission status"""
        if self.fake_execution:
            return 'I do not know'
        sm = getGPSRVisualQuestionAnsweringMachine(question)
        outcome = sm.execute()
        return outcome
    
    def speak(self, utterance: str)->str:
        """Speak an utterance and return mission status"""
        if self.fake_execution:
            return 'succeeded'
        sm = getGPSRSpeakMachine(utterance)
        outcome = sm.execute()
        return outcome

    def pick(self, object: str)->str:
        """Pick an object and return mission status"""
        if self.fake_execution:
            return 'succeeded'
        sm = getGPSRPickMachine(obj=object)
        outcome = sm.execute()
        return outcome

    def place(self, placement_zone: str)->str:
        """Place previously picked object down and return mission status"""
        if self.fake_execution:
            return 'succeeded'
        sm = getGPSRPlaceMachine(obj=placement_zone)
        outcome = sm.execute()
        return outcome

    def caption(self):
        """Describes the content of the egocentric camera observation"""
        if self.fake_execution:
            return {
                'caption': 'Nothing visible here!',
            }
        sm = getGPSRCaptionMachine()
        outcome = sm.execute()
        caption: str = sm.userdata['caption']
        return {
            'caption': caption,
        }
    
    def get_code_tools_hf(self)->List[RobotTool]:
        """Get code tools for huggingface agents"""
        return [
            RobotTool.from_function(self.move_to, 'move_to', str),
            RobotTool.from_function(self.follow, 'follow', str),
            RobotTool.from_function(self.answer, 'answer', str),
            RobotTool.from_function(self.visual_question_answering, 'visual_question_answering', str),
            RobotTool.from_function(self.speak, 'speak', str)
        ]
