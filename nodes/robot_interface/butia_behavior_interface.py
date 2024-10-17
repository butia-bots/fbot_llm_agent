from butia_behavior.machines.gpsr_move_to import getGPSRMoveToMachine
from butia_behavior.machines.gpsr_follow import getGPSRFollowMachine
from butia_behavior.machines.gpsr_answer import getGPSRAnswerMachine
from butia_behavior.machines.gpsr_visual_question_answering import getGPSRVisualQuestionAnsweringMachine
from butia_behavior.machines.gpsr_speak import getGPSRSpeakMachine
from butia_behavior.machines.goto_fixed import getGoToFixedMachine
from robot_interface.robot_tool import RobotTool
import smach
from typing import List, Callable

class ButiaBehaviorInterface:
    def __init__(self, fake_execution=False):
        self.fake_execution = fake_execution

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
    
    def get_code_tools_hf(self)->List[RobotTool]:
        """Get code tools for huggingface agents"""
        return [
            RobotTool.from_function(self.move_to, 'move_to', str),
            RobotTool.from_function(self.follow, 'follow', str),
            RobotTool.from_function(self.answer, 'answer', str),
            RobotTool.from_function(self.visual_question_answering, 'visual_question_answering', str),
            RobotTool.from_function(self.speak, 'speak', str)
        ]
