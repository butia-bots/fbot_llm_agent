from butia_behavior.machines.gpsr_move_to import getGPSRMoveToMachine
from butia_behavior.machines.gpsr_follow import getGPSRFollowMachine
from butia_behavior.machines.gpsr_answer import getGPSRAnswerMachine
from butia_behavior.machines.gpsr_visual_question_answering import getGPSRVisualQuestionAnsweringMachine
from butia_behavior.machines.gpsr_speak import getGPSRSpeakMachine
from robot_interface.robot_tool import RobotTool
from typing import List

class ButiaBehaviorInterface:
    def move_to(self, destination: str)->str:
        """Navigate to a destination"""
        sm = getGPSRMoveToMachine(destination)
        outcome = sm.execute()
        return outcome
    
    def follow(self)->str:
        """Follow a person"""
        sm = getGPSRFollowMachine()
        outcome = sm.execute()
        return outcome
    
    def answer(self)->str:
        """Answer a question"""
        sm = getGPSRAnswerMachine()
        outcome = sm.execute()
        return outcome
    
    def visual_question_answering(self, question: str)->str:
        """Answer a visual question"""
        sm = getGPSRVisualQuestionAnsweringMachine(question)
        outcome = sm.execute()
        return outcome
    
    def speak(self, utterance: str)->str:
        """Speak an utterance"""
        sm = getGPSRSpeakMachine(utterance)
        outcome = sm.execute()
        return outcome
    
    def get_code_tools_hf(self)->List[RobotTool]:
        """Get code tools for huggingface agents"""
        return [
            RobotTool.from_function(self.move_to, 'move_to', str, {'destination': {'type': 'str', 'required': True}}),
            RobotTool.from_function(self.follow, 'follow', str, {}),
            RobotTool.from_function(self.answer, 'answer', str, {}),
            RobotTool.from_function(self.visual_question_answering, 'visual_question_answering', str, {'question': {'type': 'str', 'required': True}}),
            RobotTool.from_function(self.speak, 'speak', str, {'utterance': {'type': 'str', 'required': True}})
        ]