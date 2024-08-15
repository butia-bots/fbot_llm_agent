from openai import OpenAI
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_tool_choice_option_param import ChatCompletionNamedToolChoiceParam
import instructor
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Union
import json
#from butia_behavior.machines.gpsr_move_to import getGPSRMoveToMachine
#from butia_behavior.machines.gpsr_follow import getGPSRFollowMachine
#from butia_behavior.machines.gpsr_answer import getGPSRAnswerMachine
#from butia_behavior.machines.gpsr_visual_question_answering import getGPSRVisualQuestionAnsweringMachine
#from butia_behavior.machines.gpsr_speak import getGPSRSpeakMachine

class MoveTo(BaseModel):
    stage_type: Literal["move_to"]
    destination: str

class Follow(BaseModel):
    stage_type: Literal["follow"]
    person: str

class Speak(BaseModel):
    stage_type: Literal["speak"]
    utterance: str

class VisualQuestionAnswering(BaseModel):
    stage_type: Literal["visual_question_answering"]
    question: str

class Answer(BaseModel):
    stage_type: Literal["answer"]

class Grasp(BaseModel):
    stage_type: Literal["grasp"]
    object: str

class PassTo(BaseModel):
    stage_type: Literal["pass_to"]
    object_destination: str

class Plan(BaseModel):
    plan: str = Field(description="A step-by-step plan that will allow a service robot to solve the task.")
    #stages: List[Union[MoveTo, Follow, Speak, VisualQuestionAnswering, Answer]] = Field(description="Each stage of the plan.")

'''STAGE_TO_SM = {
    'move_to': getGPSRMoveToMachine,
    'follow': getGPSRFollowMachine,
    'speak': getGPSRSpeakMachine,
    'visual_question_answering': getGPSRVisualQuestionAnsweringMachine,
    'answer': getGPSRAnswerMachine
}'''

class FbotLLMAgent:
    def __init__(self):
        self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
        )

    def execute_plan(self, plan: Plan):
        pass

    def compute_plan(self, command: str):
        plan = Plan(**json.loads(self.client.chat.completions.create(
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a general purpose service robot'
                },
                {
                    'role': 'user',
                    'content': command
                }
            ],
            model='llama3.1:8b-instruct-q2_K',
            tools=[
                {
                    'type': 'function',
                    'function': {
                        'name': 'execute_general_purpose_service_robot_task',
                        'description': 'Executes a general purpose service robot task',
                        'parameters': Plan.model_json_schema()
                    }
                }
            ],
            tool_choice={
                'type': 'function',
                'function': {
                    'name': 'execute_general_purpose_service_robot_task'
                }
            }
        ).choices[0].message.tool_calls[0].function.arguments))
        return plan

if __name__ == '__main__':
    agent = FbotLLMAgent()
    print(agent.compute_plan("Tell me how many people are waving in the kitchen"))