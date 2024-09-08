import rospy
from .llm_engine_quiz import OllamaEngineQuiz

class LLMContextQuiz:
    def __init__(self, llm_model: str) -> None:
        self.llm_engine = OllamaEngineQuiz(model=llm_model)

    def run(self):
        rospy.Service('llm_context_quiz', LLMContextQuizSrv, self.handle_llm_context_quiz)
        rospy.loginfo('LLM Context Quiz service is running')
        rospy.spin()
    
    def receiveFromRag(self, request):
        #Todo
        return request

    def handle_llm_context_quiz(self, request):
        question = request.question
        rag_question = self.receiveFromRag(question)
        response = self.llm_engine(messages=rag_question)
        return response