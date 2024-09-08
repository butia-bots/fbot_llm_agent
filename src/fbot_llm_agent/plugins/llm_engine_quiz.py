import rospy
import subprocess
from openai import OpenAI

class OllamaEngineQuiz:
    def __init__(self, model: str) -> None:
        try:
            subprocess.run(["jetson-container", "run", "dustynv/ollama:r36.2.0", "/bin/ollama", "run", llm_model])
            rospy.loginfo("Started Ollama container with model: " + llm_model)
        except Exception as e:
            rospy.logerr("Failed to start Ollama container with model: " + llm_model)
            rospy.logerr(e)
            rospy.signal_shutdown("Failed to start Ollama container")
            
        self.client = OpenAI(api_key='ollama', base_url='http://localhost:11434/v1')
        self.model = model

    def __call__(self, messages) -> Any:
        return self.client.chat.completions.create(messages=messages, model=self.model, temperature=0).choices[0].message.content