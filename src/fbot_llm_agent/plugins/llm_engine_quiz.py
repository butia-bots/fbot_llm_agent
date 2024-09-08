#import rospy
import subprocess
from openai import OpenAI
import time

class OllamaEngineQuiz:
    def __init__(self, model: str) -> None:
        self.model = model
        
        try:
            #subprocess.run(["jetson-container", "run", "dustynv/ollama:r36.2.0", "/bin/ollama", "run", llm_model])
            #docker run --gpus=all -d -v C:\\Users\\luisf\\.ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
            #docker exec -it ollama ollama run gemma2:2b-instruct-q5_1
            #subprocess.run(["docker", "run","--gpus=all","-d", "-v", "C:\\Users\\luisf\\.ollama:/root/.ollama", "-p", "11434:11434", "--name", "ollama", "ollama/ollama"])
            #subprocess.run(["docker", "exec","-it","ollama", "ollama", "run", self.model])
            #rospy.loginfo("Started Ollama container with model: " + llm_model)
            print("Started Ollama container with model: " + self.model)
        except Exception as e:
            rospy.logerr("Failed to start Ollama container with model: " + llm_model)
            rospy.logerr(e)
            rospy.signal_shutdown("Failed to start Ollama container")
            print(e)
            
        self.client = OpenAI(api_key='ollama', base_url='http://localhost:11434/v1/')

    def __call__(self, messages):
        return self.client.chat.completions.create(messages=messages, model=self.model, temperature=0).choices[0].message.content