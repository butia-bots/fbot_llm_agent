# Install

Install Ollama on the Jetson Orin AGX (ignore the 404 error)

```sh
curl https://ollama.com/install.sh | sh
```

On the Jetson Orin AGX, pull the `llava-llama3` from the Ollama repository

```sh
ollama pull llava-llama3
```

Install dependencies on the Intel NUC with pip (might require a python 3.10 conda environment)

```sh
pip install -r requirements.txt
```

# How to Use

Edit `config/vlm_agent.yaml` and update the parameters if required.

Change the "localhost" in the parameter `llm_api_base_url` to the IP adress of the Jetson Orin AGX.

Export the API keys as environment variables (also add the lines to the .zshrc and .bashrc files in the home directory)

```sh
export OPENAI_API_KEY=...(place a dummy value if using local model)
export LANGCHAIN_API_KEY=...(configure this for logging agent runs to LangSmith)
```

Make sure the arm, navigation stack, object_recognition, image2world, butia_world and doris_face have been launched and are running.

Launch VLM agent node

```sh
roslaunch fbot_llm_agent vlm_agent.launch
```

Then call the `/fbot_llm_agent/execute_tasks` service, passing a list of tasks for the agent to execute.