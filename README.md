# Install

Install dependencies with pip (might require python 3.10)

```sh
pip install -r requirements.txt
```

# How to Use

Edit `config/vlm_agent.yaml` and update the parameters if required

Export the API keys as environment variables

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