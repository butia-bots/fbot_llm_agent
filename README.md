# Install

Follow [this guide](https://github.com/butia-bots/butia_vision/blob/feature/vlm-recognition/butia_recognition/scripts/butia_recognition/paligemma_recognition/README.md) for installing the `feature/vlm-recognition` branch of butia_vision.

Install dependencies on the Intel NUC with pip

```sh
pip install -r requirements.txt
```

# How to Use

Edit `config/hf_react_code_agent.yaml` and update the parameters if required.

Make sure the arm, navigation stack, [object_recognition](https://github.com/butia-bots/butia_vision/blob/feature/vlm-recognition/butia_recognition/scripts/butia_recognition/paligemma_recognition/paligemma_recognition.py), image2world, butia_world and doris_face have been launched and are running.

Launch autogen agent node

```sh
roslaunch fbot_llm_agent autogen_agent.launch
```

Then call the `/fbot_llm_agent/execute_tasks` service, passing a list of tasks for the agent to execute.
