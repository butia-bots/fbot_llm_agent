#!/usr/bin/env python3

import rospy
from fbot_llm_agent.plugins import LLMContextQuiz

if __name__ == '__main__':
    rospy.init_node('llm_context_quiz_node', anonymous=True)
    llm_model = rospy.get_param("llm_model")
    plugin = LLMContextQuiz(llm_model=llm_model)
    plugin.run()
