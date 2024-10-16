#!/usr/bin/env python3
from robot_interface.butia_behavior_interface import ButiaBehaviorInterface
import rospy

if __name__ == '__main__':
    robot_interface = ButiaBehaviorInterface(fake_execution=False)
    assert robot_interface.visual_question_answering(question='what do you see here?') == 'succeeded'