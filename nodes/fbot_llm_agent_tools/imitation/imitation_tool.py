from langchain_core.tools import StructuredTool
from fbot_robot_learning.msg import PolicyInfo
from fbot_robot_learning.srv import ExecutePolicy, ExecutePolicyRequest
from butia_vision_msgs.msg import Recognitions3D, Description3D
from butia_vision_msgs.srv import SetClass, SetClassRequest
from geometry_msgs.msg import PoseStamped
from gym_fbot.arm.moveit_arm import MoveItArm
import rospy
from tf import TransformListener
from tf.transformations import euler_from_quaternion
from typing import Optional, Literal
import numpy as np
import math

recognitions3d = Recognitions3D()

def update_recognitions3d(msg: Recognitions3D):
    global recognitions3d
    recognitions3d = msg

def make_imitation_learning_tool(policy_info: PolicyInfo):
    set_class_service_proxy = rospy.ServiceProxy('/butia_vision/br/object_recognition/set_class', SetClass)
    execute_policy_service_proxy = rospy.ServiceProxy('/fbot_robot_learning/execute_policy', ExecutePolicy)
    recognitions_sub = rospy.Subscriber('/butia_vision/br/object_recognition3d', Recognitions3D, callback=update_recognitions3d)
    arm = MoveItArm()
    tfl = TransformListener()

    def imitation_learning_tool(object_to_approach: Optional[str]=None, sort_axis: Optional[Literal["x", "y", "z"]]=None, sort_order: Optional[Literal["ascending", "descending"]]=None, approach_distance_from_centroid: Optional[float]=None)->str:
        global recognitions3d
        global tfl
        descriptions = None
        if object_to_approach != None:
            set_class_service_proxy.call(class_name=object_to_approach)
            rospy.wait_for_message('/butia_vision/br/object_recognition3d')
            descriptions = filter(lambda d: d.label == object_to_approach, recognitions3d.descriptions)
        else:
            rospy.wait_for_message('/butia_vision/br/object_recognition3d')
            descriptions = recognitions3d.descriptions
        reverse = sort_order == 'descending'
        if sort_axis == 'x':
            descriptions = sorted(descriptions, key=lambda d: d.bbox.center.position.x, reverse=reverse)
        if sort_axis == 'y':
            descriptions = sorted(descriptions, key=lambda d: d.bbox.center.position.y, reverse=reverse)
        if sort_axis == 'z':
            descriptions = sorted(descriptions, key=lambda d: d.bbox.center.position.z, reverse=reverse)
        if object_to_approach != None and sort_axis != None:
            ps = PoseStamped()
            ps.header.frame_id = descriptions[0].header.frame_id
            ps.pose.position = descriptions[0].bbox.center.position
            ps.pose.orientation = descriptions[0].bbox.center.orientation
            tfl.waitForTransform(arm.move_group.get_pose_reference_frame(), ps.header.frame_id, rospy.Time(), rospy.Duration(10.0))
            ps = tfl.transformPose(arm.move_group.get_pose_reference_frame(), ps)
            rpy = euler_from_quaternion([
                ps.pose.orientation.x,
                ps.pose.orientation.y,
                ps.pose.orientation.z,
                ps.pose.orientation.w
            ])
            xyz = [
                ps.pose.position.x,
                ps.pose.position.y,
                ps.pose.position.z
            ]
            yaw = math.atan2(xyz[1], xyz[0])
            rpy[2] = yaw
            rpy[0] = 0.0
            rpy[1] = 0.0
            if approach_distance_from_centroid == None:
                approach_distance_from_centroid = 0.0
            xyz[0] -= approach_distance_from_centroid*math.cos(yaw)
            xyz[1] -= approach_distance_from_centroid*math.sin(yaw)
            joints = arm.compute_ik(np.array([*xyz, *rpy]))
            arm.set_arm_joints(joints)
            rospy.Rate(1.0).sleep()
        req = ExecutePolicyRequest()
        req.policy_name = policy_info.name
        execute_policy_service_proxy.call(req)
        return 'execution succeeded'

    return StructuredTool.from_function(func=imitation_learning_tool, name=policy_info.name, description=policy_info.description)