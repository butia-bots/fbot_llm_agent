from langchain_core.tools import StructuredTool
from butia_vision_msgs.srv import ListClasses, ListClassesRequest, ListClassesResponse
from butia_vision_msgs.srv import SetClass, SetClassRequest, SetClassResponse
from butia_vision_msgs.msg import Description3D, Recognitions3D
from butia_world_msgs.srv import GetPose, GetPoseRequest, GetPoseResponse
from butia_speech.srv import SynthesizeSpeech, SynthesizeSpeechRequest, SynthesizeSpeechResponse
import rospy
from typing import List, Tuple
import numpy as np
import ros_numpy
from tf import TransformListener
from tf.transformations import quaternion_from_euler, euler_from_quaternion, euler_from_matrix
from geometry_msgs.msg import PoseStamped
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from std_msgs.msg import Float64MultiArray
from actionlib.simple_action_client import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sensor_msgs.msg import Image
from maestro.visualizers import MarkVisualizer
import supervision as sv
import math
from transformers import Tool

class RobotInterface:
    def __init__(self, manipulator_model="doris_arm"):
        self.manipulator_model = manipulator_model

        if self.manipulator_model != None:
            self.manipulator = InterbotixManipulatorXS(robot_model=self.manipulator_model, init_node=False)

        self.move_base_client = SimpleActionClient("move_base", MoveBaseAction)

        self.neck_pub = rospy.Publisher("neck", Float64MultiArray, queue_size=1)

        self.list_classes_proxy = rospy.ServiceProxy('/butia_vision/br/object_recognition/list_classes', ListClasses)
        self.set_class_proxy = rospy.ServiceProxy('/butia_vision/br/object_recognition/set_class', SetClass)
        self.recognitions3d_sub = rospy.Subscriber('/butia_vision/br/object_recognition3d', Recognitions3D, callback=self._update_recognitions3d)
        self.image_subscriber = rospy.Subscriber('/butia_vision/bvb/image_rgb', Image, callback=self._update_image_rgb)

        self.synthesize_speech_proxy = rospy.ServiceProxy('/butia_speech/ss/say_something', SynthesizeSpeech)

        self.get_waypoint_proxy = rospy.ServiceProxy('/butia_world/get_pose', GetPose)

        self.tfl = TransformListener()

        self.neck_pub.publish(data=[180.0, 180.0])

    def _update_recognitions3d(self, msg: Recognitions3D):
        self.recognitions3d_msg = msg

    def _update_image_rgb(self, msg: Image):
        self.image_rgb_msg = msg

    def speak(self, utterance: str):
        req = SynthesizeSpeechRequest()
        req.lang = 'en'
        req.text = utterance
        res: SynthesizeSpeechResponse = self.synthesize_speech_proxy.call(req)

    def annotate_camera_view(self):
        image_rgb = self.image_rgb_msg
        recognitions = self.recognitions3d_msg
        image_arr = ros_numpy.numpify(image_rgb)
        visualizer = MarkVisualizer()
        xyxy = []
        for description in recognitions.descriptions:
            x1 = description.bbox2D.center.x - (description.bbox2D.size_x/2.0)
            y1 = description.bbox2D.center.y - (description.bbox2D.size_y/2.0)
            x2 = description.bbox2D.center.x + (description.bbox2D.size_x/2.0)
            y2 = description.bbox2D.center.y + (description.bbox2D.size_y/2.0)
            xyxy.append([x1,y1,x2,y2])
        xyxy = np.array(xyxy)
        marks = sv.Detections(xyxy=xyxy)
        annotated_image_arr = visualizer.visualize(image=image_arr, marks=marks, with_box=True, with_polygon=False)
        return recognitions, annotated_image_arr

    def get_available_waypoints(self)->List[str]:
        """Gets the names of the available waypoints in the map"""
        return rospy.get_param('/butia_world/pose/targets', {}).keys()

    def get_waypoint_pose(self, waypoint_name: str)->List[float]:
        """Gets the pose of the waypoint, in the map reference frame"""
        req = GetPoseRequest()
        req.key = f'target/{waypoint_name}/pose'
        res: GetPoseResponse = self.get_waypoint_proxy.call(req)
        position = [
            res.pose.position.x,
            res.pose.position.y,
            res.pose.position.z
        ]
        quat = [
            res.pose.orientation.x,
            res.pose.orientation.y,
            res.pose.orientation.z,
            res.pose.orientation.w
        ]
        orientation = euler_from_quaternion(quat)
        return [*position, *orientation]

    def get_mobile_base_pose(self)->List[float]:
        """Gets the current mobile base pose, in the map reference frame"""
        self.tfl.waitForTransform(self.get_map_reference_frame(), "base_footprint", rospy.Time(), rospy.Duration(10.0))
        translation, rotation = self.tfl.lookupTransform(self.get_map_reference_frame(), "base_footprint", rospy.Time())
        return [*translation, euler_from_quaternion(rotation)]
    
    def move_mobile_base(self, pose: List[float], blocking: bool=True):
        """Navigates the mobile base to the given pose in the map reference frame, given as a x, y, z, roll, pitch, yaw list. Only the x, y, and yaw values are used."""
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.pose.position.x = pose[0]
        ps.pose.position.y = pose[1]
        quat = quaternion_from_euler([0.0, 0.0, pose[5]])
        ps.pose.orientation.x = quat[0]
        ps.pose.orientation.y = quat[1]
        ps.pose.orientation.z = quat[2]
        ps.pose.orientation.w = quat[3]
        goal = MoveBaseGoal()
        goal.target_pose = ps
        if blocking == True:
            self.move_base_client.send_goal_and_wait(goal=goal)
        else:
            self.move_base_client.send_goal(goal=goal)

    def retreat_arm(self):
        assert self.manipulator_model is not None
        self.manipulator.arm.go_to_home_pose()

    def grasp(self, grasp_pose: List[float]):
        """Executes a grasp at the given grasp pose. The grasp pose must be in the arm reference frame"""
        pre_grasp_pose = grasp_pose.copy()
        yaw = math.atan2(pre_grasp_pose[1], pre_grasp_pose[0])
        pre_grasp_pose[0] -= 0.2*math.cos(yaw)
        pre_grasp_pose[1] -= 0.2*math.sin(yaw)
        post_grasp_pose = grasp_pose.copy()
        post_grasp_pose[2] += 0.2
        self.move_arm(pose=pre_grasp_pose)
        self.open_gripper()
        self.move_arm(pose=grasp_pose)
        self.close_gripper()
        self.move_arm(pose=post_grasp_pose)

    def place(self, place_pose: List[float]):
        """Executes a placement at the given place pose. The place pose must be in the arm reference frame"""
        place_pose = place_pose.compy()
        place_pose[2] += 0.2
        self.move_arm(pose=place_pose)
        self.open_gripper()
    
    def move_arm(self, pose: List[float]):
        """Move the end-effector to the pose specified as a 1-D list of length 6, representing x, y, z, roll, pitch, yaw in the arm coordinate frame"""
        assert self.manipulator_model is not None
        x, y, z, roll, pitch, yaw = pose
        self.manipulator.arm.set_ee_pose_components(x=x, y=y, z=z, roll=roll, pitch=pitch)

    def get_arm_pose(self)->List[float]:
        """Gets the current end-effector pose in the arm reference frame"""
        assert self.manipulator_model is not None
        pose_matrix: np.ndarray = self.manipulator.arm.get_ee_pose()
        position = pose_matrix[:3,3].flatten()
        orientation = euler_from_matrix(pose_matrix[:3,:3])
        return [*position, *orientation]

    def open_gripper(self):
        """Open the gripper"""
        assert self.manipulator_model is not None
        self.manipulator.gripper.open()

    def close_gripper(self):
        """Close the gripper"""
        assert self.manipulator_model is not None
        self.manipulator.gripper.close()

    def get_arm_reference_frame(self)->str:
        """Gets the arm reference frame"""
        assert self.manipulator_model is not None
        return f'{self.manipulator_model}/base_link'

    def get_camera_reference_frame(self)->str:
        """Gets the camera reference frame"""
        return self.recognitions3d_msg.header.frame_id

    def get_map_reference_frame(self)->str:
        """Gets the map reference frame"""
        return "map"

    def transform_pose(self, pose: List[float], source_frame: str, target_frame: str)->List[float]:
        """Transform a pose from the source to the target reference frame."""
        key2frame = {
            'map': self.get_map_reference_frame(),
            'camera': self.get_camera_reference_frame(),
            'arm': self.get_arm_reference_frame()
        }
        for waypoint in self.get_available_waypoints():
            key2frame[waypoint] = self.get_map_reference_frame()
        for obj in self.list_detection_classes():
            key2frame[obj] = self.get_camera_reference_frame()
        source_frame = key2frame[source_frame]
        target_frame = key2frame[target_frame]
        ps = PoseStamped()
        ps.header.frame_id = source_frame
        ps.pose.position.x = pose[0]
        ps.pose.position.y = pose[1]
        ps.pose.position.z = pose[2]
        quat = quaternion_from_euler(pose[3], pose[4], pose[5])
        ps.pose.orientation.x = quat[0]
        ps.pose.orientation.y = quat[1]
        ps.pose.orientation.z = quat[2]
        ps.pose.orientation.w = quat[3]
        self.tfl.waitForTransform(target_frame=target_frame, source_frame=source_frame, time=rospy.Time(), timeout=rospy.Duration(10.0))
        ps = self.tfl.transformPose(target_frame=target_frame, ps=ps)
        position = [
            ps.pose.position.x,
            ps.pose.position.y,
            ps.pose.position.z
        ]
        quat = [
            ps.pose.orientation.x,
            ps.pose.orientation.y,
            ps.pose.orientation.z,
            ps.pose.orientation.w,
        ]
        orientation = euler_from_quaternion(quat)
        return [*position, *orientation]

    def object_detection_3d(self, class_name: str)->Tuple[List[List[float]],List[List[List[float]]]]:
        """Performs 3d object detection, given a class name. Returns a tuple of ([xyz_centroid_position_as_list, ...], [xyz_point_cloud_as_list, ...]). Poses are in the camera reference frame."""
        if class_name not in self.list_detection_classes():
            self.add_detection_class(class_name=class_name)
        rospy.wait_for_message(self.recognitions3d_sub.name, Recognitions3D)
        descriptions = filter(lambda e: e.label, self.recognitions3d_msg.descriptions)
        positions = [[description.bbox.center.position.x, description.bbox.center.position.y, description.bbox.center.position.z] for description in descriptions]
        clouds = []
        for description in descriptions:
            cloud = ros_numpy.numpify(description.filtered_cloud)
            clouds.append(list(zip(
                cloud['x'],
                cloud['y'],
                cloud['z']
            )))
        return positions, clouds

    def list_detection_classes(self)->List[str]:
        """Gets a list of currently set object detection class names"""
        req = ListClassesRequest()
        res: ListClassesResponse = self.list_classes_proxy.call(req)
        return res.classes

    def add_detection_class(self, class_name: str):
        """Configure and adds a new class name to the object detection system"""
        req = SetClassRequest()
        req.class_name = class_name
        res: SetClassResponse = self.set_class_proxy.call(req)

    def get_code_tools_langchain(self):
        return [
            StructuredTool.from_function(self.move_arm, name='move_arm'),
            StructuredTool.from_function(self.get_arm_pose, name='get_arm_pose'),
            StructuredTool.from_function(self.get_arm_reference_frame, name='get_arm_reference_frame'),
            StructuredTool.from_function(self.open_gripper, name='open_gripper'),
            StructuredTool.from_function(self.close_gripper, name='close_gripper'),
            StructuredTool.from_function(self.grasp, name="grasp"),
            StructuredTool.from_function(self.object_detection_3d, name="object_detection_3d"),
            StructuredTool.from_function(self.list_detection_classes, name="list_object_detection_classes"),
            StructuredTool.from_function(self.get_camera_reference_frame, name='get_camera_reference_frame'),
            StructuredTool.from_function(self.move_mobile_base, name="move_mobile_base"),
            StructuredTool.from_function(self.get_mobile_base_pose, name='get_mobile_base_pose'),
            StructuredTool.from_function(self.get_map_reference_frame, name="get_map_reference_frame"),
            StructuredTool.from_function(self.get_available_waypoints, name='get_available_waypoints'),
            StructuredTool.from_function(self.get_waypoint_pose, name="get_waypoint_pose"),
            StructuredTool.from_function(self.transform_pose, name='transform_pose'),
        ]

    def get_code_tools_hf(self):
        return [Tool.from_langchain(t) for t in self.get_code_tools_langchain()]