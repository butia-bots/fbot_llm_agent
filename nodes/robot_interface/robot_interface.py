from langchain_core.tools import StructuredTool
from butia_vision_msgs.srv import ListClasses, ListClassesRequest, ListClassesResponse
from butia_vision_msgs.srv import SetClass, SetClassRequest, SetClassResponse
from butia_vision_msgs.srv import VisualQuestionAnswering, VisualQuestionAnsweringRequest, VisualQuestionAnsweringResponse
from butia_vision_msgs.srv import LookAtDescription3D, LookAtDescription3DRequest, LookAtDescription3DResponse
from butia_vision_msgs.msg import Description3D, Recognitions3D
from butia_world_msgs.srv import GetPose, GetPoseRequest, GetPoseResponse
from butia_speech.srv import SynthesizeSpeech, SynthesizeSpeechRequest, SynthesizeSpeechResponse
import rospy
from typing import List, Tuple
import ros_numpy
from tf import TransformListener
from tf.transformations import quaternion_from_euler, euler_from_quaternion, euler_from_matrix
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest, GetPositionFKResponse
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Empty, Float64
from std_srvs.srv import Empty as EmptySrv
from actionlib.simple_action_client import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sensor_msgs.msg import Image
import math
from transformers import Tool
from threading import Thread, Event

class RobotTool(Tool):
    def from_function(function, name, output_type, inputs):
        t = RobotTool()
        t.function = function
        t.name = name
        t.description = function.__doc__
        t.inputs = inputs
        t.output_type = output_type
        return t

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

class RobotInterface:
    arm_joint_names = ["waist", "shoulder", "elbow", "wrist_angle", "wrist_rotate"]
    arm_sleep_joint_positions = [0, -1.88, 1.5, 0.8, 0, 0]
    gripper_joint_names = ["left_finger", "right_finger"]
    gripper_open_joint_positions = [0.037, -0.037]
    gripper_close_joint_positions = [0.015, -0.015]

    def __init__(self, manipulator_model="doris_arm", arm_group_name="arm", arm_controller_ns="/doris_arm/arm_controller/command", gripper_controller_ns="/doris_arm/gripper_controller/command", arm_joint_state="/doris_arm/joint_state"):
        self.manipulator_model = manipulator_model

        if self.manipulator_model != None:
            self.arm_group_name = arm_group_name
            self.compute_ik_proxy = rospy.ServiceProxy(f'/{self.manipulator_model}/move_group/compute_ik', GetPositionIK)
            self.compute_fk_proxy = rospy.ServiceProxy(f'/{self.manipulator_model}/move_group/compute_fk', GetPositionFK)
            self.arm_publisher = rospy.Publisher(arm_controller_ns, JointTrajectory, queue_size=1)
            self.gripper_publisher = rospy.Publisher(gripper_controller_ns, JointTrajectory, queue_size=1)
            self.joint_state_subscriber = rospy.Subscriber(arm_joint_state, JointState, self._update_joint_state)

        self.move_base_client = SimpleActionClient("move_base", MoveBaseAction)

        self.neck_pub = rospy.Publisher("neck", Float64MultiArray, queue_size=1)
        self.head_pan_pub = rospy.Publisher("/doris_head/head_pan_position_controller/command", Float64, queue_size=1)
        self.head_tilt_pub = rospy.Publisher("/doris_head/head_tilt_position_controller/command", Float64, queue_size=1)

        self.look_at_start_proxy = rospy.ServiceProxy("/lookat_start", LookAtDescription3D)
        self.look_at_stop_proxy = rospy.ServiceProxy("/lookat_stop", EmptySrv)

        self.list_classes_proxy = rospy.ServiceProxy('/butia_vision/br/object_recognition/list_classes', ListClasses)
        self.set_class_proxy = rospy.ServiceProxy('/butia_vision/br/object_recognition/set_class', SetClass)
        self.recognitions3d_sub = rospy.Subscriber('/butia_vision/br/object_recognition3d', Recognitions3D, callback=self._update_recognitions3d)
        self.image_subscriber = rospy.Subscriber('/butia_vision/bvb/image_rgb', Image, callback=self._update_image_rgb)
        self.vqa_proxy = rospy.ServiceProxy('/butia_vision/br/object_recognition/visual_question_answering', VisualQuestionAnswering)

        self.start_tracking_proxy = rospy.ServiceProxy('/butia_vision/pt/start', EmptySrv)
        self.tracking3d_sub = rospy.Subscriber('/butia_vision/pt/tracking3D', Recognitions3D, callback=self._update_tracking3d)
        self.tracking_rate = rospy.Rate(1.0)

        self.synthesize_speech_proxy = rospy.ServiceProxy('/butia_speech/ss/say_something', SynthesizeSpeech)

        self.hotword_event = Event()
        self.hotword_rate = rospy.Rate(0.5)
        self.hotword_subscriber = rospy.Subscriber('/butia_speech/bhd/detected', Empty, self._detect_hotword)

        self.get_waypoint_proxy = rospy.ServiceProxy('/butia_world/get_pose', GetPose)

        self.guide_rate = rospy.Rate(0.1)

        self.tfl = TransformListener()

        self.set_neck_angle(180, 180)

    def _update_joint_state(self, msg: JointState):
        self.joint_state_msg = msg

    def _detect_hotword(self, msg):
        self.hotword_event.set()
        self.hotword_rate.sleep()
        self.hotword_event.clear()

    def _update_recognitions3d(self, msg: Recognitions3D):
        self.recognitions3d_msg = msg

    def _update_tracking3d(self, msg: Recognitions3D):
        self.tracking3d_msg = msg

    def _update_image_rgb(self, msg: Image):
        self.image_rgb_msg = msg

    def start_looking_at_person(self):
        req = LookAtDescription3DRequest()
        req.global_id = 0
        req.id = 0
        req.recognitions3d_topic = self.tracking3d_sub.name
        req.label = ''
        try:
            self.look_at_start_proxy.call(req)
        except:
            pass

    def stop_looking_at(self):
        try:
            self.look_at_stop_proxy.call()
        except:
            pass

    def set_neck_angle(self, horizontal: float, vertical: float):
        self.neck_pub.publish(data=[horizontal, vertical])
        self.head_pan_pub.publish(data=math.radians(horizontal - 180))
        self.head_tilt_pub.publish(data=math.radians(vertical - 180))

    def compute_ik(self, pose: List[float])->List[float]:
        req = GetPositionIKRequest()
        req.ik_request.group_name = self.arm_group_name
        req.ik_request.pose_stamped.header.frame_id = f'{self.manipulator_model}/base_link'
        req.ik_request.pose_stamped.pose.position.x = pose[0]
        req.ik_request.pose_stamped.pose.position.y = pose[1]
        req.ik_request.pose_stamped.pose.position.z = pose[2]
        quat = quaternion_from_euler(pose[3], pose[4], pose[5])
        req.ik_request.pose_stamped.pose.orientation.x = quat[0]
        req.ik_request.pose_stamped.pose.orientation.y = quat[1]
        req.ik_request.pose_stamped.pose.orientation.z = quat[2]
        req.ik_request.pose_stamped.pose.orientation.w = quat[3]
        res: GetPositionIKResponse = self.compute_ik_proxy.call(req)
        joints = []
        for name, position in zip(res.solution.joint_state.name, res.solution.joint_state.position):
            if name in self.arm_joint_names:
                joints.append(position)
        return joints
    
    def set_arm_joints(self, joints: List[float]):
        jt = JointTrajectory()
        jt.joint_names = self.arm_joint_names
        jtp = JointTrajectoryPoint()
        jtp.positions = joints
        jtp.time_from_start = rospy.Duration(1.0)
        jt.points.append(jtp)
        self.arm_publisher.publish(jt)
        rospy.Rate(1.0).sleep()

    def set_gripper_joints(self, joints: List[float]):
        jt = JointTrajectory()
        jt.joint_names = self.gripper_joint_names
        jtp = JointTrajectoryPoint()
        jtp.positions = joints
        jtp.time_from_start = rospy.Duration(1.0)
        jt.points.append(jtp)
        self.gripper_publisher.publish(jt)
        rospy.Rate(1.0).sleep()

    def compute_fk(self, joints: List[float])->List[float]:
        req = GetPositionFKRequest()
        req.fk_link_names = [f'{self.manipulator_model}/ee_gripper_link']
        req.robot_state.joint_state.name = self.arm_joint_names
        req.robot_state.joint_state.position = joints
        res: GetPositionFKResponse = self.compute_fk_proxy.call(req)
        position = [
            res.pose_stamped[0].pose.position.x,
            res.pose_stamped[0].pose.position.y,
            res.pose_stamped[0].pose.position.z
        ]
        quat = [
            res.pose_stamped[0].pose.orientation.x,
            res.pose_stamped[0].pose.orientation.y,
            res.pose_stamped[0].pose.orientation.z,
            res.pose_stamped[0].pose.orientation.w
        ]
        orientation = euler_from_quaternion(quat)
        return [*position, *orientation]
    
    def get_arm_joints(self)->List[float]:
        rospy.wait_for_message(self.joint_state_subscriber.name, JointState)
        joints = []
        for name, position in zip(self.joint_state_msg.name, self.joint_state_msg.position):
            if name in self.arm_joint_names:
                joints.append(position)
        return joints

    def speak(self, utterance: str):
        """Uses a TTS engine to speak"""
        req = SynthesizeSpeechRequest()
        req.lang = 'en'
        req.text = utterance
        try:
            res: SynthesizeSpeechResponse = self.synthesize_speech_proxy.call(req)
        except:
            pass

    def person_tracking_3d(self)->Tuple[List[List[float]],List[List[List[float]]]]:
        rospy.wait_for_message(self.tracking3d_sub.name, Recognitions3D)
        descriptions = filter(lambda e: e.label, self.tracking3d_msg.descriptions)
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

    def approach_position(self, position: List[float], offset=1.5, blocking=True):
        current_pose = self.get_mobile_base_pose()
        current_yaw = current_pose[5]
        x = position[0] - current_pose[0]
        y = position[1] - current_pose[1]
        yaw = current_yaw + math.atan2(y, x)
        x = position[0] - offset*math.cos(yaw)
        y = position[1] - offset*math.sin(yaw)
        self.move_mobile_base([x, y, 0.0, 0.0, 0.0, yaw], blocking=blocking)

    def rotate(self, angle):
        pose = self.get_mobile_base_pose()
        pose[5] += angle
        self.move_mobile_base(pose)

    def follow(self, person_position: List[float]):
        """Follows the person at the given person_position. The reference frame for the person_position must be the map reference frame."""
        self.approach_position(person_position, offset=1.5, blocking=True)
        self.speak("Please, come to me and stand one and a half meters in front of me.")
        self.speak("I will follow you, I have a important instruction for you, say Hello DoRIS to make me stop when you arrived in the next waypoint")
        self.start_tracking_proxy.call()
        self.start_looking_at_person()
        def check_stop_following():
            self.hotword_event.wait()
        hotword_thread = Thread(target=check_stop_following)
        hotword_thread.start()
        while hotword_thread.is_alive():
            person_positions, person_clouds = self.person_tracking_3d()
            if len(person_positions) == 0:
                self.rotate(math.radians(45))
            person_position = person_positions[0]
            self.approach_position(person_position, offset=1.5, blocking=False)
            self.tracking_rate.sleep()
        self.cancel_move_mobile_base()
        self.stop_looking_at()

    def follow_person_by_description(self, person_description: str, waypoint_name: str):
        """Follows the person with the given description"""
        self.navigate_to_waypoint(waypoint_name=waypoint_name)
        person_positions, person_clouds = self.object_detection_3d(class_name=person_description)
        if len(person_positions) == 0:
            return
        self.follow(self.transform_pose(person_positions[0]+[0,0,0], 'camera', 'map'))

    def guide(self, person_position: List[float], destination_waypoint_pose: List[float]):
        """Guides the person at the given person_position to the given destination_waypoint_pose. the person_position and the destination_waypoint_pose must be on the map reference frame."""
        self.approach_position(person_position, offset=1.5, blocking=True)
        self.speak("Please follow me, I will take you to the destination. Keep 1.5 meters behind me, I will turn every 10 seconds to check if you are still following me.")
        while math.dist(destination_waypoint_pose[:3], self.get_mobile_base_pose()[:3]) > 1.0:
            self.move_mobile_base(destination_waypoint_pose, blocking=False)
            self.guide_rate.sleep()
            self.cancel_move_mobile_base()
            self.rotate(math.radians(180))
            person_positions, person_clouds = self.object_detection_3d(class_name="person")
            person_poses = [self.transform_pose([*p, 0.0, 0.0, 0.0], source_frame='camera', target_frame='map') for p in person_positions]
            person_poses = [p for p in person_poses if math.dist(p[:3], self.get_mobile_base_pose()[:3]) <= 1.5]
            if len(person_poses) == 0:
                self.speak("You are getting lost! Please return to follow me, so that I can finish my task.")
                while len(person_poses) == 0:
                    person_positions, person_clouds = self.object_detection_3d(class_name="person")
                    person_poses = [self.transform_pose([*p, 0.0, 0.0, 0.0], source_frame='camera', target_frame='map') for p in person_positions]
                    person_poses = [p for p in person_poses if math.dist(p[:3], self.get_mobile_base_pose()[:3]) <= 1.5]
            self.rotate(math.radians(180))
        self.speak("We have arrived at the destination, you can now stop following me.")

    def guide_person_by_description(self, person_description: str, start_waypoint_name: str, destination_waypoint_name: str):
        """Guides the person with the given description"""
        self.navigate_to_waypoint(waypoint_name=start_waypoint_name)
        person_positions, person_clouds = self.object_detection_3d(class_name=person_description)
        if len(person_positions) == 0:
            return
        self.guide(self.transform_pose(person_positions[0]+[0,0,0], 'camera', 'map'), self.get_waypoint_pose(destination_waypoint_name))

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

    def cancel_move_mobile_base(self):
        self.move_base_client.cancel_goal()
    
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
        self.set_arm_joints(self.arm_sleep_joint_positions)

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
        self.retreat_arm()

    def grasp_object(self, class_name: str, waypoint_name: str):
        """Grasps the object of the given class name"""
        self.navigate_to_waypoint(waypoint_name=waypoint_name)
        positions, clouds = self.object_detection_3d(class_name)
        if len(positions) == 0:
            return
        grasp_pose = positions[0]
        self.grasp(self.transform_pose(grasp_pose, 'camera', 'arm'))

    def place(self, place_pose: List[float]):
        """Executes a placement at the given place pose. The place pose must be in the arm reference frame"""
        pre_place_pose = place_pose.copy()
        pre_place_pose[2] += 0.2
        self.move_arm(pose=pre_place_pose)
        self.move_arm(pose=place_pose)
        self.open_gripper()
        self.retreat_arm()

    def place_on_surface(self, surface_class_name: str, waypoint_name: str):
        """Places the object on the surface of the given class name"""
        self.navigate_to_waypoint(waypoint_name=waypoint_name)
        positions, clouds = self.object_detection_3d(surface_class_name)
        if len(positions) == 0:
            return
        place_pose = positions[0]
        self.place(self.transform_pose(place_pose, 'camera', 'arm'))
    
    def move_arm(self, pose: List[float]):
        """Move the end-effector to the pose specified as a 1-D list of length 6, representing x, y, z, roll, pitch, yaw in the arm coordinate frame"""
        assert self.manipulator_model is not None
        x, y, z, roll, pitch, yaw = pose
        yaw = math.atan2(y, x)
        joints = self.compute_ik([x, y, z, roll, pitch, yaw])
        self.set_arm_joints(joints)

    def get_arm_pose(self)->List[float]:
        """Gets the current end-effector pose in the arm reference frame"""
        assert self.manipulator_model is not None
        joints = self.get_arm_joints()
        pose = self.compute_fk(joints)
        return pose

    def open_gripper(self):
        """Open the gripper"""
        assert self.manipulator_model is not None
        self.set_gripper_joints(self.gripper_open_joint_positions)

    def close_gripper(self):
        """Close the gripper"""
        assert self.manipulator_model is not None
        self.set_gripper_joints(self.gripper_close_joint_positions)

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
        if source_frame in key2frame:
            source_frame = key2frame[source_frame]
        if target_frame in key2frame:
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
    
    def visual_question_answering(self, question: str, waypoint_name: str)->str:
        """Answers a visual question"""
        self.navigate_to_waypoint(waypoint_name=waypoint_name)
        req = VisualQuestionAnsweringRequest()
        req.question = question
        try:
            res: VisualQuestionAnsweringResponse = self.vqa_proxy.call(req)
            return res.answer
        except:
            return "I am sorry, I could not answer the question."

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

    def navigate_to_waypoint(self, waypoint_name: str):
        """Navigates the robot to the given waypoint"""
        if waypoint_name not in self.get_available_waypoints():
            raise ValueError(f"Waypoint {waypoint_name} is not available. The following waypoints are available instead: {self.get_available_waypoints()}")
        pose = self.get_waypoint_pose(waypoint_name)
        self.move_mobile_base(pose)

    def get_code_tools_langchain(self):
        return [
            StructuredTool.from_function(self.move_arm, name='move_arm'),
            StructuredTool.from_function(self.get_arm_pose, name='get_arm_pose'),
            StructuredTool.from_function(self.get_arm_reference_frame, name='get_arm_reference_frame'),
            StructuredTool.from_function(self.open_gripper, name='open_gripper'),
            StructuredTool.from_function(self.close_gripper, name='close_gripper'),
            StructuredTool.from_function(self.grasp, name="grasp"),
            StructuredTool.from_function(self.place, name="place"),
            StructuredTool.from_function(self.object_detection_3d, name="object_detection_3d"),
            StructuredTool.from_function(self.list_detection_classes, name="list_object_detection_classes"),
            StructuredTool.from_function(self.get_camera_reference_frame, name='get_camera_reference_frame'),
            StructuredTool.from_function(self.move_mobile_base, name="move_mobile_base"),
            StructuredTool.from_function(self.get_mobile_base_pose, name='get_mobile_base_pose'),
            StructuredTool.from_function(self.get_map_reference_frame, name="get_map_reference_frame"),
            StructuredTool.from_function(self.get_available_waypoints, name='get_available_waypoints'),
            StructuredTool.from_function(self.get_waypoint_pose, name="get_waypoint_pose"),
            StructuredTool.from_function(self.transform_pose, name='transform_pose'),
            StructuredTool.from_function(self.follow, name='follow'),
            StructuredTool.from_function(self.guide, name='guide'),
            StructuredTool.from_function(self.speak, 'speak')
        ]

    def get_code_tools_hf(self):
        return [
            RobotTool.from_function(self.navigate_to_waypoint, name='navigate_to_waypoint', output_type=None, inputs={'waypoint_name': {'type': 'string'}}),
            RobotTool.from_function(self.follow_person_by_description, name='follow_person_by_description', output_type=None, inputs={'person_description': {'type': 'string'}, 'waypoint_name': {'type': 'string'}}),
            RobotTool.from_function(self.guide_person_by_description, name='guide_person_by_description', output_type=None, inputs={'person_description': {'type': 'string'}, 'start_waypoint_name': {'type': 'string'}, 'destination_waypoint_name': {'type': 'string'}}),
            RobotTool.from_function(self.speak, 'speak', output_type=None, inputs={'utterance': {'type': 'string'}}),
            RobotTool.from_function(self.grasp_object, name='grasp_object', output_type=None, inputs={'class_name': {'type': 'string'}, 'waypoint_name': {'type': 'string'}}),
            RobotTool.from_function(self.place_on_surface, name='place_on_surface', output_type=None, inputs={'surface_class_name': {'type': 'string'}, 'waypoint_name': {'type': 'string'}}),
            RobotTool.from_function(self.visual_question_answering, name='visual_question_answering', output_type='string', inputs={'question': {'type': 'string'}, 'waypoint_name': {'type': 'string'}}),
        ]