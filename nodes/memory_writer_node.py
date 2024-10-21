#!/usr/bin/env python3
import rospy
from butia_vision_msgs.srv import VisualQuestionAnswering, VisualQuestionAnsweringRequest, VisualQuestionAnsweringResponse
from geometry_msgs.msg import PoseWithCovarianceStamped
import weaviate
import weaviate.classes as wvc
import time
from tf.transformations import euler_from_quaternion

def pose_callback(msg: PoseWithCovarianceStamped):
    timestamp = time.time()
    req = VisualQuestionAnsweringRequest()
    req.question = "caption en"
    res: VisualQuestionAnsweringResponse = vqa_service_proxy.call(req)
    description = res.answer
    rospy.loginfo(description)
    position = [
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.position.z,
    ]
    orientation = [
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w,
    ]
    orientation = euler_from_quaternion(orientation)
    frame_id = msg.header.frame_id
    location_collection.data.insert({
        'description': description,
        'position': {
            'x': position[0],
            'y': position[1],
        },
        'orientation': {
            'yaw': orientation[2]
        },
    })

if __name__ == '__main__':
    rospy.init_node('memory_writer_node', anonymous=True)
    vector_client = weaviate.connect_to_embedded()
    if vector_client.collections.exists('Location'):
        vector_client.collections.delete('Location')
    location_collection = vector_client.collections.create(
        name='Location',
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers()
    )
    vqa_service_proxy = rospy.ServiceProxy('/butia_vision/br/object_recognition/visual_question_answering', VisualQuestionAnswering)
    pose_subscriber = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, callback=pose_callback, queue_size=0)
    try:
        rospy.spin()
    finally:
        vector_client.close()
