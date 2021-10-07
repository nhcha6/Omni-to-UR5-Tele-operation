#!/usr/bin/env python  
import roslib
roslib.load_manifest('haptic_control')

from pyquaternion import Quaternion as quat
import rospy
import tf
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

# this script establishes a ROS node to broadcast the desired transformation matrix to 
# orientate a point cloud scan in front of the robot at the desired anglel.
if __name__ == '__main__':
    # initialise the node
    rospy.init_node('fixed_tf_broadcaster', anonymous=True)

    # define the broadcaster to update at a rate of 10Hz
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)

    # define quarternion for the rotation angle of the tree in front of the robot
    rotation = -20
    q = quat(axis=[0.0, 0.0, 1.0], degrees=rotation)

    # define goal tree position in world frame, and tree position in local frame
    # this was defined by inspection to positoin the tree in front of the robot
    tree_goal_pos = [0.44, -0.6, 0.4]
    tree_frame_pos = [1.35, 0.05, 0]

    # find rotated position of the tree
    R = q.rotation_matrix
    rotated_pos = np.dot(R,np.transpose(tree_frame_pos))

    # find translation of the point_cloud frame required to position it at the tree_goal_pos
    translation = np.subtract(tree_goal_pos, rotated_pos)

    # sleep for the update rate
    rate.sleep()

    # broadcast the desired tf
    while not rospy.is_shutdown():
        br.sendTransform(translation,
                         (q.x, q.y, q.z, q.w),
                         rospy.Time.now(),
                         "point_cloud",
                         "world")
        rate.sleep()