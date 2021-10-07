#!/usr/bin/env python
import rospy
import rosbag
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import struct
import ros_numpy
import time
import numpy as np
import tf

class ConvertPointCloud():
    # initialise shared variables needed for control
    def __init__(self, scan_path, no_display=False):
        # declare publisher for point cloud data, to be read by Rviz
        self.pub = rospy.Publisher('point_cloud', PointCloud2, queue_size=10)

        self.all_data = []
        self.all_data_array = []
        self.new_msg = None
        bag = rosbag.Bag(scan_path)
        for topic, msg, t in bag.read_messages(topics=['/livox/lidar']):
            self.all_data.extend([ord(c) for c in msg.data])
            self.all_data_array.extend(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg))
            self.new_msg = msg
        bag.close()

        self.transform_lidar_data()

        # publish empty pc and update frame
        self.new_msg.data = []
        self.new_msg.header.frame_id = 'point_cloud'
        self.new_msg.width = 0
        self.new_msg.header.stamp = rospy.get_rostime()
        # publish entire scan
        self.pub.publish(self.new_msg)

        time.sleep(1)

        # add new data to msg
        self.new_msg.data = self.all_data
        # change width to expand to length of entir scan
        self.new_msg.width = len(self.all_data_array)
        self.new_msg.header.stamp = rospy.get_rostime()
        # publish entire scan
        if not no_display:
            self.pub.publish(self.new_msg)

        time.sleep(1)

    def transform_lidar_data(self):
        # apply translation
        i = 0
        delete_indeces = []
        all_data_array_new = []
        all_data_new = []

        for point in self.all_data_array:
            # downsample to make it easier to process
            if i%50:
                i+=1
                continue

            # skip all scans far away from the tree
            if np.linalg.norm(point) > 2:
                i+=1
                continue

            all_data_array_new.append(point)
            all_data_new.extend(self.all_data[i*32:(i+1)*32])
            i+=1

        self.all_data = all_data_new
        self.all_data_array = all_data_array_new
