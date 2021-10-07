#!/usr/bin/env python
import numpy as np
from ikfastpy import PyKinematics
from pyquaternion import Quaternion
from sklearn.metrics import mean_squared_error
from ur5_control.srv import CollCheck
import pandas as pd
import math

from haptic_control.srv import ur5_ik, ur5_ikResponse, ur5_fk, ur5_fkResponse
from geometry_msgs.msg import Pose
import rospy

# this object establishes as a ROS service to return the inverse kinematic solution of the UR5
# it establishes a similar service to return the forward kinematic solution of the UR5
# these services call the ikfastpy scripts which generate solutions very quickly
class UR5IK():
	def __init__(self):
		# Initialize kinematics for UR5 robot arm using ikfastpy module
		self.ur5_kin = PyKinematics()
		self.n_joints = self.ur5_kin.getDOF()
		# collision object server
		self._collision_check = rospy.ServiceProxy('/ur5_control/scene/collision/check', CollCheck)

	# service function called by omni_to_ur5 node when it wants the ik solutions for a given
	# pose found
	def calculate_ur5_ik(self, req):
		# extract pose from request
		pose = req.Pose
		joint_configs = self.ur5_kin.inverse(pose)

		# separate solutions from list format
		n_solutions = int(len(joint_configs)/self.n_joints)
		print("%d solutions found:"%(n_solutions))
		joint_configs = np.asarray(joint_configs).reshape(n_solutions,self.n_joints)

		# return all solutions that do not cause a collision
		lower_error = 10000
		joint_pos = [[] for i in range(8)]
		index = 0
		for joint_config in joint_configs:
			# check collision
			collision = self._collision_check(joint_config, False)
			if collision.Collision[0]:
				continue
			# add configuration to list of solutions
			print(joint_config)
			joint_pos[index] = joint_config
			index+=1
		
		# return the ik solutions to the caller
		return ur5_ikResponse(joint_pos[0], joint_pos[1], joint_pos[2], joint_pos[3], joint_pos[4], joint_pos[5], joint_pos[6] ,joint_pos[7]) 

	# service which calculates the pose o the ur5 given the joint states
	def calculate_ur5_fk(self, req):
		# initial joint state of ur5 that we are transforming based on omni motion
		joint_state = req.JointStates
		joint_state = [round(x,2) for x in joint_state]

		# forward kinematics of old pose
		ee_pose = self.ur5_kin.forward(joint_state)
		
		# return pose to the caller
		return ur5_fkResponse(ee_pose) 
	
	# called on initiasation to setup the server to listen for requests to calculate the 
	# ur5 joint states based on the relative omni changes.
	def ur5_ik_server(self):
		# setup node and ik service function
		rospy.init_node('ur5_ik', anonymous=True)
		s_ik = rospy.Service('calculate_ur5_ik', ur5_ik, self.calculate_ur5_ik)
		s_fk = rospy.Service('calculate_ur5_fk', ur5_fk, self.calculate_ur5_fk)
		print("Ready to calculate ik")
		rospy.spin()

# create object and clients when run
if __name__ == "__main__":
	ur5_ik_obj = UR5IK()
	ur5_ik_obj.ur5_ik_server()
