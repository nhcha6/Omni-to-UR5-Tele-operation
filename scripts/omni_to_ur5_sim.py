#!/usr/bin/env python
# Software License Agreement (BSD License)

# import ros messages and services
import rospy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Bool, Header, ColorRGBA
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Vector3
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray
from ur5_control.srv import *
from haptic_control.srv import *
from phantom_omni.msg import PhantomButtonEvent, OmniFeedback

# import python libraries
import math
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from time import sleep, time
from sklearn.metrics import mean_squared_error
from random import randint, seed
from random import randint
from datetime import datetime
import pickle
import pandas as pd
from os import listdir
from os.path import isfile, join

# import scripts from haptic_control package
from trajectory_action_client import Trajectory
from omni_kinematics import OmniKinematics
from ur5_motion import UR5Kinematics
from pose_to_js import PoseConversion
from convert_point_cloud import ConvertPointCloud

# class to control flow of node
class OmniToUR5():
	# initialise shared variables needed for control
	def __init__(self):
		# ur5 state information
		self.ur5_js = {'shoulder_pan_joint': 0.0, 'shoulder_lift_joint':0.0, 'elbow_joint':0.0, 'wrist_1_joint': 0.0, 'wrist_2_joint': 0.0, 'wrist_3_joint': 0.0}
		self.ur5_tf = None
		self.relative_ur5_js = {'shoulder_pan_joint': 0.0, 'shoulder_lift_joint':0.0, 'elbow_joint':0.0, 'wrist_1_joint': 0.0, 'wrist_2_joint': 0.0, 'wrist_3_joint': 0.0}
		self.ur5_velocity_control = UR5Kinematics()
		self.ur5_home = [-1.0472, -1.6057, -0.8727, -0.2094, -1.5708, 0]

		# omni state information
		self.relative_omni_js = {'waist': 0, 'shoulder': 0, 'elbow': 0, 'wrist1': 0, 'wrist2': 0, 'wrist3': 0}
		self.omni_js = {'waist': 0, 'shoulder': 0, 'elbow': 0, 'wrist1': 0, 'wrist2': 0, 'wrist3': 0}
		self.omni_kinematics = OmniKinematics()
		
		# joint  and link names
		self.ur5_joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
		self.ur5_link_names = ['shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link']
		self.omni_joint_names = ['waist', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3']
		
		# code flow variables
		self.enable_control = False
		self.plans_in_queue = None
		self.rate = 0.2
		self.max_vel = 5

		# markers and arrows
		self.goal_pos=None
		self.goal_marker= None
		self.attraction_arrow = None 
		self.repulsive_arrow_goal = None
		self.repulsive_arrow_ee = None
		self.repulsive_arrow_closest = None 

		# define services
		self.plan_queue = rospy.ServiceProxy('ur5_control/plan/queue', PlanPath)
		self.plan_execute = rospy.ServiceProxy('ur5_control/plan/execute', ServiceInt)
		self.calculate_ur5_ik = rospy.ServiceProxy('calculate_ur5_ik', ur5_ik)
		self.calculate_ur5_fk = rospy.ServiceProxy('calculate_ur5_fk', ur5_fk)
		self.marker_array_publisher = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=100)
		self.pointcloud_tf_pub = rospy.Publisher('pointcloud_tf', Bool, queue_size=100)
		self.collision_object_pub = rospy.Publisher('collision_object', CollisionObject, queue_size=100)

		# define static TFs 
		self.get_static_tf()
		self.omni_to_ur5_eeframe = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

		# lists for storing trajectories for demonstration and validation
		self.picking_trajectory = []
		self.repulsive_goal_list = []
		self.repulsive_ee_list = []
		self.repulsive_closest_list = []
		self.attractive_list = []
		self.RRT_trajectory = []

		# boolean values to control the saving of trajectories
		# set to true if you want to run the validation test
		self.run_validation = False
		# used to record a trajectory that is being executed
		self.save_RRT_flag = False		

		# send to start position to begin control/demonstration
		self.move_to_point(self.ur5_home)
	
	# main function which runs on start-up and controls the switching on and off
	# of teleoperation
	def omni_to_ur5(self):
		
		# subscrive to omni joint states
		sub_omni_js = rospy.Subscriber('omni1_joint_states', JointState, self.update_omni_js)
		
		# subscribe to omni button press topic
		sub_omni_push_ = rospy.Subscriber('omni1_button', PhantomButtonEvent, self.update_control)
		
		# subscribe to ur5 joint states
		sub_ur5_js = rospy.Subscriber('joint_states', JointState, self.update_ur5_js)

		# subscribe to tranformation matrix 
		self.sub_tf = rospy.Subscriber('/tf', TFMessage, self.process_tf)

		# node initialisation
		rospy.init_node('omni_to_ur5', anonymous=True)

		# display lidar scan of apple tree and transform into current frame
		print('Displaying lidar scan')
		self.convert_pointcloud = ConvertPointCloud('/home/nic/catkin_ws/src/haptic_control/scripts/bagfiles/APPLE TREE.bag', self.run_validation)
		self.transform_lidar_data()
		self.add_marker()
		print('ready to run')

		# trajectory object for implemention low level control of the ur5
		self.traj = Trajectory()
		rospy.on_shutdown(self.traj.stop)

		# run the validation code
		if self.run_validation:
			self.run_final_validation('/home/nic/catkin_ws/src/haptic_control/model validation traj/')

		while True:
			# if the grey button of the omni has been pressed, we are in tele-operation mode
			# and we need to update the position of the ur5
			if self.enable_control:
				self.change_ur5()

######################### DEMONSTRATION DELIVERY FUNCTIONS ############################

	# called on repeat when control is enabled, this function calculates the relative
	# change in the omni position and orientation and applied the change to the ur5.
	def change_ur5(self):
		try:
			# define path planning services
			plan_queue = rospy.ServiceProxy('ur5_control/plan/queue', PlanPath)
			plan_execute = rospy.ServiceProxy('ur5_control/plan/execute', ServiceInt)
			self.traj.clear()

			# calculate the relative change of wrist3
			relative_wrist = self.omni_js['wrist3'] - self.relative_omni_js['wrist3']

			# get relative change in joints, and set wrist 3 change to 0 (it will be updated at the end)
			omni_delta_js = [self.omni_js[name] - self.relative_omni_js[name] for name in self.omni_joint_names]
			omni_delta_js[5] = 0
			
			# calculate the omni jacobian for the current joint positions
			J0_omni = self.omni_kinematics.jacobian(self.relative_omni_js, inverse=False)
			
			# calculate the velocity of the end effector based on omni joint position velocity.
			omni_ee_vel = [delta_pos/self.rate for delta_pos in np.dot(J0_omni, omni_delta_js)]
			ur5_ee_vel = [3*vel for vel in omni_ee_vel[0:3]] + omni_ee_vel[3:]
			print('\n')
			
			# apply the jacobian of the ur5 to the desired end effector velocity to calculate joint velocities
			J0_ur5 = self.ur5_velocity_control.apply_jacobian(self.ur5_js, inverse=True)
			js_vel = np.dot(J0_ur5, ur5_ee_vel)
			js_vel[5] = -4*relative_wrist

			# execute trajectory and return if successful or not
			new_pos, success = self.traj.move_robot_js(js_vel, [self.ur5_js[name] for name in self.ur5_joint_names], self.rate)
			
			# convert current UR5 JS to a list
			joint_states = []
			for i in range(len(self.ur5_joint_names)):
				joint_states.append(self.ur5_js[self.ur5_joint_names[i]])
			
			# calculate current pose
			response = self.calculate_ur5_fk(joint_states)
			current_pose = response.Pose

			# save pose to trajectory list and calculate potential field to be
			self.calculate_potential_field(current_pose)

			# if successful, execute the trajectory
			if success:
				# resest relative position so that change in js can be calculated 
				# at the next timestep
				self.relative_omni_js = self.omni_js.copy()
				self.relative_ur5_js = self.ur5_js.copy()

				# start timing, and send execution command
				start = time()
				self.traj.start_topic()

				# wait for the specified update rate to allow the robot to execute its trajectory
				sleep((self.rate))

				# cancel the goal and sleep to ensure it is received before moving
				self.traj.cancel_publisher.publish(self.traj.past_goal_id)
				sleep(0.01)

				# print the velocity vector and the js velocity to check all is working
				print('ts velocity')
				print(ur5_ee_vel)
				print('js velocity')
				print(js_vel)
				print('final pos')
				print([self.ur5_js[name] for name in self.ur5_joint_names])
				print('\n')

			# if no solution was found due to a collision or a reaching a singularity, we look 
			# to apply inverse kinematics here as an alternative solution
			else:
				# find alternative JS solution for current pose and execute using RRTConnect
				self.move_using_ik(current_pose, joint_states)
				
				# resest relative position and sleep:
				self.relative_omni_js = self.omni_js.copy()
				self.relative_ur5_js = self.ur5_js.copy()

				# wait for the update rate to recommence
				sleep(self.rate)
		
		# catch any errors
		except rospy.ServiceException as e:
			print("service call failed: %s"%e)


	# called to find a new joint configuration to execute change in ur5 end-effector due
	# to reaching a singularity or collision when usin velocity control
	def move_using_ik(self, pose, old_joint_state):
		try:
			# define path planning services
			plan_queue = rospy.ServiceProxy('ur5_control/plan/queue', PlanPath)
			plan_execute = rospy.ServiceProxy('ur5_control/plan/execute', ServiceInt)
			
			# call ur5 ik service, providing it with the desired pose of the ur5
			response = self.calculate_ur5_ik(pose)
			ik_sols = [response.Sol1, response.Sol2, response.Sol3, response.Sol4, response.Sol5, response.Sol6, response.Sol7, response.Sol8]
			
			# select closest, different joint state
			new_joint_state = []
			lower_error = 100
			for sol in ik_sols:
				print(sol)
				# skip if empty list
				if not sol:
					continue
				mse = mean_squared_error(sol, old_joint_state)
				# skip previous joint state as we wish to change configurations
				print(sol)
				if mse < 0.001:
					continue
				print(sol)
				# select closest other js
				if mse < lower_error:
					new_joint_state = sol
					lower_error = mse

			# if a joint state has been found, exectute path planning in JS
			if new_joint_state:
				print(new_joint_state)
				# define variables for path planning message
				# can alternatively just call joint change method for simple relative joint control
				joint_pos = new_joint_state
				pose = None
				speed = 1
				add_waypoint_only = 0
				add_as_joint_space = 0
				clear_all_waypoints = 0
				clear_all_plans = 0
				exclude_gripper = 0
				planner = ''
				plan_time_limit = 0
				joint_space_bound = [math.radians(d) for d in [10, 10, 10, 45, 45, 45]]

				# plan change in joint space
				rospy.wait_for_service('ur5_control/plan/queue')
				plan_response = self.plan_queue(joint_pos, pose, speed, add_waypoint_only, add_as_joint_space, clear_all_waypoints, clear_all_plans, exclude_gripper, planner, plan_time_limit, joint_space_bound)
				
				# execute
				rospy.wait_for_service('ur5_control/plan/execute')
				print(self.plan_execute(1))

		# catch any errors
		except rospy.ServiceException as e:
			print("service call failed: %s"%e)

	# converts rotation and translation infomration to a pose message
	def get_pose_msg(self, rel_rotation, rel_translation):
		q_pose = Quaternion(matrix = rel_rotation)
		pose_msg = Pose()
		# rotation
		pose_msg.orientation.x = q_pose.x
		pose_msg.orientation.y = q_pose.y
		pose_msg.orientation.z = q_pose.z
		pose_msg.orientation.w = q_pose.w
		# translation
		pose_msg.position.x = rel_translation[0]
		pose_msg.position.y = rel_translation[1]
		pose_msg.position.z = rel_translation[2]

		# return the relative rotation/translation as pose and the change in wrist 3
		return pose_msg

	# the following functions plan using RRTConnect a JS path to the given point,
	# and execute if successful.
	def move_to_point(self, point):
		# plan to point
		plan_response = self.plan_to_point(point)
		
		# execute if successful
		if plan_response.PlanSuccess:
			start = time()
			rospy.wait_for_service('ur5_control/plan/execute')
			self.plan_execute(1)
			self.clear_plans()
			exec_time = time() - start
			return 1, exec_time
		else:
			self.clear_plans()
			return 0, 0

	def plan_to_point(self, point):
		joint_pos = point
		pose = None
		speed = 1
		add_waypoint_only = 0
		add_as_joint_space = 0
		clear_all_waypoints = 0
		clear_all_plans = 0
		exclude_gripper = 0
		planner = ''
		plan_time_limit = 0
		joint_space_bound = [math.radians(d) for d in [10, 10, 10, 45, 45, 45]]

		# plan change in joint space
		rospy.wait_for_service('ur5_control/plan/queue')
		plan_response = self.plan_queue(joint_pos, pose, speed, add_waypoint_only, add_as_joint_space, clear_all_waypoints, clear_all_plans, exclude_gripper, planner, plan_time_limit, joint_space_bound)
		return plan_response

	def clear_plans(self):
		joint_pos = self.ur5_home
		pose = None
		speed = 1
		add_waypoint_only = 0
		add_as_joint_space = 0
		clear_all_waypoints = 0
		clear_all_plans = 1
		exclude_gripper = 0
		planner = ''
		plan_time_limit = 0
		joint_space_bound = [math.radians(d) for d in [10, 10, 10, 45, 45, 45]]

		# plan change in joint space
		rospy.wait_for_service('ur5_control/plan/queue')
		plan_response = self.plan_queue(joint_pos, pose, speed, add_waypoint_only, add_as_joint_space, clear_all_waypoints, clear_all_plans, exclude_gripper, planner, plan_time_limit, joint_space_bound)
		return plan_response

	# this function saves static transformation matrices useful for converting the omni
	# pose between frames
	def get_static_tf(self):
		# used to construct the static variables requied to convert omni pose 
		# to the required frame

		# get wrist_3 to ee
		q_w3_ee = Quaternion(0.5, 0.5, -0.5, 0.5)
		self.T_w3_ee = q_w3_ee.transformation_matrix

		# get base_link to mobile_base
		q_w_mb = Quaternion(0.923879532511, 0, 0, -0.382683432365)
		T_w_mb = q_w_mb.transformation_matrix

		self.T_mb_bl = np.linalg.inv(T_w_mb)	

################### SUBSCRIBER FUNCTIONS #####################
	
	# called when one of the buttons on the omni is pressed to enable and disable control
	# mode
	def update_control(self, data):
		# grey button enables control mode, and saves the initial omni and u5 joint state
		if data.grey_button == 1:
			self.relative_omni_js = self.omni_js.copy()
			self.relative_ur5_js = self.ur5_js.copy()
			self.enable_control = True
			print('set path')
		
		# white button disables control mode, saves the demonstration to file
		# and replays the demonstration with smooth JS plan
		if data.white_button == 1:
			print('end and execute')
			if not self.enable_control:
			
				# replay demonstrated trajectory
				if self.picking_trajectory:
					
					# convert current UR5 JS to a list
					goal_joint_states = []
					for i in range(len(self.ur5_joint_names)):
						goal_joint_states.append(self.ur5_js[self.ur5_joint_names[i]])
					
					# move back home
					self.move_to_point(self.ur5_home)
					
					# replay and save demo
					self.save_picking_trajectory()

				# move back home
				self.move_to_point(self.ur5_home)
				
				# update marker
				self.add_marker()
				
			else:
				self.enable_control = False

	# updates ur5 joint space when published
	# if save_RRT_flag has been set to True, it also adds the current position to 
	# a list that is later saved to file
	def update_ur5_js(self, data):
		# update omni joint state parameters when they are change
		for i in range(len(self.ur5_joint_names)):
			self.ur5_js[self.ur5_joint_names[i]] = data.position[i]

		if self.save_RRT_flag:
			self.RRT_trajectory.append(data.position)

	# saves the transformation matrix between the point_cloud frame and the world frame
	# required to express the point cloud scan in the world frame
	def process_tf(self, data):
		for trans in data.transforms:
			if trans.header.frame_id == 'world':
				if trans.child_frame_id == 'point_cloud':
					
					self.world_to_pc = trans.transform
					# save as transformation matrix
					w_p_quaternion = Quaternion(self.world_to_pc.rotation.w, self.world_to_pc.rotation.x, self.world_to_pc.rotation.y, self.world_to_pc.rotation.z)
					w_p_rotation = w_p_quaternion.rotation_matrix
					self.w_p_transformation = w_p_quaternion.transformation_matrix
					w_p_translation = [self.world_to_pc.translation.x, self.world_to_pc.translation.y, self.world_to_pc.translation.z, 1]
					self.w_p_transformation[:, -1] = np.array(w_p_translation)

					print('Saved point cloud tf')
					self.sub_tf.unregister()


	# updates omni joint space when published
	def update_omni_js(self, data):
		# update omni joint state parameters when they are change
		for i in range(len(self.omni_joint_names)):
			self.omni_js[self.omni_joint_names[i]] = data.position[i]

######################### MISCELLANEOUS FUNCTIONS ########################################
	
	# converts the lidar data to the world frame
	def transform_lidar_data(self):
		self.point_cloud_w = []
		self.closest_point = [10, 10, 10]

		# apply translation
		i = 0
		for point in self.convert_pointcloud.all_data_array:
			# transform point
			point = np.append(point, 1)
			point_world = np.dot(self.w_p_transformation, point)
			# append to new point cloud list
			self.point_cloud_w.append(point_world[0:3])
			# update closest point for display
			if np.linalg.norm(point_world) < np.linalg.norm(self.closest_point):
				self.closest_point = point_world[0:3]

	# calculates the repulsive and attractive vectors used in LfD so they can be visualised during
	# demonstration
	def calculate_potential_field(self, pose):
		self.picking_trajectory.append(pose)

		# calculate attractive and repulsive verctors
		self.calculate_arrows(pose)
		
		self.add_marker_array()
		self.attraction_arrow = None 
		self.repulsive_arrow_ee = None
		self.repulsive_arrow_goal = None
		self.repulsive_arrow_closest = None 

	def calculate_arrows(self, pose):
		# extract the attractive vector
		current_pos = [-pose[3], -pose[7], pose[11]]
		attract_vector = np.subtract(self.goal_pos, current_pos)
		self.attract_vector = [x/(4*np.linalg.norm(attract_vector)) for x in attract_vector]
		self.update_attractive_arrow(self.goal_pos, self.attract_vector)
		self.attractive_list.append(self.goal_pos)

		# repulsive vector based on weighted distance from ee
		self.calculate_weighted_centre(current_pos)
		self.repulsive_vector_ee = np.subtract(self.weighted_average_point, current_pos)
		self.repulsive_vector_ee = [x/(4*np.linalg.norm(self.repulsive_vector_ee)) for x in self.repulsive_vector_ee]
		self.repulsive_arrow_ee = self.update_repulsive_arrow(self.weighted_average_point, self.repulsive_vector_ee)
		self.repulsive_ee_list.append(self.weighted_average_point)
		
		# repulsive vector based on closest point
		self.closest_vector = np.subtract(self.closest_point, current_pos)
		self.closest_vector = [x/(4*np.linalg.norm(self.closest_vector)) for x in self.closest_vector]
		self.repulsive_arrow_closest = self.update_repulsive_arrow(self.closest_point, self.closest_vector)
		self.repulsive_closest_list.append(self.closest_point)

		# repulsive vector based on weighted distance from goal
		self.calculate_weighted_centre(current_pos, goal_flag=True)
		self.repulsive_vector_goal = np.subtract(self.weighted_average_point, current_pos)
		self.repulsive_vector_goal = [x/(4*np.linalg.norm(self.repulsive_vector_goal)) for x in self.repulsive_vector_goal]
		self.repulsive_arrow_goal = self.update_repulsive_arrow(self.weighted_average_point, self.repulsive_vector_goal)
		self.repulsive_goal_list.append(self.weighted_average_point)

	def calculate_weighted_centre(self, current_pos, goal_flag=False):	
		# extract the repulsive vector from the point cloud data
		inverse_distance_sum = 0
		self.weighted_average_point = [0, 0, 0]
		self.max_repulsion_dist = 0.5
		self.closest_poimt = None 
		closest_distance = 100
		# iterate through each point
		for point in self.point_cloud_w:
			# either consider points close to goal or ee
			if goal_flag:
				goal_to_point = np.subtract(self.goal_pos, point)
				dist = np.linalg.norm(goal_to_point)
			else:
				ee_to_point = np.subtract(current_pos, point)
				dist = np.linalg.norm(ee_to_point)
			# if point is close to goal/ee, point contributes to weighted centre
			if (dist < self.max_repulsion_dist):
				inverse_distance_sum += (1/dist)
				for i in range(len(point)):
					self.weighted_average_point[i] += point[i]*(1/dist)
				if dist<closest_distance:
					self.closest_point = point
					closest_distance = dist
		try:
			self.weighted_average_point = [x/inverse_distance_sum for x in self.weighted_average_point]
		except ZeroDivisionError:
			self.weighted_average_point = [0,0,0]
			self.closest_point = [0, 0, 0]

	# adds a random goal position in front of the robot
	def add_marker(self):
		marker = Marker()
		marker_array = MarkerArray()
		marker.id = 0
		marker.action = Marker.DELETEALL
		marker_array.markers.append(marker)
		self.marker_array_publisher.publish(marker_array)

		x = float(randint(25,75))/100
		y = -float(randint(25,75))/100
		z = float(randint(25,75))/100

		self.goal_pos = [x, y, z]
		self.define_marker(self.goal_pos)
		self.add_marker_array()

	# functions for adding a series of markers to the simulation environment
	def add_marker_array(self):
		i = 0
		marker_array = MarkerArray()
		for marker in [self.goal_marker, self.attraction_arrow, self.repulsive_arrow_ee, self.repulsive_arrow_goal, self.repulsive_arrow_closest]:
			if marker:
				marker.id = i
				marker_array.markers.append(marker)
				i += 1
		self.marker_array_publisher.publish(marker_array)

	def define_marker(self, pos):
		marker = Marker()
		marker.header.frame_id = "/base_link"
		marker.type = marker.SPHERE
		marker.action = marker.ADD
		marker.scale.x = 0.05
		marker.scale.y = 0.05
		marker.scale.z = 0.05
		marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.8)
		marker.pose.orientation.w = 1.0
		marker.pose.position.x = pos[0]
		marker.pose.position.y = pos[1]
		marker.pose.position.z = pos[2]
		self.goal_marker = marker

	def update_attractive_arrow(self, pos, vector):
		marker = Marker()
		marker.header.frame_id = "/base_link"
		marker.type = marker.ARROW
		marker.action = marker.ADD
		marker.scale.x = 0.02
		marker.scale.y = 0.02
		marker.scale.z = 0.02
		marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.8)
		
		start_point = Point()
		start_point.x = pos[0] - vector[0]
		start_point.y = pos[1] - vector[1]
		start_point.z = pos[2] - vector[2]
		end_point = Point()
		end_point.x = pos[0]
		end_point.y = pos[1]
		end_point.z = pos[2]
		marker.points = [start_point, end_point]
		self.attraction_arrow = marker

	def update_repulsive_arrow(self, pos, vector):
		marker = Marker()
		marker.header.frame_id = "/base_link"
		marker.type = marker.ARROW
		marker.action = marker.ADD
		marker.scale.x = 0.02
		marker.scale.y = 0.02
		marker.scale.z = 0.02
		marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
		
		end_point = Point()
		end_point.x = pos[0] - vector[0]
		end_point.y = pos[1] - vector[1]
		end_point.z = pos[2] - vector[2]
		start_point = Point()
		start_point.x = pos[0]
		start_point.y = pos[1]
		start_point.z = pos[2]
		marker.points = [start_point, end_point]
		return marker

	# called upon completion of a demonstration to save the trajectory, point cloud, and goal
	# to file so that it can be used to train the LfD model
	def save_picking_trajectory(self):
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		current_time = current_time.replace(':', '-')

		# save trajectory to file
		traj_path = '/home/nic/catkin_ws/src/haptic_control/scripts/trajectories/' + current_time + '.pkl'
		with open(traj_path, 'wb') as f:
			pickle.dump(self.picking_trajectory, f)

		# save all training data to file
		training_path = '/home/nic/catkin_ws/src/haptic_control/training_trajectories/' + current_time + '.pkl'
		data_df = pd.concat([pd.Series(self.picking_trajectory), pd.Series(self.repulsive_ee_list), pd.Series(self.repulsive_goal_list), pd.Series(self.repulsive_closest_list), pd.Series(self.attractive_list)], axis=1, keys=['traj', 'repulsive_ee', 'repulsive_goal', 'repulsive_closest', 'attractive'])
		# for col in data_df.columns:
			# print(data_df.head()[col])
		data_df.to_pickle(training_path)

		# save point cloud and goal to file
		point_cloud_path = '/home/nic/catkin_ws/src/haptic_control/training_trajectories/' + current_time + '_point_cloud.pkl'		
		data_df = pd.DataFrame(data = self.point_cloud_w)
		# print(data_df.head())
		data_df.to_pickle(point_cloud_path)

		# save point cloud and goal to file
		goal_path = '/home/nic/catkin_ws/src/haptic_control/training_trajectories/' + current_time + '_goal.pkl'		
		data_df = pd.DataFrame(data = self.goal_pos)
		# print(data_df.head())
		data_df.to_pickle(goal_path)

		self.repulsive_goal_list = []
		self.repulsive_ee_list = []
		self.repulsive_closest_list = []
		self.attractive_list = []
		self.picking_trajectory = []

		return traj_path

###################### VALIDATION FUNCTIONS ############################################

	# the following functions add/remove the point cloud to the collision object, so that 
	# RRTConnect considers it as an obstacle when planning
	def add_lidar_collision(self, remove=False):
		i = 0
		for point in self.point_cloud_w:
			if remove:
				self.remove_collision_point('lida_' + str(i), point)
				sleep(0.001)
			else:	
				self.add_collision_point('lida_' + str(i), point)
			i+=1
		sleep(2)

	def add_collision_point(self, name, pos):
		co = CollisionObject()
		co.operation = CollisionObject.ADD 
		co.id = name

		pose_msg = Pose()
		# rotation
		pose_msg.orientation.x = 0
		pose_msg.orientation.y = 0
		pose_msg.orientation.z = 0
		pose_msg.orientation.w = 1
		# translation
		pose_msg.position.x = pos[0]
		pose_msg.position.y = pos[1]
		pose_msg.position.z = pos[2]

		co.header.frame_id = 'world'

		sphere=SolidPrimitive()
		sphere.type = SolidPrimitive.SPHERE
		sphere.dimensions = [0.01]
		co.primitives = [sphere]
		co.primitive_poses = [pose_msg]

		self.collision_object_pub.publish(co)

	def remove_collision_point(self, name, pos):
		co = CollisionObject()
		co.operation = CollisionObject.REMOVE
		co.id = name

		pose_msg = Pose()
		# rotation
		pose_msg.orientation.x = 0
		pose_msg.orientation.y = 0
		pose_msg.orientation.z = 0
		pose_msg.orientation.w = 1
		# translation
		pose_msg.position.x = pos[0]
		pose_msg.position.y = pos[1]
		pose_msg.position.z = pos[2]

		co.header.frame_id = 'world'

		sphere=SolidPrimitive()
		sphere.type = SolidPrimitive.SPHERE
		sphere.dimensions = [0.01]
		co.primitives = [sphere]
		co.primitive_poses = [pose_msg]

		self.collision_object_pub.publish(co)

	# this function is called in omni_to_ur5_sim.py if the validation flag is set to True
	# it imports the trajectories produced by the LfD model, runs them with a smooth JS
	# it then plans an RRT path to the final pose for comparison of success rate, planning time
	# execution time.
	def run_final_validation(self, folder):
		onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
		for file in onlyfiles:
			if 'goal' not in file:
				continue

			file_name = file[0:8]
			print(file_name)
			goal_df = pd.read_csv(folder + file_name + '_goal_pos.csv')
			goal = [goal_df['0'].iloc[0], goal_df['0'].iloc[1], goal_df['0'].iloc[2]]
			self.imported_point_cloud = pd.read_pickle(folder+file_name+ '_point_cloud.pkl')
			self.imported_point_cloud = [self.imported_point_cloud.iloc[i].values for i in range(self.imported_point_cloud.shape[0])]
			self.point_cloud_w = self.imported_point_cloud

			times_lfd = []
			times_rrt = []
			collision_points_list = []
			for i in range(10):
				traj_df = pd.read_csv(folder + file_name + '_traj_' + str(i) + '.csv')
				traj_list = [traj_df.iloc[j].values for j in range(len(traj_df.index))]
				path = folder + file_name + '_traj_' + str(i) + '.pkl'
				with open(path, 'wb') as f:
					pickle.dump(traj_list, f)
				
				time_lfd, js_difference_lfd, time_rrt, js_difference_rrt, collision_points = self.replay_final_validation(path, goal = goal)
				
				print('\n')
				print('lfd times')
				print(time_lfd)
				print('time rrt')
				print(time_rrt)
				print('collision points')
				print(collision_points)

				times_lfd.append(time_lfd)
				times_rrt.append(time_rrt)
				collision_points_list.append(collision_points)

				js_difference_df = pd.DataFrame(js_difference_lfd)
				js_difference_df.to_pickle(folder + 'lfd output/' + file_name + '_js_difference_' + str(i) + '.pkl')
				
				js_difference_df = pd.DataFrame(js_difference_rrt)
				js_difference_df.to_pickle(folder + 'rrt output/' + file_name + '_js_difference_' + str(i) + '.pkl')

				self.move_to_point(self.ur5_home)

			time_df = pd.DataFrame(times_lfd)
			print(time_df)
			time_df.to_pickle(folder + 'lfd output/' + file_name + '_times.pkl')

			time_df = pd.DataFrame(times_rrt)
			print(time_df)
			time_df.to_pickle(folder + 'rrt output/' + file_name + '_times.pkl')

			collision_points_df = pd.DataFrame(collision_points_list)
			print(collision_points_df)
			collision_points_df.to_pickle(folder + 'lfd output/' + file_name + '_collision_points.pkl')

	# replays the generated LfD trajectory imported from file
	def replay_final_validation(self, traj_path, goal=False):
		start = time()
		# generate pose conversion object:
		self.PoseConversionObject = PoseConversion(traj_path, goal=goal)
		# self.PoseConversionObject.plan_execute_real_time()
		self.PoseConversionObject.convert_pose()

		js_conversion_time = time() - start

		# call function to execute js trajectory using plan and execute methods
		self.move_to_point(self.PoseConversionObject.js_trajectory[0])

		# generate collision free traj and record for playback
		self.save_RRT_flag = True

		start = time()
		# generate collision free path, and time to plan it (total - time spent actuating RVIZ)
		self.recorded_execution_time = 0
		js_difference = self.js_trajectory(self.PoseConversionObject.js_trajectory, [self.ur5_js[name] for name in self.ur5_joint_names], self.rate)
		added_plan_time = time()-start-self.recorded_execution_time
		self.save_RRT_flag = False

		js_conversion_time += added_plan_time

		# downsample generated trajectory and execute
		recorded_traj = self.downsample_recorded_trajectory()
		# check recorded trajectory doesn't collide with the environment
		self.move_to_point(recorded_traj[0])

		# record execution time
		start = time()
		js_difference = self.traj.js_trajectory(recorded_traj, [self.ur5_js[name] for name in self.ur5_joint_names], self.rate)
		execution_time = time() - start

		js_difference_RRT, planning_time_RRT, execution_time_RRT, collision_points = self.RRT_and_collisions(recorded_traj, goal)

		return [js_conversion_time, execution_time], js_difference, [planning_time_RRT, execution_time_RRT], js_difference_RRT, collision_points

	# checks collisions in the LfD path, adds collision objects and calls the RRTConnect planner
	# returs the planning and execution time of the RRTConnect, and the collision points of the LfD
	def RRT_and_collisions(self, trajectory, goal):
		# add lidar_collisions
		self.move_to_point(trajectory[0])
		self.add_lidar_collision()

		# check if RRT can make it to the point
		RRT_goal = trajectory[0]
		collision_points = []
		for i in range(len(trajectory)):
			js = trajectory[i]
			response, exection = self.move_to_point(js)
			if not response:
				# calculate goal pose
				response = self.calculate_ur5_fk(js)
				ee_pose = list(response.Pose)
				dist = np.linalg.norm(np.subtract([-ee_pose[3], -ee_pose[7], ee_pose[11]], goal))
				collision_points.append((float(i+1)/len(trajectory)))
				collision_points.append(dist)
			else:
				RRT_goal = js

		# plan, execute and save RRT path
		self.move_to_point(self.ur5_home)
		planning_time, RRT_js = self.plan_RRT_trajectory(RRT_goal, self.ur5_home)

		# now that the plan has been generated, remove the collision
		self.add_lidar_collision(remove=True)

		# record execution time
		if RRT_js:
			self.move_to_point(RRT_js[0])
			start = time()
			js_difference = self.traj.js_trajectory(RRT_js, [self.ur5_js[name] for name in self.ur5_joint_names], self.rate)
			execution_time = time() - start
		else:
			js_difference = []
			execution_time = 0

		return js_difference, planning_time, execution_time, collision_points

	# plans an Js trajectoring using RRT connect from the defined start js to end js
	# runs configuration matching to ensure the planned path is as efficient as possible
	def plan_RRT_trajectory(self, goal_js, start_pos):
		
		start = time()
		move_time = 0
		planning_time = 0

		# calculate goal pose
		response = self.calculate_ur5_fk(goal_js)
		ee_pose = list(response.Pose)

		response = self.calculate_ur5_ik(ee_pose)
		ik_sols_goal = [response.Sol1, response.Sol2, response.Sol3, response.Sol4, response.Sol5, response.Sol6, response.Sol7, response.Sol8]
		
		# calculate start pose
		response = self.calculate_ur5_fk(start_pos)
		ee_pose = list(response.Pose)

		response = self.calculate_ur5_ik(ee_pose)
		ik_sols_start = [response.Sol1, response.Sol2, response.Sol3, response.Sol4, response.Sol5, response.Sol6, response.Sol7, response.Sol8]
		
		# config matching
		configs_goal = []
		goal_sols = []
		for sol in ik_sols_goal:
			if sol:
				n = self.calc_config_number(sol)
				configs_goal.append(n)
				goal_sols.append(sol)

		configs_start = []
		start_sols = []
		for sol in ik_sols_start:
			if sol:
				n = self.calc_config_number(sol)
				configs_start.append(n)
				start_sols.append(sol)

		# matching configs
		intersection = set(configs_start).intersection(configs_goal)

		search_configs = list(intersection)
		for n in configs_goal:
			if n in search_configs:
				continue
			search_configs.append(n)

		solution_found = False

		for config in search_configs:
			index = configs_goal.index(config)
			sol = goal_sols[index]

			try:
				index = configs_start.index(config)
				start_position = start_sols[index]
			except ValueError:
				start_position = start_sols[0]

			# print(start_position)
			# print(sol)
			
			move_start = time()
			self.move_to_point(start_position)	
			move_time += (time()- move_start)		

			joint_pos = sol
			pose = None
			speed = 1
			add_waypoint_only = 0
			add_as_joint_space = 0
			clear_all_waypoints = 0
			clear_all_plans = 0
			exclude_gripper = 0
			planner = ''
			plan_time_limit = 0
			joint_space_bound = [math.radians(d) for d in [10, 10, 10, 45, 45, 45]]

			# plan change in joint space
			rospy.wait_for_service('ur5_control/plan/queue')
			plan_response = self.plan_queue(joint_pos, pose, speed, add_waypoint_only, add_as_joint_space, clear_all_waypoints, clear_all_plans, exclude_gripper, planner, plan_time_limit, joint_space_bound)

			planning_time = time()-start-move_time

			if plan_response.PlanSuccess:
				start = time()
				# start recording RRT path
				self.save_RRT_flag = True

				# plan change in joint space
				rospy.wait_for_service('ur5_control/plan/execute')
				self.plan_execute(1)

				while True:
					error = np.linalg.norm(np.subtract(self.RRT_trajectory[-1], sol))
					if error < 0.001:
						break

				execution_time = time()-start

				# stop recording RRT trajectry
				self.save_RRT_flag = False

				ts_traj = self.downsample_recorded_trajectory()

				self.clear_plans()
				solution_found = True
				break

		if not solution_found:
			ts_traj = []
			print('\n\nNo Solution Found!!!\n\n')

		return planning_time, ts_traj

	# scripts to return configuration number
	def calc_config_number(self, js):
		# declare joint config counter
		config = 1
		th_plus, th_minus = self.th1_solution(js)
		# check first joint configuration
		if abs(js[0] - th_plus) < 0.001:
			# check third joint
			if js[2] >= 0:
				# check fifth joint
				if js[4] < 0:
					config += 1 
			else:
				config += 2
				# check fifth joint
				if js[4] < 0:
					config += 1

		else:
			config+=4
			# check third joint
			if js[2] >= 0:
				# check fifth joint
				if js[4] < 0:
					config += 1 
			else:
				config += 2
				# check fifth joint
				if js[4] < 0:
					config += 1

		return config

	# solution for th1 taken from: http://rasmusan.blog.aau.dk/files/ur5_kinematics.pdf
	# needed to calculate robot joint configuration
	def th1_solution(self, th_n):
		d1 = 0.089159
		a2 = -0.425
		a3 = -0.39225
		d4 = 0.10915
		d5 = 0.09465
		d6 = 0.0823

		s = [math.sin(th) for th in th_n]
		[s1, s2, s3, s4, s5, s6] = s
		c = [math.cos(th) for th in th_n]
		[c1, c2, c3, c4, c5, c6] = c

		x = 1.0*d5*(c4*(1.0*c1*c2*s3 + 1.0*c1*c3*s2) - 1.0*s4*(1.0*c1*s2*s3 - c1*c2*c3)) + d6*(1.0*c5*s1 + 1.0*s5*(s4*(1.0*c1*c2*s3 + 1.0*c1*c3*s2) + c4*(1.0*c1*s2*s3 - c1*c2*c3))) - 1.0*a3*(1.0*c1*s2*s3 - c1*c2*c3) - d6*(1.0*c5*s1 + 1.0*s5*(s4*(1.0*c1*c2*s3 + 1.0*c1*c3*s2) + c4*(1.0*c1*s2*s3 - c1*c2*c3))) + 1.0*d4*s1 + 1.0*a2*c1*c2

		y =  1.0*d5*(c4*(1.0*c2*s1*s3 + 1.0*c3*s1*s2) - 1.0*s4*(1.0*s1*s2*s3 - c2*c3*s1)) - d6*(1.0*c1*c5 - 1.0*s5*(s4*(1.0*c2*s1*s3 + 1.0*c3*s1*s2) + c4*(1.0*s1*s2*s3 - c2*c3*s1))) - 1.0*a3*(1.0*s1*s2*s3 - c2*c3*s1) - 1.0*d4*c1 + d6*(1.0*c1*c5 - 1.0*s5*(s4*(1.0*c2*s1*s3 + 1.0*c3*s1*s2) + c4*(1.0*s1*s2*s3 - c2*c3*s1))) + 1.0*a2*c2*s1

		th_plus = math.atan2(y,x) + math.acos(d4/math.sqrt(math.pow(x,2)+math.pow(y,2))) + math.pi/2
		th_minus = math.atan2(y,x) - math.acos(d4/math.sqrt(math.pow(x,2)+math.pow(y,2))) + math.pi/2

		# bring to within -pi, pi
		if th_plus > math.pi:
			th_plus = th_plus - math.pi*2

		if th_minus < -math.pi:
			th_minus = math.pi*2 - th_minus

		return th_plus, th_minus

	# changes a recorded trajectory to delete duplicate positions
	def downsample_recorded_trajectory(self):
		# only include new js
		downsample_traj = [self.RRT_trajectory[0]]
		for js in self.RRT_trajectory:
			if mean_squared_error(js, downsample_traj[-1]) >0.001:
				downsample_traj.append(js)

		self.RRT_trajectory = []

		return downsample_traj

	# script to execute a JS trajectory using low level trajectory controller
	# checks for self-collisions and runs RRTConnect if necessary
	def js_trajectory(self, trajectory, current_pos, rate, total_diff=[]):
		t = 0
		self.traj.clear()
		self.traj.add_point(current_pos, t)
		rate = 0.15
		count = -1
		start = time()
		for js in trajectory:
			count+=1
			difference = np.subtract(js, current_pos)
			max_vel = max(difference)

			# check collision of point
			collision = self.traj._collision_check(js, False)
			if collision.Collision[0]:
				print('js is a collision, skip')
				continue

			# check interpolated collision
			failed = False
			for i in range(3):
				check_interpolated_js = np.add(current_pos, [i*x/3 for x in difference])
				# check collision
				collision = self.traj._collision_check(check_interpolated_js, False)
				# print(check_interpolated_js)

				# return if collision
				if collision.Collision[0]:
					print('collision')
					# execute
					self.traj.start_topic()
					sleep(t)
					self.recorded_execution_time += t
					self.traj.cancel_publisher.publish(self.traj.past_goal_id)
					# call ik
					response, exec_time = self.move_to_point(js)
					self.recorded_execution_time += exec_time
					if not response:
						failed = True
						response, exec_time = self.move_to_point(self.ur5_home)
						self.recorded_execution_time += exec_time
						self.RRT_trajectory = []
						break
					# recall js_trajectory
					total_diff = self.js_trajectory(trajectory[count+1:], js, rate=self.rate, total_diff=total_diff)
					return total_diff

			if failed:
				print('Failed')
				continue

			# update the rate, with max joint velocity of 2pi/sec
			# rate = max_vel/(2*math.pi)
			t+=rate
			self.traj.add_point(js, t)
			total_diff.append(difference)

			current_pos = js

		self.traj.start_topic()
		sleep(t)
		self.traj.cancel_publisher.publish(self.traj.past_goal_id)
		self.recorded_execution_time += t

		return total_diff


if __name__ == '__main__':
	try:
		omniUR5Object = OmniToUR5()
		omniUR5Object.omni_to_ur5()
	except rospy.ROSInterruptException:
		pass
