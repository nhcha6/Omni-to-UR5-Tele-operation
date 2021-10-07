import rospy
import actionlib
from copy import copy
import argparse
from moveit_msgs.msg import ExecuteTrajectoryAction, ExecuteTrajectoryGoal, ExecuteTrajectoryActionGoal
from actionlib_msgs.msg import GoalID
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from ur5_control.srv import CollCheck
import time
from matplotlib import pyplot as plt
import math
import sys
from sklearn.metrics import mean_squared_error
import numpy as np
import math
from time import sleep

# class for implementing low level control of the simulated UR5 robotic arm
class Trajectory(object):
	def __init__(self, ):
		# start server and required variables
		self._client = actionlib.SimpleActionClient('execute_trajectory', ExecuteTrajectoryAction)
		self._goal = ExecuteTrajectoryGoal()
		self._goal.trajectory.joint_trajectory.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
		self.goal_publisher = rospy.Publisher('execute_trajectory/goal', ExecuteTrajectoryActionGoal)
		self.cancel_publisher = rospy.Publisher('execute_trajectory/cancel', GoalID)
		self._goal_message = ExecuteTrajectoryActionGoal()
		self.past_goal_id = None
		self.max_vel = 5

		# collision object server
		self._collision_check = rospy.ServiceProxy('/ur5_control/scene/collision/check', CollCheck)

		# check server is launched
		server_up = self._client.wait_for_server(timeout=rospy.Duration(10.0))
		if not server_up:
			rospy.logerr("Timed out waiting for Joint Trajectory"
						 " Action Server to connect. Start the action server"
						 " before running example.")
			rospy.signal_shutdown("Timed out waiting for Action Server")
			sys.exit(1)

	# adds a point to the trajectory goal
	def add_point(self, positions, time):
		point = JointTrajectoryPoint()
		point.positions = copy(positions)
		point.time_from_start = rospy.Duration(time)
		self._goal.trajectory.joint_trajectory.points.append(point)

	# set the velocity of the ur5 using the trajectory goal
	def add_velocity(self, current_pos, velocities, time_start, time_end):
		# set the velocity at the current point
		point = JointTrajectoryPoint()
		point.positions = current_pos
		point.velocities = velocities
		point.time_from_start = rospy.Duration(time_start)
		self._goal.trajectory.joint_trajectory.points.append(point)

		# new pos to move towards and stop velocity
		delta_pos = [x*(time_end-time_start) for x in velocities]
		new_pos = [sum(x) for x in zip(delta_pos, current_pos)]
		self.add_point(new_pos, time_end)

		# return the position that the robot will finish at
		return new_pos

	# send the trajectory goal to the trajectory action client
	def start(self):
		self._client.send_goal(self._goal)

	# command the trajectory action client to execute the goal
	def start_topic(self):
		current_time = rospy.Time.now()
		self._goal.trajectory.joint_trajectory.header.stamp = current_time
		self._goal_message.goal = self._goal
		self._goal_message.header.stamp = current_time
		# goal ID
		self.past_goal_id = GoalID()
		# self.past_goal_id.stamp = current_time
		self.past_goal_id.id = 'topic_goal-' + str(time.time())
		self._goal_message.goal_id = self.past_goal_id
		self.goal_publisher.publish(self._goal_message)

	# cancel exectution of the goal
	def stop(self):
		self._client.cancel_goal()

	# clears the trajectory action goal
	def clear(self):
		self._goal = ExecuteTrajectoryGoal()
		self._goal.trajectory.joint_trajectory.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

	# wait for the goal to be completed
	def wait(self, timeout=15.0):
		self._client.wait_for_result(timeout=rospy.Duration(timeout))

	# return the result of the trajectory execution
	def result(self):
		return self._client.get_result()

	# called by omni_to_ur5_sim to execute a velocity command an check for collision/singularity
	# if successful, the start command will be called in omni_to_ur5_sim.py
	def move_robot_js(self, vel, current_angles, rate=0.5):
		# command the robot to execute the desired velocity
		new_pos = self.add_velocity(current_angles, vel, 0, rate)
		
		# avoid execution if calculated velocity is too large
		if max(vel) > self.max_vel:
			print('velocity too large')
			print(vel)
			return new_pos, False

		# check collision: only check final point, as the workflow is designed for small changes
		collision = self._collision_check(new_pos, False)

		# return if collision
		if collision.Collision[0]:
			print('collision')
			return new_pos, False

		print("current js pos")
		print(current_angles)
		print('new pos')
		print(new_pos)

		return new_pos, True

	# this function executes a planned js trajectory, with the max joint velocity set
	# to be half a rotation per second.
	def js_trajectory(self, trajectory, current_pos, rate):
		t = 0
		total_diff = []

		# clear trajectory goal and add start point
		self.clear()
		self.add_point(current_pos, t)
		
		# iterate through trajectory
		for js in trajectory:
			# calculate the max velocity
			difference = np.subtract(js, current_pos)
			max_vel = max([abs(diff) for diff in difference])
			total_diff.append(difference)
			
			# update the rate, with max joint velocity of pi/sec
			rate = max_vel/(math.pi)
			t+=rate
			self.add_point(js, t)

			# update current position
			current_pos = js

		# execute the trajectory goal and sleep for the time taken to run it
		self.start_topic()
		time.sleep(t)
		self.cancel_publisher.publish(self.past_goal_id)

		return total_diff

# ur5 joint state service callback function
def update_ur5_js(data):
	global current_angles 
	global position_list
	current_angles = data.position
	position_list.append(current_angles[0])
