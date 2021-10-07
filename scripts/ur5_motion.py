# coding: utf-8 
import math
import numpy as np
from pyquaternion import Quaternion
from time import time

# class for storing the solution to the ur5 Jacobian
class UR5Kinematics():
	def __init__(self):
		# define default joint values and names
		self.ur5_joint_vel = {'shoulder_pan_joint': 0.0, 'shoulder_lift_joint':0.1, 'elbow_joint':0.0, 'wrist_1_joint': 0.0, 'wrist_2_joint': 0.0, 'wrist_3_joint': 0.0}
		self.ur5_joint_pos = {'shoulder_pan_joint': -1.0472, 'shoulder_lift_joint':-1.6057, 'elbow_joint':-0.8727, 'wrist_1_joint': -0.2094, 'wrist_2_joint': -1.5708, 'wrist_3_joint': 0.0}
		self.ur5_joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
	
	# called to calculate the ur5 Jacobian or inverse Jacobian given the current joint state
	def apply_jacobian(self, js, inverse=False):
		# store joint angles in list
		th_n = [js[name] for name in self.ur5_joint_names]
		[th1, th2, th3, th4, th5, th6] = th_n

		# link lengths: https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
		d2 = 0.089159
		a3 = -0.425
		a4 = -0.39225
		d5 = 0.10915
		d6 = 0.09465
		d7 = 0.0823
		
		# Jacobian solution	 
		r_x1 = d2*math.cos(th1) + d6*math.cos(th1)*math.cos(th5) + d7*math.cos(th1)*math.cos(th5) - a3*math.cos(th2)*math.sin(th1) - a4*math.cos(th2)*math.cos(th3)*math.sin(th1) + a4*math.sin(th1)*math.sin(th2)*math.sin(th3) - d5*math.cos(th2)*math.cos(th3)*math.sin(th1)*math.sin(th4) - d5*math.cos(th2)*math.cos(th4)*math.sin(th1)*math.sin(th3) - d5*math.cos(th3)*math.cos(th4)*math.sin(th1)*math.sin(th2) + d5*math.sin(th1)*math.sin(th2)*math.sin(th3)*math.sin(th4) + d6*math.cos(th2)*math.cos(th3)*math.cos(th4)*math.sin(th1)*math.sin(th5) + d7*math.cos(th2)*math.cos(th3)*math.cos(th4)*math.sin(th1)*math.sin(th5) - d6*math.cos(th2)*math.sin(th1)*math.sin(th3)*math.sin(th4)*math.sin(th5) - d6*math.cos(th3)*math.sin(th1)*math.sin(th2)*math.sin(th4)*math.sin(th5) - d6*math.cos(th4)*math.sin(th1)*math.sin(th2)*math.sin(th3)*math.sin(th5) - d7*math.cos(th2)*math.sin(th1)*math.sin(th3)*math.sin(th4)*math.sin(th5) - d7*math.cos(th3)*math.sin(th1)*math.sin(th2)*math.sin(th4)*math.sin(th5) - d7*math.cos(th4)*math.sin(th1)*math.sin(th2)*math.sin(th3)*math.sin(th5)
		r_x2 = -math.cos(th1)*(0.5*d6*math.cos(th2 + th3 + th4 + th5) + 0.5*d7*math.cos(th2 + th3 + th4 + th5) + a4*math.sin(th2 + th3) + a3*math.sin(th2) - 0.5*d6*math.cos(th2 + th3 + th4 - th5) - 0.5*d7*math.cos(th2 + th3 + th4 - th5) - 1.0*d5*math.cos(th2 + th3 + th4))
		r_x3 = -math.cos(th1)*(0.5*d6*math.cos(th2 + th3 + th4 + th5) + 0.5*d7*math.cos(th2 + th3 + th4 + th5) + a4*math.sin(th2 + th3) - 0.5*d6*math.cos(th2 + th3 + th4 - th5) - 0.5*d7*math.cos(th2 + th3 + th4 - th5) - 1.0*d5*math.cos(th2 + th3 + th4))
		r_x4 = math.cos(th1)*(0.5*d6*math.cos(th2 + th3 + th4 - th5) - 0.5*d7*math.cos(th2 + th3 + th4 + th5) - 0.5*d6*math.cos(th2 + th3 + th4 + th5) + 0.5*d7*math.cos(th2 + th3 + th4 - th5) + 1.0*d5*math.cos(th2 + th3 + th4))
		r_x5 = (d6 + d7)*(math.cos(th1)*math.cos(th2)*math.cos(th5)*math.sin(th3)*math.sin(th4) - math.cos(th1)*math.cos(th2)*math.cos(th3)*math.cos(th4)*math.cos(th5) - math.sin(th1)*math.sin(th5) + math.cos(th1)*math.cos(th3)*math.cos(th5)*math.sin(th2)*math.sin(th4) + math.cos(th1)*math.cos(th4)*math.cos(th5)*math.sin(th2)*math.sin(th3))
		r_x6 = 0

		r_y1 = d2*math.sin(th1) + a3*math.cos(th1)*math.cos(th2) + d6*math.cos(th5)*math.sin(th1) + d7*math.cos(th5)*math.sin(th1) + a4*math.cos(th1)*math.cos(th2)*math.cos(th3) - 1.0*a4*math.cos(th1)*math.sin(th2)*math.sin(th3) + d5*math.cos(th1)*math.cos(th2)*math.cos(th3)*math.sin(th4) + d5*math.cos(th1)*math.cos(th2)*math.cos(th4)*math.sin(th3) + d5*math.cos(th1)*math.cos(th3)*math.cos(th4)*math.sin(th2) - 1.0*d5*math.cos(th1)*math.sin(th2)*math.sin(th3)*math.sin(th4) - 1.0*d6*math.cos(th1)*math.cos(th2)*math.cos(th3)*math.cos(th4)*math.sin(th5) - 1.0*d7*math.cos(th1)*math.cos(th2)*math.cos(th3)*math.cos(th4)*math.sin(th5) + d6*math.cos(th1)*math.cos(th2)*math.sin(th3)*math.sin(th4)*math.sin(th5) + d6*math.cos(th1)*math.cos(th3)*math.sin(th2)*math.sin(th4)*math.sin(th5) + d6*math.cos(th1)*math.cos(th4)*math.sin(th2)*math.sin(th3)*math.sin(th5) + d7*math.cos(th1)*math.cos(th2)*math.sin(th3)*math.sin(th4)*math.sin(th5) + d7*math.cos(th1)*math.cos(th3)*math.sin(th2)*math.sin(th4)*math.sin(th5) + d7*math.cos(th1)*math.cos(th4)*math.sin(th2)*math.sin(th3)*math.sin(th5)
		r_y2 = -math.sin(th1)*(0.5*d6*math.cos(th2 + th3 + th4 + th5) + 0.5*d7*math.cos(th2 + th3 + th4 + th5) + a4*math.sin(th2 + th3) + a3*math.sin(th2) - 0.5*d6*math.cos(th2 + th3 + th4 - th5) - 0.5*d7*math.cos(th2 + th3 + th4 - th5) - 1.0*d5*math.cos(th2 + th3 + th4))
		r_y3 = -math.sin(th1)*(0.5*d6*math.cos(th2 + th3 + th4 + th5) + 0.5*d7*math.cos(th2 + th3 + th4 + th5) + a4*math.sin(th2 + th3) - 0.5*d6*math.cos(th2 + th3 + th4 - th5) - 0.5*d7*math.cos(th2 + th3 + th4 - th5) - 1.0*d5*math.cos(th2 + th3 + th4))
		r_y4 = math.sin(th1)*(0.5*d6*math.cos(th2 + th3 + th4 - th5) - 0.5*d7*math.cos(th2 + th3 + th4 + th5) - 0.5*d6*math.cos(th2 + th3 + th4 + th5) + 0.5*d7*math.cos(th2 + th3 + th4 - th5) + 1.0*d5*math.cos(th2 + th3 + th4))
		r_y5 = (d6 + d7)*(math.cos(th1)*math.sin(th5) - 1.0*math.cos(th2)*math.cos(th3)*math.cos(th4)*math.cos(th5)*math.sin(th1) + math.cos(th2)*math.cos(th5)*math.sin(th1)*math.sin(th3)*math.sin(th4) + math.cos(th3)*math.cos(th5)*math.sin(th1)*math.sin(th2)*math.sin(th4) + math.cos(th4)*math.cos(th5)*math.sin(th1)*math.sin(th2)*math.sin(th3))
		r_y6 = 0

		r_z1 = 0
		r_z2 = 1.0*a4*math.cos(th2 + th3) - 0.5*d7*math.sin(th2 + th3 + th4 + th5) - 0.5*d6*math.sin(th2 + th3 + th4 + th5) + a3*math.cos(th2) + 0.5*d6*math.sin(th2 + th3 + th4 - th5) + 0.5*d7*math.sin(th2 + th3 + th4 - th5) + 1.0*d5*math.sin(th2 + th3 + th4)
		r_z3 = 1.0*a4*math.cos(th2 + th3) - 0.5*d7*math.sin(th2 + th3 + th4 + th5) - 0.5*d6*math.sin(th2 + th3 + th4 + th5) + 0.5*d6*math.sin(th2 + th3 + th4 - th5) + 0.5*d7*math.sin(th2 + th3 + th4 - th5) + 1.0*d5*math.sin(th2 + th3 + th4)
		r_z4 = 0.5*d6*math.sin(th2 + th3 + th4 - th5) - 0.5*d7*math.sin(th2 + th3 + th4 + th5) - 0.5*d6*math.sin(th2 + th3 + th4 + th5) + 0.5*d7*math.sin(th2 + th3 + th4 - th5) + 1.0*d5*math.sin(th2 + th3 + th4)
		r_z5 = -(math.sin(th2 + th3 + th4 - th5) + math.sin(th2 + th3 + th4 + th5))*(d6/2 + d7/2)
		r_z6 = 0
		 
		r_wx1 = 0
		r_wx2 = math.sin(th1)
		r_wx3 = math.sin(th1)
		r_wx4 = math.sin(th1)
		r_wx5 = 0.5*math.sin(th2 - th1 + th3 + th4) + 0.5*math.sin(th1 + th2 + th3 + th4)
		r_wx6 = math.cos(th5)*math.sin(th1) - math.cos(th1)*math.cos(th2)*math.cos(th3)*math.cos(th4)*math.sin(th5) + math.cos(th1)*math.cos(th2)*math.sin(th3)*math.sin(th4)*math.sin(th5) + math.cos(th1)*math.cos(th3)*math.sin(th2)*math.sin(th4)*math.sin(th5) + math.cos(th1)*math.cos(th4)*math.sin(th2)*math.sin(th3)*math.sin(th5)

		r_wy1 = 0
		r_wy2 = -math.cos(th1)
		r_wy3 = -math.cos(th1)
		r_wy4 = -math.cos(th1)
		r_wy5 = math.cos(th2 - th1 + th3 + th4)/2 - math.cos(th1 + th2 + th3 + th4)/2
		r_wy6 = math.cos(th2)*math.sin(th1)*math.sin(th3)*math.sin(th4)*math.sin(th5) - math.cos(th2)*math.cos(th3)*math.cos(th4)*math.sin(th1)*math.sin(th5) - math.cos(th1)*math.cos(th5) + math.cos(th3)*math.sin(th1)*math.sin(th2)*math.sin(th4)*math.sin(th5) + math.cos(th4)*math.sin(th1)*math.sin(th2)*math.sin(th3)*math.sin(th5)

		r_wz1 = 1.0
		r_wz2 = 0
		r_wz3 = 0
		r_wz4 = 0
		r_wz5 = -1.0*math.cos(th2 + th3 + th4)
		r_wz6 = -math.sin(th2 + th3 + th4)*math.sin(th5)

		J0 = np.array([[r_x1, r_x2, r_x3, r_x4, r_x5, r_x6], [r_y1, r_y2, r_y3, r_y4, r_y5, r_y6], [r_z1, r_z2, r_z3, r_z4, r_z5, r_z6], [r_wx1, r_wx2, r_wx3, r_wx4, r_wx5, r_wx6], [r_wy1, r_wy2, r_wy3, r_wy4, r_wy5, r_wy6], [r_wz1, r_wz2, r_wz3, r_wz4, r_wz5, r_wz6]])

		if inverse:
			J0 = np.linalg.inv(J0)

		return J0
	





