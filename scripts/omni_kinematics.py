# coding: utf-8 
import math
import numpy as np
from pyquaternion import Quaternion
from time import time

# class for storing the solution to the jacobian for the omni
class OmniKinematics():
	def __init__(self):
		# intialise joint states and names
		self.omni_js = {'waist': 0, 'shoulder': 0, 'elbow': 0, 'wrist1': 0, 'wrist2': 0, 'wrist3': 0}
		self.omni_joint_names = ['waist', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3']

	# given the joint position, this function returns the jacobian or inverse jacobian
	# of the omni
	def jacobian(self, js, inverse=False):
		# store joint angles in list
		th_n = [js[name] for name in self.omni_joint_names]
		[th1, th2, th3, th4, th5, th6] = th_n

		# define link lengths
		d4 = 149.352e-3
		a2 = 127.508e-3

		# solution from MATLAB analysis
		rx1 = -math.sin(th1)*(d4*math.sin(th2 + th3) + a2*math.cos(th2))
		rx2 = math.cos(th1)*(d4*math.cos(th2 + th3) - a2*math.sin(th2))
		rx3 = d4*math.cos(th2 + th3)*math.cos(th1)
		rx4 = 0
		rx5 = 0
		rx6 = 0

		ry1 = math.cos(th1)*(d4*math.sin(th2 + th3) + a2*math.cos(th2))
		ry2 = math.sin(th1)*(d4*math.cos(th2 + th3) - a2*math.sin(th2))
		ry3 = d4*math.cos(th2 + th3)*math.sin(th1)
		ry4 = 0
		ry5 = 0
		ry6 = 0

		rz1 = 0            
		rz2 = d4*math.sin(th2 + th3) + a2*math.cos(th2)         
		rz3 = d4*math.sin(th2 + th3)
		rz4 = 0
		rz5 = 0
		rz6 = 0

		rwx1 = 0
		rwx2 = math.sin(th1)
		rwx3 = math.sin(th1)
		rwx4 = math.sin(th2 + th3)*math.cos(th1)
		rwx5 = math.cos(th4)*math.sin(th1) - math.cos(th1)*math.cos(th2)*math.cos(th3)*math.sin(th4) + math.cos(th1)*math.sin(th2)*math.sin(th3)*math.sin(th4)
		rwx6 = math.sin(th1)*math.sin(th4)*math.sin(th5) + math.cos(th1)*math.cos(th2)*math.cos(th5)*math.sin(th3) + math.cos(th1)*math.cos(th3)*math.cos(th5)*math.sin(th2) + math.cos(th1)*math.cos(th2)*math.cos(th3)*math.cos(th4)*math.sin(th5) - 1.0*math.cos(th1)*math.cos(th4)*math.sin(th2)*math.sin(th3)*math.sin(th5)

		rwy1 = 0
		rwy2 = -math.cos(th1)
		rwy3 = -math.cos(th1)
		rwy4 = math.sin(th2 + th3)*math.sin(th1)
		rwy5 = math.sin(th1)*math.sin(th2)*math.sin(th3)*math.sin(th4) - math.cos(th2)*math.cos(th3)*math.sin(th1)*math.sin(th4) - math.cos(th1)*math.cos(th4)
		rwy6 = math.cos(th2)*math.cos(th5)*math.sin(th1)*math.sin(th3) - math.cos(th1)*math.sin(th4)*math.sin(th5) + math.cos(th3)*math.cos(th5)*math.sin(th1)*math.sin(th2) + math.cos(th2)*math.cos(th3)*math.cos(th4)*math.sin(th1)*math.sin(th5) - math.cos(th4)*math.sin(th1)*math.sin(th2)*math.sin(th3)*math.sin(th5)

		rwz1 = 1.0         
		rwz2 = 0         
		rwz3 = 0     
		rwz4 = -1.0*math.cos(th2 + th3)                                                                 
		rwz5 = -1.0*math.sin(th2 + th3)*math.sin(th4)
		rwz6 = math.sin(th2 + th3)*math.cos(th4)*math.sin(th5) - math.cos(th2 + th3)*math.cos(th5)

		J0 = np.array([[rx1, rx2, rx3, rx4, rx5, rx6], [ry1, ry2, ry3, ry4, ry5, ry6], [rz1, rz2, rz3, rz4, rz5, rz6], [rwx1, rwx2, rwx3, rwx4, rwx5, rwx6], [rwy1, rwy2, rwy3, rwy4, rwy5, rwy6], [rwz1, rwz2, rwz3, rwz4, rwz5, rwz6]])

		if inverse:
			J0 = np.linalg.inv(J0)

		# return Jacobian
		return J0

    




