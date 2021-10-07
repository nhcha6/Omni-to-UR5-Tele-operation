import numpy as np
from ikfastpy import PyKinematics
import sys

# Initialize kinematics for UR5 robot arm
ur5_kin = PyKinematics()
n_joints = ur5_kin.getDOF()

joint_angles = [0, 0, 0, 0, 0, 0] # in radians

# Test forward kinematics: get end effector pose from joint angles
print("\nTesting forward kinematics:\n")
print("Joint angles:")
print(joint_angles)
ee_pose = ur5_kin.forward(joint_angles)
ee_pose = np.asarray(ee_pose).reshape(3,4) # 3x4 rigid transformation matrix
print("\nEnd effector pose:")
print(ee_pose)
print("\n-----------------------------")

# ee_pose = np.array([[ 1.00000000e+00,  2.06823107e-13,  4.23773785e-23, -8.17250000e-01],[ 8.48403205e-23, -2.05310499e-10, -1.00000000e+00, -1.91450000e-01], [-2.06823107e-13,  1.00000000e+00, -2.05310463e-10, -5.49100004e-03]])
# ee_pose = np.array([[ 1.00000000e+00,  0,  0, -8.17250000e-01],[0, 0, -1.00000000e+00, -1.91450000e-01], [0,  1.00000000e+00, 0, -5.49100004e-03]])

new_pose = ee_pose.reshape(-1).tolist()
# print(ee_pose)
# new_pose = []
# for element in ee_pose:
# 	if abs(element)<0.000001:
# 		new_pose.append(0)
# 	else:
# 		new_pose.append(element)
# print(new_pose)

# Test inverse kinematics: get joint angles from end effector pose
print("\nTesting inverse kinematics:\n")
print(new_pose)
joint_configs = ur5_kin.inverse(new_pose)
n_solutions = int(len(joint_configs)/n_joints)
print("%d solutions found:"%(n_solutions))
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/path/to/application/app/folder')
joint_configs = np.asarray(joint_configs).reshape(n_solutions,n_joints)
for joint_config in joint_configs:
    print(joint_config)

# Check cycle-consistency of forward and inverse kinematics
assert(np.any([np.sum(np.abs(joint_config-np.asarray(joint_angles))) < 1e-4 for joint_config in joint_configs]))
print("\nTest passed!")