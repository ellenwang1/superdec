import numpy as np

def update_robot_kinematics(robot_scene, q):
    # q mapping: [0: X_base, 1: Z_base, 2: Yaw_base, 3+: joints]
    x_b, z_b, theta_b = q[0], q[1], q[2]
    handler = robot_scene.superquadrics
    
    # Strictly enforce a Y-height of -1.35 (just above floor)
    target_y = -1.35
    
    # Rotation around Y-axis only
    c, s = np.cos(theta_b), np.sin(theta_b)
    R_base = np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ])
    T_base = np.array([x_b, target_y, z_b])

    if not hasattr(handler, 'orig_trans'):
        handler.orig_trans = handler.translation.copy()
        handler.orig_rot = handler.rotation.copy()

    for p in range(handler.translation.shape[1]):
        handler.rotation[0, p] = R_base @ handler.orig_rot[0, p]
        handler.translation[0, p] = (R_base @ handler.orig_trans[0, p]) + T_base

def compute_mobile_jacobian(q, p_world, handler, link_idx):
    num_joints = len(q)
    J = np.zeros((3, num_joints))
    
    # Base X translation affects World X
    J[0, 0] = 1.0 
    # Base Z translation affects World Z
    J[2, 1] = 1.0 
    # Base Yaw (around Y) affects World X and Z
    r_base = p_world - np.array([q[0], 0.05, q[1]])
    J[0, 2] = -r_base[2]
    J[2, 2] = r_base[0]
    
    # Note: J[1, :] remains all ZEROS. This mathematically proves 
    # to the solver that no joint movement can change the Y position.
    
    for i in range(link_idx + 1):
        joint_idx = i + 3
        if joint_idx >= num_joints: break
        axis = np.array([0, 1, 0]) # Robot joints rotate around Y
        r_link = p_world - handler.translation[0, i]
        J[:, joint_idx] = np.cross(axis, r_link)
    return J