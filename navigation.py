import numpy as np
from scipy.linalg import solve
from kinematics import update_robot_kinematics, compute_mobile_jacobian

def get_stable_distance(p_loc, a, b, c, e1, e2):
    """
    Calculates the Euclidean radial distance for a Superquadric.
    This ensures the gradient is a unit vector, preventing 'explosions.'
    """
    # 1. Algebraic inside-outside function
    # F = ((|x/a|^(2/e2) + |y/b|^(2/e2))^(e2/e1) + |z/c|^(2/e1))
    f2, f1 = 2.0/max(e2, 1e-2), 2.0/max(e1, 1e-2)
    inner = (np.abs(p_loc[0]/a)**f2 + np.abs(p_loc[1]/b)**f2)**(e2/e1)
    F = inner + np.abs(p_loc[2]/c)**f1
    
    dist_norm = np.linalg.norm(p_loc)
    if dist_norm < 1e-6:
        return -min(a, b, c), np.array([0, 1, 0])
        
    # 2. Radial Distance Approximation (Positive = Outside)
    # This formula creates a uniform field regardless of shape 'blockiness'
    dist = dist_norm * (F**(e1/2.0) - 1.0)
    
    # 3. Unit Gradient (The direction of the push)
    grad = p_loc / dist_norm
    
    return dist, grad

class MobileNavigator:
    def __init__(self, robot_scene, room_scene):
        self.robot_scene = robot_scene
        self.room_scene = room_scene
        num_links = self.robot_scene.superquadrics.translation.shape[1]
        self.q = np.zeros(3 + num_links)
        
        # --- TUNED PHYSICS PARAMETERS ---
        self.influence_dist = 0.5   # 50cm detection bubble
        self.rep_gain = 2.5         # Repulsion strength
        self.att_gain = 1.2         # Attraction strength
        self.max_step = 0.04        # 4cm hard cap per frame
        self.damp = 0.05            # Jacobian stability damping

    def step(self, goal_pos):
        # 1. Update Kinematics (Locks Y to floor inside kinematics.py)
        update_robot_kinematics(self.robot_scene, self.q)
        r_h = self.robot_scene.superquadrics
        s_h = self.room_scene.superquadrics
        
        # 2. ATTRACTIVE FORCE (Goal)
        ee_pos = r_h.translation[0, -1]
        v_att = goal_pos - ee_pos
        v_att[1] = 0 # STRICT PLANE LOCK: No vertical pulling
        
        d_goal = np.linalg.norm(v_att)
        v_att_unit = v_att / (d_goal + 1e-6)
        v_att_final = v_att_unit * min(d_goal * self.att_gain, 0.4)

        # 3. REPULSIVE FORCE (Obstacles)
        dq_rep = np.zeros_like(self.q)
        min_d = 100.0

        # Loop through every link of the robot
        for i in range(r_h.translation.shape[1]):
            p_w = r_h.translation[0, i]
            
            # NEW: Loop through every OBJECT (B) in the scene
            for b in range(s_h.translation.shape[0]): 
                
                # Loop through every PRIMITIVE (P) belonging to that object
                for j in range(s_h.translation.shape[1]):
                    
                    # Check existence if your model uses it
                    if hasattr(s_h, 'exist') and s_h.exist[b, j] < 0.5:
                        continue
                        
                    p_rel = p_w - s_h.translation[b, j]
                    p_loc = s_h.rotation[b, j].T @ p_rel
                    
                    # Use data from the specific Object 'b' and Primitive 'j'
                    d, g_loc = get_stable_distance(p_loc, 
                                                *s_h.scale[b, j], 
                                                *s_h.exponents[b, j])
                    
                    min_d = min(min_d, d)

                    if d < self.influence_dist:
                        mag = self.rep_gain * (1.0 - max(d, 0) / self.influence_dist)
                        
                        # Transform gradient back to world space using object b's rotation
                        v_push_w = s_h.rotation[b, j] @ g_loc
                        v_push_w[1] = 0 
                        
                        v_side = np.array([-v_push_w[2], 0, v_push_w[0]])
                        v_final_push = (v_push_w + 0.4 * v_side) * mag
                        
                        J = compute_mobile_jacobian(self.q, p_w, r_h, i)
                        dq_rep += J.T @ v_final_push

        # 4. JACOBIAN SOLVER
        J_ee = compute_mobile_jacobian(self.q, ee_pos, r_h, r_h.translation.shape[1]-1)
        A = J_ee.T @ J_ee + self.damp * np.eye(len(self.q))
        dq_att = solve(A, J_ee.T @ v_att_final)

        # 5. BRAKE & INTEGRATION
        # Brake affects Attraction more than Repulsion to prioritize safety
        brake = np.clip(min_d / 0.2, 0.1, 1.0)
        delta_q = (dq_att * brake) + dq_rep
        
        # 6. HARD VELOCITY CAP (The anti-explosion insurance)
        norm_q = np.linalg.norm(delta_q)
        if norm_q > self.max_step:
            delta_q = (delta_q / norm_q) * self.max_step

        self.q += delta_q
        return self.q