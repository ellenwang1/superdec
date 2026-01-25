import numpy as np
import viser
import time
import torch
from navigation import MobileNavigator
from scipy.spatial.transform import Rotation as R
# Assuming these are accessible in your env
from superdec.utils.predictions_handler import PredictionHandler 
from superdec.utils.visualizations import generate_ncolors

RESOLUTION = 10 # Lower for smoother real-time loop

def load_scene_assets(path):
    print(f"Opening {path}...")
    handler = PredictionHandler.from_npz(path)
    meshes = handler.get_meshes(resolution=RESOLUTION)
    existence_mesh = torch.ones(len(meshes), dtype = torch.bool)
    pcs = handler.get_segmented_pcs()
    return handler, meshes, pcs, existence_mesh

def main():
    server = viser.ViserServer()
    server.scene.set_up_direction([0.0, 0.0, 1.0])

    # 1. Load Room (Static)
    # Using your snippet's structure
    # another scene to try: data/output_npz/scene_example.npz
    room_handler, room_meshes, room_pcs, existence_mesh = load_scene_assets("examples/room0.npz")
    room_colors = generate_ncolors(len(room_meshes)) / 255.0

    
    for idx in range(len(room_meshes)):
        if room_meshes[idx] == None or not existence_mesh[idx]:
            continue
        
        # Borrowing your color assignment logic
        room_meshes[idx].visual.face_colors = np.ones((room_meshes[idx].visual.face_colors.shape[0], 3)) * room_colors[idx]
        server.scene.add_mesh_trimesh(f"/room/sq_{idx}", mesh=room_meshes[idx])
        # server.scene.add_point_cloud(f"/room/pc_{idx}", 
        #                              points=np.array(room_pcs[idx].points), 
        #                              colors=room_pcs[idx].colors, 
        #                              visible=False)

    # 2. Load Robot (Dynamic)
    # We need a 'Scene' object wrapper for the Navigator, or just pass the handlers
    # For this example, let's assume Scene is a simple namespace if not available
    class SimpleScene:
        def __init__(self, h): self.superquadrics = h
        
    robot_handler, robot_meshes, _, robot_exist = load_scene_assets("data/robots/franka/superquadrics/franka.npz")
    robot_scene = SimpleScene(robot_handler)
    room_scene = SimpleScene(room_handler)

    robot_handles = []
    for idx in range(len(robot_meshes)):
        if robot_meshes[idx] == None or not robot_exist[idx]:
            continue
        
        handle = server.scene.add_mesh_trimesh(
            name=f"/robot/link_{idx}",
            mesh=robot_meshes[idx],
        )
        robot_handles.append((idx, handle))
    
    nav = MobileNavigator(robot_scene, room_scene)
    # Update start location
    q_start = np.array([1.5, -1.35, -2, 0, 0, 0, 0, 0, 0, 0]) 
    goal = server.add_transform_controls("/goal", position=(5.0, -1.35, -0.5))

    # 3. Main Loop
    while True:
        nav.step(goal.position)
        
        # Check if robot has reached the goal (distance between EE and goal < 0.1m)
        ee_pos = robot_handler.translation[0, -1]
        dist_to_goal = np.linalg.norm(ee_pos - goal.position)

        # # Inside your while True loop in main_viser.py
        # # Updated debug loop to find ALL objects in ALL batches
        # for b in range(room_handler.translation.shape[0]): # Loop through batches
        #     for j in range(room_handler.translation.shape[1]): # Loop through primitives
                
        #         # Check if the object actually 'exists' according to the model
        #         if room_handler.exist[b, j] > 0.5:
        #             pos = room_handler.translation[b, j]
                    
        #             server.scene.add_frame(
        #                 f"/debug/obs_b{b}_j{j}", # Unique name per batch and primitive
        #                 position=pos,
        #                 axes_length=0.2
        #             )
        
        if dist_to_goal < 0.15:
            print("Goal reached! Resetting to start...")
            time.sleep(5.0) # Pause for a second at the goal
            nav.q = q_start.copy() # Reset the configuration
            continue # Skip the pose update for this tick to avoid a "snap" visual
        
        # Update Viser Positional Data
        for i, handle in robot_handles:
            pos = robot_handler.translation[0, i]
            rot_mat = robot_handler.rotation[0, i]
            
            quat = R.from_matrix(rot_mat).as_quat() 
            wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
            
            handle.position = pos
            handle.wxyz = wxyz
            
        time.sleep(0.01)

if __name__ == "__main__":
    main()