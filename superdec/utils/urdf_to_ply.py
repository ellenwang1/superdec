import numpy as np
import time
import viser
from urdf_parser_py.urdf import URDF
from superdec.utils.predictions_handler import PredictionHandler
from superdec.utils.visualizations import generate_ncolors

RESOLUTION = 30


# -------------------------
# Math utilities
# -------------------------

def rot_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)],
    ])


def quat_from_rot(R):
    qw = np.sqrt(max(0.0, 1 + np.trace(R))) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    return (float(qx), float(qy), float(qz), float(qw))


# -------------------------
# Load URDF kinematics
# -------------------------

def load_urdf_kinematics(urdf_path):
    robot = URDF.from_xml_file(urdf_path)

    joints = []
    for joint in robot.joints:
        if joint.type == "fixed":
            continue

        origin_xyz = joint.origin.xyz if joint.origin else [0, 0, 0]
        axis = joint.axis if joint.axis else [0, 0, 1]

        joints.append({
            "joint_name": joint.name,
            "parent": joint.parent,
            "child": joint.child,
            "axis": axis,
            "origin": origin_xyz,
        })

    return joints


# -------------------------
# Build link → SQ mapping
# -------------------------

def build_link_to_sq_map(predictions_sq):
    link_to_sqs = {}
    for idx, name in enumerate(predictions_sq.names):
        if not predictions_sq.exist[idx]:
            continue
        link_to_sqs.setdefault(name, []).append(idx)
    return link_to_sqs


# -------------------------
# MAIN
# -------------------------

def main():
    server = viser.ViserServer()

    # ---- Paths ----
    urdf_path = "data/robots/franka/franka.urdf"
    npz_path = "data/robots/franka/superquadrics/franka.npz"

    # ---- Load data ----
    predictions_sq = PredictionHandler.from_npz(npz_path)
    meshes = predictions_sq.get_meshes(resolution=RESOLUTION)
    names = predictions_sq.names

    urdf_joints = load_urdf_kinematics(urdf_path)
    link_to_sqs = build_link_to_sq_map(predictions_sq)

    # ---- Add SQs to Viser ----
    colors = generate_ncolors(len(meshes)) / 255.0
    sq_handles = {}

    for idx, mesh in enumerate(meshes):
        if mesh is None or not predictions_sq.exist[idx]:
            continue

        mesh.visual.face_colors = (
            np.ones((mesh.visual.face_colors.shape[0], 3)) * colors[idx]
        )
        mesh.visual.vertex_colors = (
            np.ones((mesh.visual.vertex_colors.shape[0], 3)) * colors[idx]
        )

        handle = server.scene.add_mesh_trimesh(
            name=f"sq_{idx}",
            mesh=mesh,
            visible=True,
        )
        handle.position = (0.0, 0.0, 0.0)
        handle.orientation = (0.0, 0.0, 0.0, 1.0)

        sq_handles[idx] = handle

    # ---- Build runtime kinematic links ----
    runtime_links = []
    for joint in urdf_joints:
        runtime_links.append({
            "name": joint["child"],
            "parent": joint["parent"],
            "axis": joint["axis"],
            "origin": joint["origin"],
            "sqs": link_to_sqs.get(joint["child"], []),
        })

    # ---- FK storage ----
    link_world_rot = {}
    link_world_pos = {}

    # ---- Animate ----
    joint_angle = 0.0

    while True:
        joint_angle += 0.01

        for link in runtime_links:
            name = link["name"]
            parent = link["parent"]

            R_joint = rot_from_axis_angle(link["axis"], joint_angle)
            t_joint = np.asarray(link["origin"])

            if parent not in link_world_rot:
                R_world = R_joint
                t_world = t_joint
            else:
                R_parent = link_world_rot[parent]
                t_parent = link_world_pos[parent]

                R_world = R_parent @ R_joint
                t_world = t_parent + R_parent @ t_joint

            link_world_rot[name] = R_world
            link_world_pos[name] = t_world

            # Apply transform to ALL SQs of this link
            for sq_idx in link["sqs"]:
                sq_handles[sq_idx].position = (
                    float(t_world[0]),
                    float(t_world[1]),
                    float(t_world[2]),
                )
                sq_handles[sq_idx].orientation = quat_from_rot(R_world)

        time.sleep(0.03)


if __name__ == "__main__":
    main()
