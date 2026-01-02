"""
URDF → per-link point cloud (.ply) exporter (visual mesh only, with world transforms)

Requirements:
  pip install urdf-parser-py trimesh numpy scipy

Usage:
  python urdf_to_link_ply.py path/to/robot.urdf output_dir \
      --package_root path/to/ros_ws/src \
      --points_per_link 5000
"""

import os
import argparse
import numpy as np
import trimesh
from urdf_parser_py.urdf import URDF, Mesh
from scipy.spatial.transform import Rotation as R


def resolve_mesh_path(filename, urdf_dir, package_root):
    """Resolve a mesh filename to an absolute path."""
    if filename.startswith("package://"):
        if package_root is None:
            raise ValueError("package:// path but --package_root not provided")
        rel = filename.replace("package://", "")
        return os.path.join(package_root, rel)

    if os.path.isabs(filename):
        return filename

    return os.path.join(urdf_dir, filename)


def mesh_to_pointcloud(mesh_path, n_points):
    """Load a mesh and sample points from its surface."""
    mesh = trimesh.load(mesh_path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"Mesh is empty: {mesh_path}")
    mesh.process(validate=True)
    return mesh.sample(n_points)


def apply_origin(points, xyz, rpy):
    """Apply a URDF visual origin transform to a point cloud."""
    rot = R.from_euler("xyz", rpy).as_matrix()
    return (rot @ points.T).T + np.array(xyz)


def joint_to_matrix(joint, joint_angle=0.0):
    """Compute 4x4 homogeneous transform for a joint."""
    xyz = np.array(joint.origin.xyz) if joint.origin else np.zeros(3)
    rpy = np.array(joint.origin.rpy) if joint.origin else np.zeros(3)
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz

    if joint.type == "revolute":
        axis = np.array(joint.axis)
        R_joint = R.from_rotvec(joint_angle * axis).as_matrix()
        T[:3, :3] = T[:3, :3] @ R_joint
    elif joint.type == "prismatic":
        axis_local = np.array(joint.axis)
        T[:3, 3] += T[:3, :3] @ (axis_local * joint_angle)
    return T


def compute_link_transform(robot, link_name, joint_angles=None):
    """Recursively compute world transform of a link (default joint angles = 0)."""
    if joint_angles is None:
        joint_angles = {}

    # Base link
    if link_name == robot.links[0].name:
        return np.eye(4)

    # Find parent joint
    parent_joint = next(
        (j for j in robot.joints if j.child == link_name), None
    )
    if parent_joint is None:
        return np.eye(4)

    parent_T = compute_link_transform(robot, parent_joint.parent, joint_angles)
    angle = joint_angles.get(parent_joint.name, 0.0)
    joint_T = joint_to_matrix(parent_joint, angle)
    return parent_T @ joint_T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("urdf", help="Path to URDF file")
    parser.add_argument("output_dir", help="Directory to save .ply files")
    parser.add_argument("--package_root", default=None, help="Root dir for package:// meshes")
    parser.add_argument("--points_per_link", type=int, default=5000, help="Points per link")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    robot = URDF.from_xml_file(args.urdf)
    urdf_dir = os.path.dirname(os.path.abspath(args.urdf))

    print(f"Loaded URDF: {robot.name}")
    print(f"Found {len(robot.links)} links")

    for link in robot.links:
        if not link.visual:
            continue

        visuals = link.visual if isinstance(link.visual, list) else [link.visual]
        all_points = []

        # Compute world transform of this link (all joint angles = 0)
        T_link = compute_link_transform(robot, link.name)

        for visual in visuals:
            # Only process Mesh geometry
            if not isinstance(visual.geometry, Mesh):
                continue

            # Pick visual mesh (.dae)
            if not visual.geometry.filename.lower().endswith(".dae"):
                continue

            mesh_path = resolve_mesh_path(visual.geometry.filename, urdf_dir, args.package_root)
            if not os.path.exists(mesh_path):
                print(f"Warning: mesh file not found: {mesh_path}")
                continue

            points = mesh_to_pointcloud(mesh_path, args.points_per_link)

            # Apply visual origin first
            if visual.origin is not None:
                points = apply_origin(points, visual.origin.xyz, visual.origin.rpy)

            # Transform points into world coordinates
            points = (T_link[:3, :3] @ points.T).T + T_link[:3, 3]

            all_points.append(points)

        if not all_points:
            continue

        points = np.vstack(all_points).astype(np.float32)
        out_path = os.path.join(args.output_dir, f"{link.name}.ply")

        # Save as PLY
        cloud = trimesh.PointCloud(points)
        cloud.export(out_path)
        print(f"Saved {link.name}: {points.shape[0]} points → {out_path}")


if __name__ == "__main__":
    main()
