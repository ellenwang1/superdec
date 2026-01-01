import os
import argparse
import numpy as np
import open3d as o3d


def similarity_transform(A, B):
    """
    Computes similarity transform (s, R, t)
    such that B ≈ s * R @ A + t
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB / A.shape[0]
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    var_A = np.sum(np.linalg.norm(AA, axis=1) ** 2) / A.shape[0]
    s = np.sum(S) / var_A
    t = centroid_B - s * R @ centroid_A

    return s, R, t


def load_ply_points(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pts


def main(args):
    canon_dir = args.input_canon_path
    world_dir = args.input_world_path
    transform_dir = args.output_dir
    
    os.makedirs(transform_dir, exist_ok=True)

    if not os.path.isdir(canon_dir):
        raise FileNotFoundError(f"Canonical folder not found: {canon_dir}")
    if not os.path.isdir(world_dir):
        raise FileNotFoundError(f"World folder not found: {world_dir}")

    canon_files = sorted(f for f in os.listdir(canon_dir) if f.endswith(".ply"))

    if len(canon_files) == 0:
        raise RuntimeError(f"No .ply files found in {canon_dir}")

    for fname in canon_files:
        canon_path = os.path.join(canon_dir, fname)
        world_path = os.path.join(world_dir, fname)

        if not os.path.exists(world_path):
            print(f"[WARN] Missing world file for {fname}, skipping")
            continue

        try:
            pc_canon = load_ply_points(canon_path)
            pc_world = load_ply_points(world_path)

            s, R, t = similarity_transform(pc_canon, pc_world)

            out_path = os.path.join(
                transform_dir, f"{os.path.splitext(fname)[0]}.npz"
            )
            np.savez(
                out_path,
                scale=s,
                rotation=R,
                translation=t,
            )

            # Optional validation
            pc_aligned = (s * (R @ pc_canon.T)).T + t
            mse = np.mean(np.linalg.norm(pc_aligned - pc_world, axis=1) ** 2)

            print(f"[OK] {fname} | MSE={mse:.6f} | saved → {out_path}")

        except Exception as e:
            print(f"[ERROR] {fname}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute canonical-to-world transforms for matching PLY files"
    )
    parser.add_argument(
        "--input_canon_path",
        type=str,
        required=True,
        help="Folder containing canonical .ply files",
    )
    parser.add_argument(
        "--input_world_path",
        type=str,
        required=True,
        help="Folder containing world .ply files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default=None,
        help="Folder to save transforms (NPZ files)",
    )

    args = parser.parse_args()

    main(args)
