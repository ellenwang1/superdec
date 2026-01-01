import os
import numpy as np
import argparse

def apply_similarity_to_superquadrics(scale, rotation, translation, s, R, t):
    """
    Apply similarity transform to superquadrics.
    """
    scale_w = s * scale
    rotation_w = R @ rotation
    translation_w = (s * (R @ translation.T)).T + t
    return scale_w, rotation_w, translation_w

def apply_transform_to_object(obj_name, obj_scale, obj_rotation, obj_translation, transforms_folder):
    """
    Apply similarity transform to all superquadrics of a single object.
    Returns transformed scale, rotation, translation arrays.
    """
    transform_path = os.path.join(transforms_folder, f"{obj_name}.npz")
    if not os.path.exists(transform_path):
        print(f"[WARN] Missing transform for object {obj_name}, skipping")
        return obj_scale, obj_rotation, obj_translation  # return original

    T = np.load(transform_path)
    s = float(T["scale"])
    R = T["rotation"]
    t = T["translation"]

    # Apply similarity transform
    scale_w, rotation_w, translation_w = apply_similarity_to_superquadrics(
        obj_scale, obj_rotation, obj_translation, s, R, t
    )
    print(f"[OK] Transformed object: {obj_name}")
    return scale_w, rotation_w, translation_w

def transform_sq_npz_multiple(canon_npz, transforms_folder, output_npz):
    """
    Apply per-object transforms to all superquadrics in a canonical NPZ file.
    Saves all transformed objects in a single NPZ.
    """
    sq = np.load(canon_npz, allow_pickle=True)
    num_objects = len(sq["names"])
    
    # Prepare output arrays
    scale_w = np.zeros_like(sq["scale"])
    rotation_w = np.zeros_like(sq["rotation"])
    translation_w = np.zeros_like(sq["translation"])
    
    for i in range(num_objects):
        obj_name = sq["names"][i]
        scale_w[i], rotation_w[i], translation_w[i] = apply_transform_to_object(
            obj_name, sq["scale"][i], sq["rotation"][i], sq["translation"][i], transforms_folder
        )
    
    # Save everything in one NPZ
    np.savez_compressed(
        output_npz,
        names=sq["names"],
        pc=sq["pc"],
        assign_matrix=sq["assign_matrix"],
        scale=scale_w,
        rotation=rotation_w,
        translation=translation_w,
        exponents=sq["exponents"],
        exist=sq["exist"],
    )
    print(f"[OK] All objects transformed and saved to {output_npz}")

def main():
    parser = argparse.ArgumentParser(
        description="Apply similarity transforms to canonical superquadric NPZs"
    )
    parser.add_argument("--npz_folder", type=str, required=True,
                        help="Folder containing a single canonical NPZ file")
    parser.add_argument("--transforms_folder", type=str, required=True,
                        help="Folder with per-object transforms")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Output NPZ folder")
    args = parser.parse_args()

    if not os.path.isdir(args.npz_folder):
        raise FileNotFoundError(f"Missing canonical NPZ folder: {args.npz_folder}")
    if not os.path.isdir(args.transforms_folder):
        raise FileNotFoundError(f"Missing transforms folder: {args.transforms_folder}")

    # 🔹 Find the single NPZ file in the folder
    npz_files = [f for f in os.listdir(args.npz_folder) if f.endswith(".npz")]
    if len(npz_files) == 0:
        raise RuntimeError(f"No .npz files found in {args.npz_folder}")
    if len(npz_files) > 1:
        raise RuntimeError(
            f"Expected exactly one .npz file in {args.npz_folder}, found {npz_files}"
        )

    canon_npz = os.path.join(args.npz_folder, npz_files[0])

    # 🔹 Output path (same filename)
    os.makedirs(args.output_folder, exist_ok=True)
    output_npz = os.path.join(args.output_folder, npz_files[0])

    transform_sq_npz_multiple(canon_npz, args.transforms_folder, output_npz)


if __name__ == "__main__":
    main()
