import argparse
import json
import os
import os.path as osp
import pickle

import imageio
import numpy as np


def imread(path):
    try:
        import cv2

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    try:
        return imageio.imread(path)
    except Exception:
        from PIL import Image

        return np.array(Image.open(path))


def parse_args():
    parser = argparse.ArgumentParser("Generate LM xyz_crop pkl files from depth + pose")
    parser.add_argument("--dataset-root", default="datasets/BOP_DATASETS/lm", help="LM dataset root")
    parser.add_argument("--split", default="train", choices=["train", "test", "both"], help="dataset split")
    parser.add_argument(
        "--mask-type",
        default="visib",
        choices=["visib", "full"],
        help="mask source: visib is safer for real depth; full assumes no occlusion artifacts",
    )
    parser.add_argument("--scene-id", type=int, default=None, help="optional single scene id")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing pkl files")
    return parser.parse_args()


def iter_scenes(split_root, scene_id=None):
    if scene_id is not None:
        yield f"{scene_id:06d}"
        return
    for name in sorted(os.listdir(split_root)):
        if name.isdigit() and osp.isdir(osp.join(split_root, name)):
            yield name


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_mask(mask_path):
    mask = imread(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask > 0


def load_depth_m(depth_path, depth_scale):
    depth = imread(depth_path).astype(np.float32)
    return depth * float(depth_scale) / 1000.0


def backproject_to_obj(depth_m, mask, K, R, t):
    xyz = np.zeros(depth_m.shape + (3,), dtype=np.float32)
    v, u = np.where(mask)
    if len(v) == 0:
        return xyz

    z = depth_m[v, u]
    valid = z > 0
    if not np.any(valid):
        return xyz

    v = v[valid]
    u = u[valid]
    z = z[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    p_cam = np.stack([x_cam, y_cam, z], axis=-1)
    p_obj = (p_cam - t.reshape(1, 3)) @ R
    xyz[v, u, :] = p_obj
    return xyz


def mask_to_xyxy(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def main():
    args = parse_args()
    splits = ["train", "test"] if args.split == "both" else [args.split]

    for split in splits:
        split_root = osp.join(args.dataset_root, split)
        xyz_root = osp.join(split_root, "xyz_crop")
        os.makedirs(xyz_root, exist_ok=True)
        total = 0
        saved = 0

        for scene_name in iter_scenes(split_root, args.scene_id):
            scene_root = osp.join(split_root, scene_name)
            if not osp.isdir(scene_root):
                continue

            gt_dict = load_json(osp.join(scene_root, "scene_gt.json"))
            cam_dict = load_json(osp.join(scene_root, "scene_camera.json"))
            out_scene_root = osp.join(xyz_root, scene_name)
            os.makedirs(out_scene_root, exist_ok=True)

            for im_key, annos in gt_dict.items():
                int_im_id = int(im_key)
                cam_info = cam_dict.get(str(int_im_id), cam_dict.get(im_key))
                if cam_info is None:
                    continue
                K = np.array(cam_info["cam_K"], dtype=np.float32).reshape(3, 3)
                depth_m = load_depth_m(
                    osp.join(scene_root, "depth", f"{int_im_id:06d}.png"),
                    cam_info.get("depth_scale", 1.0),
                )

                for anno_i, anno in enumerate(annos):
                    total += 1
                    save_path = osp.join(out_scene_root, f"{int_im_id:06d}_{anno_i:06d}.pkl")
                    if osp.exists(save_path) and not args.overwrite:
                        continue

                    mask_dir = "mask_visib" if args.mask_type == "visib" else "mask"
                    mask_path = osp.join(scene_root, mask_dir, f"{int_im_id:06d}_{anno_i:06d}.png")
                    if not osp.exists(mask_path):
                        continue

                    mask = load_mask(mask_path)
                    xyxy = mask_to_xyxy(mask)
                    if xyxy is None:
                        continue

                    R = np.array(anno["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
                    t = np.array(anno["cam_t_m2c"], dtype=np.float32) / 1000.0
                    xyz = backproject_to_obj(depth_m, mask, K, R, t)
                    x1, y1, x2, y2 = xyxy
                    xyz_crop = xyz[y1 : y2 + 1, x1 : x2 + 1, :]
                    with open(save_path, "wb") as f:
                        pickle.dump({"xyz_crop": xyz_crop.astype(np.float16), "xyxy": [x1, y1, x2, y2]}, f, protocol=4)
                    saved += 1

        print(f"[{split}] generated {saved} xyz_crop files (visited {total} instances)")


if __name__ == "__main__":
    main()
