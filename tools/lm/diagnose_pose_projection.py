import argparse
import json
import os
import os.path as osp
import pickle

import cv2
import numpy as np


ID2OBJ = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose GT/Pred 6D pose projection on one image.")
    parser.add_argument("--dataset-root", default="datasets/BOP_DATASETS/lm", help="LM dataset root")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="dataset split")
    parser.add_argument("--scene-id", type=int, required=True, help="scene id, e.g. 1")
    parser.add_argument("--im-id", type=int, required=True, help="image id, e.g. 0")
    parser.add_argument("--obj-id", type=int, default=None, help="object id to diagnose")
    parser.add_argument("--preds-path", required=True, help="path to *_preds.pkl")
    parser.add_argument("--output", default=None, help="output image path")
    parser.add_argument("--axis-scale", type=float, default=0.5, help="axis length scale relative to box extent")
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_preds(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_bbox3d_from_models_info(model_info):
    minx = float(model_info["min_x"]) / 1000.0
    miny = float(model_info["min_y"]) / 1000.0
    minz = float(model_info["min_z"]) / 1000.0
    maxx = minx + float(model_info["size_x"]) / 1000.0
    maxy = miny + float(model_info["size_y"]) / 1000.0
    maxz = minz + float(model_info["size_z"]) / 1000.0
    avgx = (minx + maxx) * 0.5
    avgy = (miny + maxy) * 0.5
    avgz = (minz + maxz) * 0.5
    return np.array(
        [
            [maxx, maxy, maxz],
            [minx, maxy, maxz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, maxy, minz],
            [minx, maxy, minz],
            [minx, miny, minz],
            [maxx, miny, minz],
            [avgx, avgy, avgz],
        ],
        dtype=np.float32,
    )


def get_axis3d_from_bbox3d(bbox3d, scale=0.5):
    center = bbox3d[8]
    front = (bbox3d[2] + bbox3d[3] + bbox3d[6] + bbox3d[7]) / 4.0
    right = (bbox3d[0] + bbox3d[3] + bbox3d[4] + bbox3d[7]) / 4.0
    up = (bbox3d[0] + bbox3d[1] + bbox3d[2] + bbox3d[3]) / 4.0
    kpts = np.array([front, right, up, center], dtype=np.float32)
    kpts = (kpts - center[None]) * scale + center[None]
    return kpts


def project_pts(pts, K, R, t):
    pts_cam = (R @ pts.T + t.reshape(3, 1)).T
    pts_2d = (K @ pts_cam.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    return pts_2d


def draw_projected_box3d(image, qs, color=(255, 0, 255), middle_color=None, bottom_color=None, thickness=2):
    qs = qs.astype(np.int32)
    bottom_default = [
        (255, 0, 0),
        (255, 255, 0),
        (0, 255, 255),
        (0, 255, 0),
    ]
    for k in range(4):
        i, j = k + 4, (k + 1) % 4 + 4
        cur_bottom = bottom_default[k] if bottom_color is None else bottom_color
        cv2.line(image, tuple(qs[i]), tuple(qs[j]), cur_bottom, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cur_mid = bottom_default[k] if middle_color is None else middle_color
        cv2.line(image, tuple(qs[i]), tuple(qs[j]), cur_mid, thickness, cv2.LINE_AA)

        i, j = k, (k + 1) % 4
        cv2.line(image, tuple(qs[i]), tuple(qs[j]), color, thickness, cv2.LINE_AA)
    return image


def draw_axes(image, axis_2d, colors, prefix):
    axis_2d = axis_2d.astype(np.int32)
    origin = tuple(axis_2d[3])
    labels = ["X", "Y", "Z"]
    for idx in range(3):
        end_pt = tuple(axis_2d[idx])
        cv2.line(image, origin, end_pt, colors[idx], 2, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{prefix}{labels[idx]}",
            end_pt,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colors[idx],
            1,
            cv2.LINE_AA,
        )
    cv2.circle(image, origin, 3, (0, 0, 255), -1)
    return image


def find_prediction(preds, file_name, obj_id=None):
    matches = []
    for obj_name, obj_preds in preds.items():
        if file_name in obj_preds:
            matches.append((obj_name, obj_preds[file_name]))
    if not matches:
        return None, None
    if obj_id is not None and obj_id in ID2OBJ:
        wanted = ID2OBJ[obj_id]
        for obj_name, pred in matches:
            if obj_name == wanted:
                return obj_name, pred
    return matches[0]


def main():
    args = parse_args()

    scene_id = args.scene_id
    im_id = args.im_id
    scene_root = osp.join(args.dataset_root, args.split, f"{scene_id:06d}")
    rgb_path = osp.join(scene_root, "rgb", f"{im_id:06d}.png")
    if not osp.exists(rgb_path):
        rgb_path = osp.join(scene_root, "rgb", f"{im_id:06d}.jpg")
    if not osp.exists(rgb_path):
        raise FileNotFoundError(rgb_path)

    scene_gt = load_json(osp.join(scene_root, "scene_gt.json"))
    scene_camera = load_json(osp.join(scene_root, "scene_camera.json"))
    models_info = load_json(osp.join(args.dataset_root, "models_eval", "models_info.json"))
    preds = load_preds(args.preds_path)

    gt_annos = scene_gt[str(im_id)]
    gt_anno = None
    if args.obj_id is None:
        gt_anno = gt_annos[0]
        obj_id = gt_anno["obj_id"]
    else:
        obj_id = args.obj_id
        for anno in gt_annos:
            if anno["obj_id"] == obj_id:
                gt_anno = anno
                break
        if gt_anno is None:
            raise ValueError(f"obj_id {obj_id} not found in GT for scene {scene_id} image {im_id}")

    obj_name = ID2OBJ.get(obj_id, str(obj_id))
    file_name = osp.join(args.dataset_root, args.split, f"{scene_id:06d}", "rgb", osp.basename(rgb_path))
    pred_obj_name, pred = find_prediction(preds, file_name, obj_id=obj_id)
    if pred is None:
        raise KeyError(f"No prediction found for {file_name}")

    cam = scene_camera[str(im_id)]
    K = np.array(cam["cam_K"], dtype=np.float32).reshape(3, 3)

    R_gt = np.array(gt_anno["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
    t_gt = np.array(gt_anno["cam_t_m2c"], dtype=np.float32).reshape(3) / 1000.0

    R_pred = np.array(pred["R"], dtype=np.float32).reshape(3, 3)
    t_pred = np.array(pred["t"], dtype=np.float32).reshape(3)

    bbox3d = get_bbox3d_from_models_info(models_info[str(obj_id)])
    bbox3d_corners = bbox3d[:8]
    axis3d = get_axis3d_from_bbox3d(bbox3d, scale=args.axis_scale)

    gt_box_2d = project_pts(bbox3d_corners, K, R_gt, t_gt)
    pred_box_2d = project_pts(bbox3d_corners, K, R_pred, t_pred)
    gt_axis_2d = project_pts(axis3d, K, R_gt, t_gt)
    pred_axis_2d = project_pts(axis3d, K, R_pred, t_pred)

    image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read {rgb_path}")

    vis = image.copy()
    vis = draw_projected_box3d(vis, gt_box_2d, color=(0, 255, 255), middle_color=(0, 220, 220), bottom_color=(0, 180, 180), thickness=2)
    vis = draw_projected_box3d(vis, pred_box_2d, color=(255, 0, 255), middle_color=(220, 0, 220), bottom_color=(180, 0, 180), thickness=2)
    vis = draw_axes(vis, gt_axis_2d, [(0, 128, 255), (0, 200, 200), (200, 128, 0)], "GT-")
    vis = draw_axes(vis, pred_axis_2d, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], "P-")

    cv2.putText(vis, f"GT: {obj_name}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Pred: {pred_obj_name}|{float(pred.get('score', 0.0)):.3f}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    if args.output is None:
        out_dir = osp.join("output", "diagnostics")
        os.makedirs(out_dir, exist_ok=True)
        args.output = osp.join(out_dir, f"{scene_id:06d}_{im_id:06d}_{obj_name}_diag.jpg")
    else:
        os.makedirs(osp.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, vis)

    np.set_printoptions(precision=3, suppress=True)
    print(f"Image: {rgb_path}")
    print(f"Prediction source: {args.preds_path}")
    print(f"Output image: {args.output}")
    print(f"Object: GT={obj_name}, Pred={pred_obj_name}")
    print("K:\n", K)
    print("GT t (m):", t_gt)
    print("Pred t (m):", t_pred)
    print("GT box 2D points:\n", gt_box_2d)
    print("Pred box 2D points:\n", pred_box_2d)
    print("GT axis 2D points [X, Y, Z, O]:\n", gt_axis_2d)
    print("Pred axis 2D points [X, Y, Z, O]:\n", pred_axis_2d)


if __name__ == "__main__":
    main()
