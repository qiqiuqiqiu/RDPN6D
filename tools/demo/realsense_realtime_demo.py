import argparse
import copy
import os
import os.path as osp
import sys
import time

import cv2
import mmcv
import numpy as np
import pyrealsense2 as rs
import torch
from mmcv import Config

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../.."))
sys.path.insert(0, PROJ_ROOT)

import ref
from core.gdrn_modeling.models import GDRN
from core.utils.data_utils import crop_resize_by_warp_affine, get_2d_coord_np, my_warp_affine
from core.utils.my_checkpoint import MyCheckpointer
from lib.pysixd import misc


def parse_args():
    parser = argparse.ArgumentParser("RealSense D435i real-time 6D pose demo")
    parser.add_argument("--config-file", required=True, help="config path")
    parser.add_argument("--weights", required=True, help="model weights")
    parser.add_argument("--obj-name", default="ape", help="object name to track")
    parser.add_argument("--ref-key", default="lm_full", help="ref module key, e.g. lm_full")
    parser.add_argument("--score", type=float, default=1.0, help="display score")
    parser.add_argument("--input-width", type=int, default=640, help="camera stream width")
    parser.add_argument("--input-height", type=int, default=480, help="camera stream height")
    parser.add_argument("--fps", type=int, default=30, help="camera stream fps")
    parser.add_argument("--bbox", nargs=4, type=int, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--tracker", default="csrt", choices=["csrt", "kcf", "none"], help="ROI tracker type")
    parser.add_argument("--save-dir", default="output/realtime_demo", help="optional output frame directory")
    parser.add_argument("--cpu", action="store_true", help="force cpu inference")
    parser.add_argument("--opts", nargs="+", action=mmcv.DictAction, help="extra config overrides")
    return parser.parse_args()


def get_tracker(name):
    if name == "none":
        return None
    creator_name = f"Tracker{name.upper()}_create"
    if hasattr(cv2, creator_name):
        return getattr(cv2, creator_name)()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, creator_name):
        return getattr(cv2.legacy, creator_name)()
    raise RuntimeError(f"OpenCV tracker {name} is not available in this build")


def normalize_image(cfg, image_chw):
    pixel_mean = np.array(cfg.MODEL.PIXEL_MEAN, dtype=np.float32).reshape(-1, 1, 1)
    pixel_std = np.array(cfg.MODEL.PIXEL_STD, dtype=np.float32).reshape(-1, 1, 1)
    return (image_chw - pixel_mean) / pixel_std


def compute_extent_from_models_info(model_info):
    return np.array(
        [
            float(model_info["size_x"]) / 1000.0,
            float(model_info["size_y"]) / 1000.0,
            float(model_info["size_z"]) / 1000.0,
        ],
        dtype=np.float32,
    )


def get_bbox3d_from_model_info(model_info):
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


def get_axis_points_from_bbox3d(bbox3d, scale=0.5):
    center = bbox3d[8]
    front = (bbox3d[2] + bbox3d[3] + bbox3d[6] + bbox3d[7]) / 4.0
    right = (bbox3d[0] + bbox3d[3] + bbox3d[4] + bbox3d[7]) / 4.0
    up = (bbox3d[0] + bbox3d[1] + bbox3d[2] + bbox3d[3]) / 4.0
    kpts = np.array([right, front, up, center], dtype=np.float32)
    kpts = (kpts - center[None]) * scale + center[None]
    return kpts


def build_model(cfg, weights):
    model, _ = GDRN.build_model_optimizer(cfg)
    MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(weights, resume=False)
    model.eval()
    return model


def prepare_inputs(cfg, color_bgr, depth_m, K, bbox_xyxy, roi_extent, fps_points, roi_cls):
    im_H, im_W = color_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=np.float32)
    scale = max(bw, bh) * cfg.INPUT.DZI_PAD_SCALE
    scale = min(scale, max(im_H, im_W)) * 1.0

    input_res = cfg.MODEL.CDPN.BACKBONE.INPUT_RES
    out_res = cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES

    coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)

    roi_img = crop_resize_by_warp_affine(
        color_bgr, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
    ).transpose(2, 0, 1).astype(np.float32)
    roi_img = normalize_image(cfg, roi_img)

    depth_crop = crop_resize_by_warp_affine(
        depth_m[:, :, None], bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
    )
    if depth_crop.ndim == 2:
        depth_crop = depth_crop[:, :, None]

    H_aff = my_warp_affine(coord_2d, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR)
    offset_matrix = np.zeros((3, 3), dtype=np.float32)
    offset_matrix[:2, :] = H_aff
    offset_matrix[2, 2] = 1.0
    resize_ratio = out_res / scale
    newK = np.matmul(offset_matrix, K.astype(np.float32))

    rows, cols = input_res, input_res
    ymap = np.array([[j for _ in range(cols)] for j in range(rows)], dtype=np.float32)
    xmap = np.array([[i for i in range(cols)] for _ in range(rows)], dtype=np.float32)
    pt2 = depth_crop.astype(np.float32) / resize_ratio
    pt0 = (xmap[:, :, None] - newK[0, 2]) * pt2 / newK[0, 0]
    pt1 = (ymap[:, :, None] - newK[1, 2]) * pt2 / newK[1, 1]
    depth_xyz = np.concatenate((pt0, pt1, pt2), axis=2).transpose(2, 0, 1).astype(np.float32)

    roi_img = np.concatenate((roi_img, depth_xyz), axis=0)

    roi_coord_2d = crop_resize_by_warp_affine(
        coord_2d, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
    ).transpose(2, 0, 1).astype(np.float32)
    roi_coord_2d = np.concatenate((depth_xyz[:, ::4, ::4], roi_coord_2d), axis=0)

    batch = {
        "roi_img": torch.from_numpy(roi_img[None]).float(),
        "roi_coord_2d": torch.from_numpy(roi_coord_2d[None]).float(),
        "roi_cls": torch.tensor([roi_cls], dtype=torch.long),
        "roi_cam": torch.from_numpy(K[None].astype(np.float32)),
        "roi_center": torch.from_numpy(bbox_center[None].astype(np.float32)),
        "roi_wh": torch.from_numpy(np.array([[bw, bh]], dtype=np.float32)),
        "resize_ratio": torch.tensor([resize_ratio], dtype=torch.float32),
        "roi_extent": torch.from_numpy(roi_extent[None].astype(np.float32)),
        "fps": torch.from_numpy(fps_points[None].astype(np.float32)),
        "bbox_est": torch.from_numpy(np.array([[x1, y1, x2, y2]], dtype=np.float32)),
        "bbox": torch.from_numpy(np.array([[x1, y1, x2, y2]], dtype=np.float32)),
    }
    return batch


def draw_pose(image, K, R, t, bbox3d, label_text):
    vis = image.copy()
    box2d = misc.project_pts(bbox3d[:8], K, R, t)
    vis = misc.draw_projected_box3d(vis, box2d, middle_color=None, bottom_color=(128, 128, 128))

    mins = bbox3d[:8].min(axis=0)
    maxs = bbox3d[:8].max(axis=0)
    axis_len = 0.35 * float(np.max(maxs - mins))
    axis_pts_3d = np.array(
        [[0.0, 0.0, 0.0], [axis_len, 0.0, 0.0], [0.0, axis_len, 0.0], [0.0, 0.0, axis_len]],
        dtype=np.float32,
    )
    axis_pts_2d = misc.project_pts(axis_pts_3d, K, R, t).astype(np.int32)
    origin = tuple(axis_pts_2d[0])
    cv2.line(vis, origin, tuple(axis_pts_2d[1]), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(vis, origin, tuple(axis_pts_2d[2]), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(vis, origin, tuple(axis_pts_2d[3]), (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, "X", tuple(axis_pts_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, "Y", tuple(axis_pts_2d[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(vis, "Z", tuple(axis_pts_2d[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    text_x = int(np.clip(np.min(box2d[:, 0]), 0, max(vis.shape[1] - 1, 0)))
    text_y = int(np.clip(np.min(box2d[:, 1]) - 8, 12, max(vis.shape[0] - 1, 12)))
    cv2.putText(vis, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config_file)
    if args.opts:
        cfg.merge_from_dict(args.opts)
    cfg = copy.deepcopy(cfg)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.TEST.VIS = False
    cfg.MODEL.DEVICE = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    data_ref = ref.__dict__[args.ref_key]
    if args.obj_name not in data_ref.obj2id:
        raise KeyError(f"{args.obj_name} not found in {args.ref_key}")
    obj_id = data_ref.obj2id[args.obj_name]
    roi_cls = data_ref.objects.index(args.obj_name)
    models_info = data_ref.get_models_info()
    model_info = models_info[str(obj_id)]
    roi_extent = compute_extent_from_models_info(model_info)
    bbox3d = get_bbox3d_from_model_info(model_info)
    fps_points = data_ref.get_fps_points()[str(obj_id)][
        f"fps{cfg.MODEL.CDPN.ROT_HEAD.NUM_REGIONS}_and_center"
    ][:-1].astype(np.float32)

    model = build_model(cfg, args.weights)
    device = torch.device(cfg.MODEL.DEVICE)

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, args.input_width, args.input_height, rs.format.bgr8, args.fps)
    rs_config.enable_stream(rs.stream.depth, args.input_width, args.input_height, rs.format.z16, args.fps)
    profile = pipeline.start(rs_config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    tracker = None
    bbox_xyxy = args.bbox
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    frame_idx = 0
    last_time = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]], dtype=np.float32)

            if tracker is not None:
                ok, box = tracker.update(color)
                if ok:
                    x, y, w, h = box
                    bbox_xyxy = [int(x), int(y), int(x + w), int(y + h)]

            display = color.copy()
            if bbox_xyxy is not None:
                batch = prepare_inputs(cfg, color, depth, K, bbox_xyxy, roi_extent, fps_points, roi_cls)
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                with torch.no_grad():
                    out_dict = model(
                        batch["roi_img"],
                        roi_classes=batch["roi_cls"],
                        roi_coord_2d=batch["roi_coord_2d"],
                        roi_cams=batch["roi_cam"],
                        roi_centers=batch["roi_center"],
                        roi_whs=batch["roi_wh"],
                        roi_extents=batch["roi_extent"],
                        resize_ratios=batch["resize_ratio"],
                        do_loss=False,
                        fps=batch["fps"],
                    )
                R_pred = out_dict["rot"][0].detach().cpu().numpy()
                t_pred = out_dict["trans"][0].detach().cpu().numpy()
                display = draw_pose(display, K, R_pred, t_pred, bbox3d, f"{args.obj_name}|{args.score:.3f}")

            now = time.time()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            cv2.putText(display, f"FPS: {fps:.2f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "s: select ROI  r: reset ROI  q: quit", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if bbox_xyxy is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
                cv2.rectangle(display, (x1, y1), (x2, y2), (80, 80, 80), 1)

            cv2.imshow("RDPN6D RealSense Demo", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                roi = cv2.selectROI("RDPN6D RealSense Demo", color, fromCenter=False, showCrosshair=True)
                if roi[2] > 0 and roi[3] > 0:
                    x, y, w, h = roi
                    bbox_xyxy = [int(x), int(y), int(x + w), int(y + h)]
                    tracker = get_tracker(args.tracker) if args.tracker != "none" else None
                    if tracker is not None:
                        tracker.init(color, tuple(roi))
            if key == ord("r"):
                bbox_xyxy = None
                tracker = None

            if bbox_xyxy is not None:
                save_path = osp.join(save_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(save_path, display)
            frame_idx += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
