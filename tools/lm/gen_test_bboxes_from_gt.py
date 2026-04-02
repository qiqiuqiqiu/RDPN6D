import json
import os
import os.path as osp
from collections import OrderedDict


def main():
    root = osp.join("datasets", "BOP_DATASETS", "lm")
    test_root = osp.join(root, "test")
    out_dir = osp.join(test_root, "test_bboxes")
    out_path = osp.join(out_dir, "bbox_faster_all.json")

    detections = OrderedDict()

    for scene_name in sorted(os.listdir(test_root)):
        scene_root = osp.join(test_root, scene_name)
        if not osp.isdir(scene_root):
            continue
        gt_info_path = osp.join(scene_root, "scene_gt_info.json")
        gt_path = osp.join(scene_root, "scene_gt.json")
        if not (osp.exists(gt_info_path) and osp.exists(gt_path)):
            continue

        with open(gt_info_path, "r") as f:
            gt_info = json.load(f)
        with open(gt_path, "r") as f:
            gt = json.load(f)

        scene_id = int(scene_name)
        for im_id_str in sorted(gt_info.keys(), key=lambda x: int(x)):
            im_id = int(im_id_str)
            scene_im_id = f"{scene_id:06d}/{im_id:06d}"
            dets = []
            annos = gt.get(im_id_str, [])
            infos = gt_info[im_id_str]
            for anno_i, info in enumerate(infos):
                if anno_i >= len(annos):
                    continue
                bbox_visib = info["bbox_visib"]
                if bbox_visib[2] <= 0 or bbox_visib[3] <= 0:
                    continue
                dets.append(
                    {
                        "obj_id": annos[anno_i]["obj_id"],
                        "bbox_est": bbox_visib,
                        "score": 1.0,
                    }
                )
            detections[scene_im_id] = dets

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(detections, f)

    print(f"saved {len(detections)} entries to {out_path}")


if __name__ == "__main__":
    main()
