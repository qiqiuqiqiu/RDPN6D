import argparse
import json
import os
import os.path as osp
import struct
from collections import OrderedDict


DEFAULT_SCENE_TO_OBJ = OrderedDict([(1, 2), (2, 1), (3, 3)])
DEFAULT_SCENE_TO_NAME = OrderedDict([(1, "benchvise"), (2, "ape"), (3, "bowl")])


def parse_args():
    parser = argparse.ArgumentParser("Fix custom LM scene_gt obj_id by scene id and validate dataset consistency")
    parser.add_argument("--dataset-root", default="datasets/BOP_DATASETS/lm", help="LM dataset root")
    parser.add_argument("--apply", action="store_true", help="write scene_gt.json changes in-place")
    parser.add_argument("--backup-suffix", default=".bak_objid_fix", help="backup suffix for original scene_gt.json")
    parser.add_argument("--tolerance-mm", type=float, default=1.0, help="models_info vs ply bbox tolerance in mm")
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def parse_ply_bbox(path):
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Invalid PLY header: {path}")
            header_lines.append(line.decode("ascii").strip())
            if header_lines[-1] == "end_header":
                break

        fmt = None
        vertex_count = None
        vertex_props = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                vertex_props.append(parts[-1])

        if fmt not in {"ascii", "binary_little_endian"}:
            raise RuntimeError(f"Unsupported PLY format {fmt} for {path}")
        if vertex_count is None:
            raise RuntimeError(f"No vertex element in {path}")

        xyz_idx = [vertex_props.index("x"), vertex_props.index("y"), vertex_props.index("z")]
        mins = [float("inf"), float("inf"), float("inf")]
        maxs = [float("-inf"), float("-inf"), float("-inf")]

        if fmt == "ascii":
            for _ in range(vertex_count):
                vals = f.readline().decode("ascii").strip().split()
                coords = [float(vals[i]) for i in xyz_idx]
                for i in range(3):
                    mins[i] = min(mins[i], coords[i])
                    maxs[i] = max(maxs[i], coords[i])
        else:
            fmt_map = {
                "char": "b",
                "uchar": "B",
                "short": "h",
                "ushort": "H",
                "int": "i",
                "uint": "I",
                "float": "f",
                "float32": "f",
                "double": "d",
                "float64": "d",
            }
            prop_types = []
            in_vertex = False
            for line in header_lines:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "element":
                    in_vertex = parts[1] == "vertex"
                elif parts[0] == "property" and in_vertex:
                    prop_types.append(parts[1])
            struct_fmt = "<" + "".join(fmt_map[t] for t in prop_types)
            row_size = struct.calcsize(struct_fmt)
            for _ in range(vertex_count):
                row = f.read(row_size)
                vals = struct.unpack(struct_fmt, row)
                coords = [float(vals[i]) for i in xyz_idx]
                for i in range(3):
                    mins[i] = min(mins[i], coords[i])
                    maxs[i] = max(maxs[i], coords[i])

    return mins, maxs


def maybe_to_mm(vals):
    if max(abs(v) for v in vals) < 10.0:
        return [v * 1000.0 for v in vals]
    return vals


def bbox_from_models_info(info):
    mins = [float(info["min_x"]), float(info["min_y"]), float(info["min_z"])]
    maxs = [
        float(info["min_x"]) + float(info["size_x"]),
        float(info["min_y"]) + float(info["size_y"]),
        float(info["min_z"]) + float(info["size_z"]),
    ]
    return mins, maxs


def check_models_and_ply(dataset_root, tolerance_mm):
    print("\n[Check] models_info / ply")
    models_info_path = osp.join(dataset_root, "models", "models_info.json")
    models_eval_info_path = osp.join(dataset_root, "models_eval", "models_info.json")
    models_info = load_json(models_info_path)
    models_eval_info = load_json(models_eval_info_path)

    for scene_id, obj_id in DEFAULT_SCENE_TO_OBJ.items():
        obj_name = DEFAULT_SCENE_TO_NAME[scene_id]
        ply_path = osp.join(dataset_root, "models", f"obj_{obj_id:06d}.ply")
        ply_eval_path = osp.join(dataset_root, "models_eval", f"obj_{obj_id:06d}.ply")
        print(f"scene {scene_id:06d} -> obj {obj_id} ({obj_name})")
        print(f"  models ply exists: {osp.exists(ply_path)}")
        print(f"  models_eval ply exists: {osp.exists(ply_eval_path)}")
        print(f"  models_info key exists: {str(obj_id) in models_info}")
        print(f"  models_eval_info key exists: {str(obj_id) in models_eval_info}")

        if not (osp.exists(ply_path) and str(obj_id) in models_info):
            continue

        ply_mins, ply_maxs = parse_ply_bbox(ply_path)
        ply_mins = maybe_to_mm(ply_mins)
        ply_maxs = maybe_to_mm(ply_maxs)
        info_mins, info_maxs = bbox_from_models_info(models_info[str(obj_id)])
        diffs = [max(abs(ply_mins[i] - info_mins[i]), abs(ply_maxs[i] - info_maxs[i])) for i in range(3)]
        ok = all(d <= tolerance_mm for d in diffs)
        print(f"  ply bbox min(mm): {ply_mins}")
        print(f"  ply bbox max(mm): {ply_maxs}")
        print(f"  info bbox min(mm): {info_mins}")
        print(f"  info bbox max(mm): {info_maxs}")
        print(f"  bbox match within {tolerance_mm:.3f} mm: {ok}")


def check_image_sets(dataset_root):
    print("\n[Check] image_set")
    image_set_root = osp.join(dataset_root, "image_set")
    for scene_id, obj_name in DEFAULT_SCENE_TO_NAME.items():
        for split in ("train", "test"):
            scene_gt_path = osp.join(dataset_root, split, f"{scene_id:06d}", "scene_gt.json")
            scene_gt = load_json(scene_gt_path)
            expected_ids = sorted(int(k) for k in scene_gt.keys())
            txt_path = osp.join(image_set_root, f"{obj_name}_{split}.txt")
            if not osp.exists(txt_path):
                print(f"  MISSING {txt_path}")
                continue
            with open(txt_path, "r") as f:
                listed_ids = sorted(int(line.strip()) for line in f if line.strip())
            ok = listed_ids == expected_ids
            print(f"  {obj_name}_{split}.txt count={len(listed_ids)} expected={len(expected_ids)} match={ok}")
        all_path = osp.join(image_set_root, f"{obj_name}_all.txt")
        print(f"  {obj_name}_all.txt exists={osp.exists(all_path)}")


def fix_scene_gt(dataset_root, apply, backup_suffix):
    print("\n[Fix] scene_gt obj_id")
    for split in ("train", "test"):
        for scene_id, expected_obj_id in DEFAULT_SCENE_TO_OBJ.items():
            scene_gt_path = osp.join(dataset_root, split, f"{scene_id:06d}", "scene_gt.json")
            if not osp.exists(scene_gt_path):
                print(f"  MISSING {scene_gt_path}")
                continue
            scene_gt = load_json(scene_gt_path)
            before = sorted({anno["obj_id"] for annos in scene_gt.values() for anno in annos})
            changed = 0
            total = 0
            for annos in scene_gt.values():
                for anno in annos:
                    total += 1
                    if anno["obj_id"] != expected_obj_id:
                        anno["obj_id"] = expected_obj_id
                        changed += 1
            after = sorted({anno["obj_id"] for annos in scene_gt.values() for anno in annos})
            print(
                f"  {split}/{scene_id:06d}: before={before} after={after} "
                f"changed={changed}/{total} apply={apply}"
            )
            if apply:
                backup_path = scene_gt_path + backup_suffix
                if not osp.exists(backup_path):
                    with open(scene_gt_path, "rb") as src, open(backup_path, "wb") as dst:
                        dst.write(src.read())
                save_json(scene_gt_path, scene_gt)


def main():
    args = parse_args()
    fix_scene_gt(args.dataset_root, args.apply, args.backup_suffix)
    check_models_and_ply(args.dataset_root, args.tolerance_mm)
    check_image_sets(args.dataset_root)


if __name__ == "__main__":
    main()
