import argparse
import os
import os.path as osp
import shutil


def parse_args():
    parser = argparse.ArgumentParser("Prepare LM image_set helper files")
    parser.add_argument("--dataset-root", default="datasets/BOP_DATASETS/lm", help="LM dataset root")
    parser.add_argument("--all-source", default="train", choices=["train", "test"], help="source file for *_all.txt")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing *_all.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    image_set_root = osp.join(args.dataset_root, "image_set")
    obj_names = []
    for name in sorted(os.listdir(image_set_root)):
        if name.endswith("_train.txt"):
            obj_names.append(name[: -len("_train.txt")])

    for obj_name in obj_names:
        src = osp.join(image_set_root, f"{obj_name}_{args.all_source}.txt")
        dst = osp.join(image_set_root, f"{obj_name}_all.txt")
        if not osp.exists(src):
            continue
        if osp.exists(dst) and not args.overwrite:
            continue
        shutil.copyfile(src, dst)
        print(f"wrote {dst} from {src}")


if __name__ == "__main__":
    main()
