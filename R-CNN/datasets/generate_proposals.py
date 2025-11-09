import argparse
import json
import os
import pickle
import cv2  # pyright: ignore[reportMissingImports]
from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]

def selective_search(image_path):
    img = cv2.imread(image_path)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    return ss.process()

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.ann_file, "r") as f:
        coco = json.load(f)
    for img in tqdm(coco["images"]):
        img_path = os.path.join(args.image_dir, img["file_name"])
        boxes = selective_search(img_path)
        out_path = os.path.join(args.out_dir, f"{img['file_name'].split('.')[0]}.pkl")
        with open(out_path, "wb") as fp:
            pickle.dump(boxes, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--out-dir", default="data/proposals/coco_subset")
    args = parser.parse_args()
    main(args)