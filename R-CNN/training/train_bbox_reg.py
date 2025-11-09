import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]
from sklearn.linear_model import Ridge  # pyright: ignore[reportMissingImports]

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from datasets.coco_dataset import class_names  # noqa: E402
from utils.roi_ops import compute_iou  # noqa: E402

IOU_THRESHOLD = 0.5


def load_metadata(ann_file: str):
    with open(ann_file, "r") as f:
        data = json.load(f)
    file_to_id = {os.path.splitext(img["file_name"])[0]: img["id"] for img in data["images"]}
    image_class_boxes: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    for ann in data["annotations"]:
        bbox = ann["bbox"]
        box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)
        image_class_boxes[(ann["image_id"], ann["category_id"])].append(box)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    cat_name_to_id = {v: k for k, v in cat_id_to_name.items()}
    return file_to_id, image_class_boxes, cat_name_to_id


def bbox_transform(proposals: np.ndarray, gt_boxes: np.ndarray):
    px1, py1, px2, py2 = proposals[:, 0], proposals[:, 1], proposals[:, 2], proposals[:, 3]
    gx1, gy1, gx2, gy2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]

    pw = px2 - px1 + 1.0
    ph = py2 - py1 + 1.0
    px = px1 + 0.5 * pw
    py = py1 + 0.5 * ph

    gw = gx2 - gx1 + 1.0
    gh = gy2 - gy1 + 1.0
    gx = gx1 + 0.5 * gw
    gy = gy1 + 0.5 * gh

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = np.log(gw / pw)
    dh = np.log(gh / ph)

    return np.stack([dx, dy, dw, dh], axis=1)


def main(args):
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    file_to_id, image_class_boxes, cat_name_to_id = load_metadata(args.ann_file)
    feature_files = sorted(f for f in os.listdir(args.cache_dir) if f.endswith("_features.npy"))

    bbox_models = {}
    for class_name in class_names[1:]:
        class_id = cat_name_to_id.get(class_name)
        if class_id is None:
            print(f"跳过类别 {class_name}，未在 annotations 中找到")
            continue

        feats_list = []
        targets_list = []

        for feat_file in feature_files:
            base = feat_file.replace("_features.npy", "")
            image_id = file_to_id.get(base)
            if image_id is None:
                continue

            gt_boxes = image_class_boxes.get((image_id, class_id), [])
            if not gt_boxes:
                continue

            gt_boxes_array = np.stack(gt_boxes)
            feature_path = os.path.join(args.cache_dir, feat_file)
            proposal_path = os.path.join(args.cache_dir, f"{base}_proposals.pkl")

            features = np.load(feature_path)
            with open(proposal_path, "rb") as fp:
                proposals = np.array(pickle.load(fp), dtype=np.float32)

            overlaps = np.array([compute_iou(box, gt_boxes_array) for box in proposals])
            if overlaps.size == 0:
                continue
            max_iou = overlaps.max(axis=1)
            assignment = overlaps.argmax(axis=1)
            positive_mask = max_iou >= IOU_THRESHOLD
            if not positive_mask.any():
                continue

            pos_indices = np.where(positive_mask)[0]
            matched_gt = gt_boxes_array[assignment[pos_indices]]
            pos_proposals = proposals[pos_indices]
            pos_features = features[pos_indices]
            targets = bbox_transform(pos_proposals, matched_gt)

            feats_list.append(pos_features)
            targets_list.append(targets)

        if not feats_list:
            print(f"类别 {class_name} 无回归样本，跳过")
            continue

        X = np.concatenate(feats_list, axis=0)
        Y = np.concatenate(targets_list, axis=0)
        print(f"训练类别 {class_name} bbox 回归器：样本 {X.shape[0]}")
        reg = Ridge(alpha=args.alpha, fit_intercept=True)
        reg.fit(X, Y)
        bbox_models[class_name] = reg

    with open(args.out_path, "wb") as fp:
        pickle.dump(bbox_models, fp)
    print(f"回归器已保存至 {args.out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, help="特征与 proposals 缓存目录")
    parser.add_argument("--ann-file", required=True, help="COCO 子集标注 json")
    parser.add_argument("--out-path", default="results/rcnn/bbox_reg.pkl")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    main(args)

