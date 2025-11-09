import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]
from sklearn.svm import LinearSVC  # pyright: ignore[reportMissingImports]

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from datasets.coco_dataset import class_names  # noqa: E402
from utils.roi_ops import compute_iou  # noqa: E402

IOU_POSITIVE = 0.5
IOU_NEGATIVE = 0.3


def load_metadata(ann_file: str):
    with open(ann_file, "r") as f:
        data = json.load(f)
    file_to_id = {os.path.splitext(img["file_name"])[0]: img["id"] for img in data["images"]}

    image_class_boxes: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        bbox = ann["bbox"]  # [x, y, w, h]
        box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)
        image_class_boxes[(image_id, category_id)].append(box)

    cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    cat_name_to_id = {v: k for k, v in cat_id_to_name.items()}
    return file_to_id, image_class_boxes, cat_name_to_id


def collect_samples(
    feature_path: str,
    proposal_path: str,
    gt_boxes: List[np.ndarray],
    positives_limit: int,
    negatives_limit: int,
):
    features = np.load(feature_path)
    with open(proposal_path, "rb") as fp:
        proposals = np.array(pickle.load(fp), dtype=np.float32)

    positives_idx: List[int] = []
    negatives_idx: List[int] = []

    if gt_boxes:
        gt_boxes_array = np.stack(gt_boxes)
        for idx, box in enumerate(proposals):
            overlaps = compute_iou(box, gt_boxes_array)
            max_iou = float(overlaps.max())
            if max_iou >= IOU_POSITIVE:
                positives_idx.append(idx)
            elif max_iou < IOU_NEGATIVE:
                negatives_idx.append(idx)
    else:
        negatives_idx = list(range(len(proposals)))

    if positives_limit and len(positives_idx) > positives_limit:
        positives_idx = positives_idx[:positives_limit]
    if negatives_limit and len(negatives_idx) > negatives_limit:
        negatives_idx = negatives_idx[:negatives_limit]

    positive_features = features[positives_idx] if positives_idx else np.empty((0, features.shape[1]))
    negative_features = features[negatives_idx] if negatives_idx else np.empty((0, features.shape[1]))
    return positive_features, negative_features


def main(args):
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    file_to_id, image_class_boxes, cat_name_to_id = load_metadata(args.ann_file)

    feature_files = sorted(
        f for f in os.listdir(args.cache_dir) if f.endswith("_features.npy")
    )

    svm_models = {}
    for class_name in class_names[1:]:
        class_id = cat_name_to_id.get(class_name)
        if class_id is None:
            print(f"跳过类别 {class_name}，未在 annotations 中找到")
            continue

        X = []
        y = []
        for feat_file in feature_files:
            base = feat_file.replace("_features.npy", "")
            image_id = file_to_id.get(base)
            if image_id is None:
                continue
            feature_path = os.path.join(args.cache_dir, feat_file)
            proposal_path = os.path.join(args.cache_dir, f"{base}_proposals.pkl")
            gt_boxes = image_class_boxes.get((image_id, class_id), [])

            pos_feats, neg_feats = collect_samples(
                feature_path,
                proposal_path,
                gt_boxes,
                positives_limit=args.positives_per_image,
                negatives_limit=args.negatives_per_image,
            )

            if pos_feats.size:
                X.append(pos_feats)
                y.append(np.ones(pos_feats.shape[0], dtype=np.int32))
            if neg_feats.size:
                X.append(neg_feats)
                y.append(np.zeros(neg_feats.shape[0], dtype=np.int32))

        if not X:
            print(f"类别 {class_name} 未收集到样本，跳过")
            continue

        X_mat = np.concatenate(X, axis=0)
        y_vec = np.concatenate(y, axis=0)
        print(f"训练类别 {class_name}：正样本 {np.sum(y_vec==1)}, 负样本 {np.sum(y_vec==0)}")
        clf = LinearSVC(
            C=args.C,
            class_weight="balanced",
            max_iter=args.max_iter,
            dual=False,
        )
        clf.fit(X_mat, y_vec)
        svm_models[class_name] = clf

    with open(args.out_path, "wb") as fp:
        pickle.dump(svm_models, fp)
    print(f"SVM 模型已保存至 {args.out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, help="特征与 proposals 缓存目录")
    parser.add_argument("--ann-file", required=True, help="COCO 子集标注 json")
    parser.add_argument("--out-path", default="results/rcnn/svm_models.pkl")
    parser.add_argument("--positives-per-image", type=int, default=64)
    parser.add_argument("--negatives-per-image", type=int, default=192)
    parser.add_argument("--C", type=float, default=0.001)
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()
    main(args)

