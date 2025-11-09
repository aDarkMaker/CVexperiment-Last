import argparse
import json
import os
import pickle
import sys
from typing import Dict, List

import numpy as np  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
from tqdm import tqdm  # pyright: ignore[reportMissingImports, reportMissingModuleSource]

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from datasets.coco_dataset import COCOSubsetDataset, class_names  # noqa: E402
from models.feature_extractor import AlexNetExtractor  # noqa: E402
from utils.roi_ops import crop_and_resize, compute_iou  # noqa: E402


def preprocess_crops(dataset, image, proposals):
    crops = []
    for box in proposals:
        crop = crop_and_resize(image, box)
        crops.append(dataset.transform(crop))
    if not crops:
        return None
    return torch.stack(crops)


def apply_bbox_deltas(boxes: np.ndarray, deltas: np.ndarray, image_size):
    if deltas.size == 0:
        return boxes
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    pred_ctr_x = ctr_x + deltas[:, 0] * widths
    pred_ctr_y = ctr_y + deltas[:, 1] * heights
    pred_w = widths * np.exp(deltas[:, 2])
    pred_h = heights * np.exp(deltas[:, 3])

    pred_boxes = np.zeros_like(boxes)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

    width, height = image_size
    pred_boxes[:, 0::2] = np.clip(pred_boxes[:, 0::2], 0, width - 1)
    pred_boxes[:, 1::2] = np.clip(pred_boxes[:, 1::2], 0, height - 1)
    return pred_boxes


def nms(boxes: np.ndarray, scores: np.ndarray, thresh: float):
    if boxes.size == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = compute_iou(boxes[i], boxes[rest])
        rest = rest[ious <= thresh]
        order = rest
    return keep


def load_metadata(ann_file):
    with open(ann_file, "r") as f:
        data = json.load(f)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    return cat_id_to_name


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = COCOSubsetDataset(
        image_dir=args.image_dir,
        ann_file=args.ann_file,
        proposal_dir=args.proposal_dir,
        max_proposals=args.max_proposals,
    )

    feature_model = AlexNetExtractor(pretrained=False)
    feature_model.load_state_dict(torch.load(args.feature_ckpt, map_location=device))
    feature_model = feature_model.to(device)
    feature_model.eval()

    with open(args.svm_model, "rb") as fp:
        svm_models: Dict[str, object] = pickle.load(fp)
    if args.bbox_reg:
        with open(args.bbox_reg, "rb") as fp:
            bbox_models: Dict[str, object] = pickle.load(fp)
    else:
        bbox_models = {}

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    cat_id_to_name = load_metadata(args.ann_file)
    name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}

    detections: List[dict] = []

    with torch.no_grad():
        for item in tqdm(dataset, desc="Inference"):
            image = item["image"]
            proposals = item["proposals"]
            image_size = image.size
            if proposals.size == 0:
                continue

            crop_tensor = preprocess_crops(dataset, image, proposals)
            if crop_tensor is None:
                continue
            crop_tensor = crop_tensor.to(device)
            features = feature_model(crop_tensor).cpu().numpy()

            for class_name in class_names[1:]:
                if class_name not in svm_models:
                    continue

                clf = svm_models[class_name]
                scores = clf.decision_function(features)
                if scores.ndim > 1:
                    scores = scores[:, 0]
                keep_mask = scores > args.score_thresh
                if not np.any(keep_mask):
                    continue

                selected_scores = scores[keep_mask]
                selected_boxes = proposals[keep_mask]
                if class_name in bbox_models:
                    deltas = bbox_models[class_name].predict(features[keep_mask])
                    selected_boxes = apply_bbox_deltas(selected_boxes, deltas, image_size)

                if selected_boxes.shape[0] == 0:
                    continue

                keep_indices = nms(selected_boxes, selected_scores, args.nms_thresh)
                cat_id = name_to_cat_id[class_name]
                for idx in keep_indices:
                    box = selected_boxes[idx]
                    score = float(selected_scores[idx])
                    x1, y1, x2, y2 = box
                    detections.append(
                        {
                            "image_id": item["image_id"],
                            "category_id": cat_id,
                            "bbox": [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)],
                            "score": score,
                            "category_name": class_name,
                        }
                    )

    with open(args.out_json, "w") as f:
        json.dump(detections, f)
    print(f"检测结果已保存至 {args.out_json}，共 {len(detections)} 条")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--proposal-dir", required=True)
    parser.add_argument("--feature-ckpt", required=True)
    parser.add_argument("--svm-model", required=True)
    parser.add_argument("--bbox-reg", default=None)
    parser.add_argument("--out-json", default="results/rcnn/detections.json")
    parser.add_argument("--max-proposals", type=int, default=800)
    parser.add_argument("--score-thresh", type=float, default=0.0)
    parser.add_argument("--nms-thresh", type=float, default=0.3)
    args = parser.parse_args()
    main(args)

