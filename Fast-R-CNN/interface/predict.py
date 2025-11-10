import argparse
import json
import os
from typing import Dict, List

import numpy as np  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
from torchvision.ops import nms  # pyright: ignore[reportMissingImports]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.environ.setdefault("TORCH_HOME", os.path.join(PROJECT_ROOT, ".cache", "torch"))

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from datasets.coco_dataset import COCOSubsetFastDataset, collate_fn, class_names  # noqa: E402
from models.fast_rcnn import FastRCNN, bbox_transform  # noqa: E402


def load_model(weights_path: str, device: torch.device) -> FastRCNN:
    model = FastRCNN(num_classes=len(class_names))
    checkpoint = torch.load(weights_path, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def clip_boxes(boxes: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    height, width = size.tolist()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=width - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height - 1)
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Fast R-CNN 推理脚本")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--proposal-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--max-proposals", type=int, default=1000)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--nms-thresh", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-json", default=os.path.join(PROJECT_ROOT, "results", "fast_rcnn", "detections.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    dataset = COCOSubsetFastDataset(
        image_dir=args.image_dir,
        ann_file=args.ann_file,
        proposal_dir=args.proposal_dir,
        max_proposals=args.max_proposals,
    )

    model = load_model(args.weights, device)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    detections: List[Dict] = []
    for idx in range(len(dataset)):
        if args.limit and idx >= args.limit:
            break
        image_tensor, proposals, target = dataset[idx]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        proposals = proposals.to(device)
        size = target["size"]

        rois = torch.cat([torch.zeros((proposals.size(0), 1), device=device), proposals], dim=1)
        with torch.no_grad():
            scores, bbox_deltas = model(image_tensor, rois)
            scores = scores.softmax(dim=1)

        proposals_boxes = proposals
        bbox_deltas = bbox_deltas.view(-1, len(class_names), 4)

        for cls_idx in range(1, len(class_names)):
            cls_scores = scores[:, cls_idx]
            keep = cls_scores >= args.score_thresh
            if keep.sum() == 0:
                continue

            cls_bbox_deltas = bbox_deltas[:, cls_idx][keep]
            cls_scores = cls_scores[keep]
            cls_boxes = bbox_transform(proposals_boxes[keep], cls_bbox_deltas)
            cls_boxes = clip_boxes(cls_boxes, size.to(device))

            keep_indices = nms(cls_boxes, cls_scores, args.nms_thresh)
            cls_boxes = cls_boxes[keep_indices]
            cls_scores = cls_scores[keep_indices]
            coco_cat_id = dataset.label_to_cat_id.get(cls_idx, None)
            if coco_cat_id is None:
                continue
            for box, score in zip(cls_boxes, cls_scores):
                x1, y1, x2, y2 = box.tolist()
                detections.append(
                    {
                        "image_id": int(target["image_id"].item()),
                        "category_id": int(coco_cat_id),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score.item()),
                    }
                )

    with open(args.out_json, "w") as fp:
        json.dump(detections, fp)
    print(f"[INFO] Saved {len(detections)} detections to {args.out_json}")


if __name__ == "__main__":
    main()

