import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
from pycocotools import mask as mask_utils  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
from torchvision.models.detection import maskrcnn_resnet50_fpn  # pyright: ignore[reportMissingImports]
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # pyright: ignore[reportMissingImports]
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor  # pyright: ignore[reportMissingImports]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "torch")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("TORCH_HOME", CACHE_DIR)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from datasets.coco_subset import COCOSubsetInstanceDataset, collate_fn  # noqa: E402


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int, weights_path: str):
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    state_dict = torch.load(weights_path, map_location="cpu")
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    model.load_state_dict(state_dict)
    return model


def masks_to_rles(masks: torch.Tensor, threshold: float = 0.5) -> List[Dict]:
    rles = []
    masks_np = masks.squeeze(1).cpu().numpy()
    for mask in masks_np:
        binary = mask >= threshold
        if binary.sum() == 0:
            rles.append(None)
            continue
        rle = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")  # type: ignore[assignment]
        rles.append(rle)
    return rles


def main():
    parser = argparse.ArgumentParser(description="Mask R-CNN 推理脚本")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--weights", required=True, help="训练好的模型权重路径")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="仅推理前 N 张图")
    parser.add_argument("--out-json", default=os.path.join(PROJECT_ROOT, "results", "mask_rcnn", "detections.json"))
    parser.add_argument("--visualize-dir", default=None, help="可选，保存叠加 mask 的可视化图像目录")
    args = parser.parse_args()

    device = get_device()
    dataset = COCOSubsetInstanceDataset(
        image_dir=args.image_dir,
        ann_file=args.ann_file,
        categories=args.categories,
    )

    model = build_model(num_classes=len(dataset.category_names) + 1, weights_path=args.weights)
    model.to(device)
    model.eval()

    detections: List[Dict] = []

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    if args.visualize_dir:
        os.makedirs(args.visualize_dir, exist_ok=True)

    total_images = len(dataset) if not args.limit else min(args.limit, len(dataset))
    print(f"[INFO] Running inference on {total_images} images...")

    with torch.no_grad():
        for idx in range(len(dataset)):
            if args.limit and idx >= args.limit:
                break
            image, _, filename = dataset[idx]
            image = image.to(device)
            outputs = model([image])[0]

            boxes = outputs["boxes"].cpu()
            labels = outputs["labels"].cpu()
            scores = outputs["scores"].cpu()
            masks = outputs["masks"].cpu()

            keep = scores >= args.score_thresh
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            masks = masks[keep]

            rles = masks_to_rles(masks, threshold=args.mask_thresh) if masks.numel() else []

            for box, label, score, rle in zip(boxes, labels, scores, rles):
                coco_cat_id = dataset.label_to_cat_id.get(label.item(), None)
                if coco_cat_id is None or rle is None:
                    continue
                x1, y1, x2, y2 = box.tolist()
                detections.append(
                    {
                        "image_id": int(dataset.ids[idx]),
                        "category_id": coco_cat_id,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score.item()),
                        "segmentation": rle,
                    }
                )

        if (idx + 1) % 10 == 0 or (args.limit and idx + 1 == args.limit) or idx + 1 == len(dataset):
            print(f"[INFO] Processed {idx + 1}/{total_images} images", flush=True)

    with open(args.out_json, "w") as fp:
        json.dump(detections, fp)
    print(f"[INFO] Saved {len(detections)} detections to {args.out_json}")


if __name__ == "__main__":
    main()

