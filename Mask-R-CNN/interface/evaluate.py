import argparse
import json
import os

from pycocotools.coco import COCO  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
from pycocotools.cocoeval import COCOeval  # pyright: ignore[reportMissingImports, reportMissingModuleSource]


def coco_evaluate(ann_file: str, det_file: str, iou_type: str = "bbox"):
    coco_gt = COCO(ann_file)
    if not os.path.exists(det_file):
        raise FileNotFoundError(f"未找到检测结果：{det_file}")
    with open(det_file, "r") as fp:
        detections = json.load(fp)
    if not detections:
        raise ValueError("检测结果为空，无法评估")
    print(f"[INFO] Loaded {len(detections)} detections, evaluating {iou_type}")
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print(f"[INFO] {iou_type} evaluation complete.")


def main():
    parser = argparse.ArgumentParser(description="Mask R-CNN 评估脚本")
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--det-file", required=True)
    parser.add_argument("--types", nargs="+", default=["bbox", "segm"])
    args = parser.parse_args()

    for eval_type in args.types:
        print(f"\n=== Evaluating {eval_type} ===")
        coco_evaluate(args.ann_file, args.det_file, iou_type=eval_type)


if __name__ == "__main__":
    main()

