import argparse
import json
import os

from pycocotools.coco import COCO  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
from pycocotools.cocoeval import COCOeval  # pyright: ignore[reportMissingImports, reportMissingModuleSource]


def main():
    parser = argparse.ArgumentParser(description="Fast R-CNN 评估脚本")
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--det-file", required=True)
    args = parser.parse_args()

    coco = COCO(args.ann_file)
    if "info" not in coco.dataset:
        coco.dataset["info"] = {"description": "subset"}
    if "licenses" not in coco.dataset:
        coco.dataset["licenses"] = []
    if not os.path.exists(args.det_file):
        raise FileNotFoundError(f"未找到检测结果：{args.det_file}")
    with open(args.det_file, "r") as f:
        detections = json.load(f)
    if not detections:
        raise ValueError("检测结果为空，无法评估")
    print(f"[INFO] Loaded {len(detections)} detections, running COCO bbox evaluation")
    coco_dt = coco_gt.loadRes(detections)  # pyright: ignore[reportUndefinedVariable]
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")  # pyright: ignore[reportUndefinedVariable]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("[INFO] Evaluation complete.")


if __name__ == "__main__":
    main()

