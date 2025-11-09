import argparse
import importlib
import json
import os


def load_coco_modules():
    try:
        coco_module = importlib.import_module("pycocotools.coco")
        eval_module = importlib.import_module("pycocotools.cocoeval")
    except ImportError as exc:  # pragma: no cover
        raise ImportError("请先安装 pycocotools: uv pip install pycocotools") from exc
    return coco_module.COCO, eval_module.COCOeval


def main(args):
    COCO, COCOeval = load_coco_modules()
    coco = COCO(args.ann_file)
    if not os.path.exists(args.det_file):
        raise FileNotFoundError(f"未找到检测结果：{args.det_file}")

    with open(args.det_file, "r") as f:
        detections = json.load(f)

    if not detections:
        raise ValueError("检测结果为空，无法评估")

    coco_dt = coco.loadRes(detections)
    coco_eval = COCOeval(coco, coco_dt, iouType="bbox")

    if args.categories:
        cat_ids = coco.getCatIds(catNms=args.categories)
        img_ids = coco.getImgIds(catIds=cat_ids)
        coco_eval.params.catIds = cat_ids
        coco_eval.params.imgIds = img_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", required=True, help="COCO 标注 json")
    parser.add_argument("--det-file", required=True, help="预测结果 json")
    parser.add_argument("--categories", nargs="*", default=None, help="指定评估的类别名，可选")
    args = parser.parse_args()
    main(args)

