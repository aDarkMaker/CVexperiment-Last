import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import matplotlib.patches as patches  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from PIL import Image  # pyright: ignore[reportMissingImports]
from pycocotools.coco import COCO  # pyright: ignore[reportMissingModuleSource]
from pycocotools import mask as mask_utils  # pyright: ignore[reportMissingModuleSource]


def load_detections(json_path: str) -> Dict[int, List[dict]]:
    with open(json_path, "r") as f:
        data = json.load(f)
    grouped: Dict[int, List[dict]] = {}
    for det in data:
        grouped.setdefault(int(det["image_id"]), []).append(det)
    return grouped


def _denormalize_bbox(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h


def _draw_bbox(ax, bbox, score=None, color="lime", label=None):
    x1, y1, x2, y2 = _denormalize_bbox(bbox)
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)
    caption = ""
    if label:
        caption += label
    if score is not None:
        caption += f" {score:.2f}"
    if caption:
        ax.text(x1, y1 - 3, caption, color=color, fontsize=10, backgroundcolor="black")


def _decode_mask(segmentation, height, width):
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        rle = segmentation
    else:
        raise ValueError("Unsupported segmentation format")
    return mask_utils.decode(rle)


def visualize_models(
    coco_ann_file: str,
    image_root: str,
    model_results: Dict[str, str],
    image_ids: Optional[List[int]] = None,
    score_thresh: float = 0.5,
    show_gt: bool = False,
    save_dir: Optional[str] = None,
):
    """
    coco_ann_file: 原始标注 JSON，方便加载图像和类别名称
    image_root: 图像目录
    model_results: { 模型名: detections.json 路径 }
    image_ids: 指定要展示的 image_id 列表；未提供则随机抽取 len(model_results) 张
    score_thresh: 最低显示的预测分数
    show_gt: 是否在原图面板叠加 GT
    save_dir: 若提供，将保存可视化图片；否则直接 plt.show
    """
    coco = COCO(coco_ann_file)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}

    detections_per_model = {
        model_name: load_detections(result_path)
        for model_name, result_path in model_results.items()
    }

    if not image_ids:
        image_ids = coco.getImgIds()
        image_ids = image_ids[: min(len(image_ids), len(model_results))]

    for image_id in image_ids:
        img_info = coco.loadImgs([image_id])[0]
        img_path = Path(image_root) / img_info["file_name"]
        image = np.array(Image.open(img_path).convert("RGB"))

        num_cols = 1 + len(model_results)
        fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))
        if num_cols == 1:
            axes = [axes]

        axes[0].imshow(image)
        axes[0].set_title(f"Original: {img_info['file_name']}")
        axes[0].axis("off")

        if show_gt:
            ann_ids = coco.getAnnIds(imgIds=[image_id])
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                _draw_bbox(
                    axes[0],
                    ann["bbox"],
                    color="cyan",
                    label=cat_id_to_name.get(ann["category_id"], "GT"),
                )

        for idx, (model_name, detections) in enumerate(detections_per_model.items(), start=1):
            ax = axes[idx]
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(model_name)

            preds = detections.get(image_id, [])
            for pred in preds:
                score = pred.get("score", 1.0)
                if score < score_thresh:
                    continue
                cat_name = cat_id_to_name.get(pred["category_id"], "unknown")
                _draw_bbox(ax, pred["bbox"], score=score, color="yellow", label=cat_name)

                if "segmentation" in pred and pred["segmentation"]:
                    mask = _decode_mask(pred["segmentation"], image.shape[0], image.shape[1])
                    color_mask = np.zeros_like(image)
                    color_mask[:, :, 1] = mask * 255  # green channel
                    ax.imshow(np.ma.masked_where(mask == 0, color_mask), alpha=0.4)

        fig.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = Path(save_dir) / f"compare_{image_id}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()