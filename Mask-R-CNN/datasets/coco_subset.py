import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]
from PIL import Image  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
from pycocotools.coco import COCO  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
from torch.utils.data import Dataset  # pyright: ignore[reportMissingImports]
from torchvision import transforms  # pyright: ignore[reportMissingImports]


class COCOSubsetInstanceDataset(Dataset):
    """为 Mask R-CNN 轻量复现提供的 COCO 子集数据集。"""

    def __init__(
        self,
        image_dir: str,
        ann_file: str,
        categories: Optional[List[str]] = None,
        transforms_fn=None,
    ) -> None:
        self.image_dir = image_dir
        self.ann_file = ann_file
        self.coco = COCO(ann_file)

        if categories is None:
            categories = [cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.category_names = sorted(categories)

        self.cat_name_to_id: Dict[str, int] = {}
        self.cat_id_to_label: Dict[int, int] = {}
        for label_idx, name in enumerate(self.category_names, start=1):
            cat_ids = self.coco.getCatIds(catNms=[name])
            if not cat_ids:
                continue
            cat_id = cat_ids[0]
            self.cat_name_to_id[name] = cat_id
            self.cat_id_to_label[cat_id] = label_idx
        self.label_to_cat_id = {label: cat_id for cat_id, label in self.cat_id_to_label.items()}

        cat_ids = list(self.cat_id_to_label.keys())
        if not cat_ids:
            raise ValueError(
                "在标注文件中未找到指定类别。请检查 --categories 是否与 JSON 中的类别名称一致。"
            )
        image_ids = set()
        for cid in cat_ids:
            image_ids.update(self.coco.getImgIds(catIds=[cid]))
        self.ids = sorted(image_ids)
        if not self.ids:
            raise ValueError(
                "根据给定类别筛选后未找到任何图像。"
                "请确认子集 JSON 是否包含这些类别，或减少 --categories 列表。"
            )

        self.transforms_fn = transforms_fn or transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs([img_id])[0]
        path = img_info["file_name"]
        img_path = os.path.join(self.image_dir, path)

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=list(self.cat_id_to_label.keys()))
        anns = self.coco.loadAnns(ann_ids)

        boxes: List[List[float]] = []
        labels: List[int] = []
        masks: List[np.ndarray] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in self.cat_id_to_label:
                continue

            bbox = self._sanitize_box(ann["bbox"], width, height)
            if bbox is None:
                continue

            mask = self.coco.annToMask(ann)
            if mask.sum() == 0:
                continue

            boxes.append(bbox)
            labels.append(self.cat_id_to_label[cat_id])
            masks.append(mask.astype(np.uint8))
            areas.append(float(ann.get("area", mask.sum())))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if not boxes:
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]
            masks = [np.zeros((height, width), dtype=np.uint8)]
            areas = [1.0]
            iscrowd = [0]

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        masks_tensor = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        areas_tensor = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd_tensor = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
        }

        if self.transforms_fn:
            image = self.transforms_fn(image)

        return image, target, path

    @staticmethod
    def _sanitize_box(bbox: List[float], width: int, height: int) -> Optional[List[float]]:
        x, y, w, h = bbox
        x1 = max(0.0, min(x, width - 1))
        y1 = max(0.0, min(y, height - 1))
        x2 = max(0.0, min(x + w, width - 1))
        y2 = max(0.0, min(y + h, height - 1))

        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]


def collate_fn(batch):
    images, targets, paths = zip(*batch)
    return list(images), list(targets), list(paths)

