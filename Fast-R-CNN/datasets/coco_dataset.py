import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]
from PIL import Image  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
from torch.utils.data import Dataset  # pyright: ignore[reportMissingImports]
from torchvision import transforms  # pyright: ignore[reportMissingImports]


class_names = ["__background__", "person", "dog", "cat", "car", "bicycle"]
name_to_idx = {name: idx for idx, name in enumerate(class_names)}


class COCOSubsetFastDataset(Dataset):
    """返回 Fast R-CNN 训练所需的图像、proposal 和标注。"""

    def __init__(
        self,
        image_dir: str,
        ann_file: str,
        proposal_dir: str,
        transform=None,
        max_proposals: int = 2000,
    ):
        self.image_dir = image_dir
        self.proposal_dir = proposal_dir
        self.max_proposals = max_proposals

        with open(ann_file, "r") as f:
            data = json.load(f)
        self.images = data["images"]
        annotations = data["annotations"]

        self.image_to_anns: Dict[int, List[dict]] = {}
        for ann in annotations:
            self.image_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.categories = data["categories"]
        self.cat_id_to_label = {
            cat["id"]: name_to_idx[cat["name"]] for cat in self.categories if cat["name"] in name_to_idx
        }
        self.label_to_cat_id = {label: cat_id for cat_id, label in self.cat_id_to_label.items()}
        self.ids = [img["id"] for img in self.images]

        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        info = self.images[idx]
        image_id = info["id"]
        file_name = info["file_name"]
        path = os.path.join(self.image_dir, file_name)
        image = Image.open(path).convert("RGB")

        width, height = image.size
        proposal_path = os.path.join(self.proposal_dir, f"{os.path.splitext(file_name)[0]}.pkl")
        with open(proposal_path, "rb") as fp:
            proposals = np.array(pickle.load(fp)[: self.max_proposals], dtype=np.float32)
        proposals = self._sanitize_boxes(proposals, width, height)

        anns = self.image_to_anns.get(image_id, [])
        boxes = []
        labels = []
        for ann in anns:
            if ann["category_id"] not in self.cat_id_to_label:
                continue
            bbox = ann["bbox"]
            x, y, w, h = bbox
            x1, y1, x2, y2 = self._sanitize_box(np.array([x, y, x + w, y + h], dtype=np.float32), width, height)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_label[ann["category_id"]])

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        image_tensor = self.transform(image)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "size": torch.tensor([height, width], dtype=torch.int32),
            "file_name": file_name,
        }

        return image_tensor, torch.from_numpy(proposals), target

    @staticmethod
    def _sanitize_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
        x1, y1, x2, y2 = box.tolist()
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        x1 = float(np.clip(x1, 0, width - 1))
        y1 = float(np.clip(y1, 0, height - 1))
        x2 = float(np.clip(x2, 0, width - 1))
        y2 = float(np.clip(y2, 0, height - 1))
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _sanitize_boxes(self, boxes: np.ndarray, width: int, height: int) -> np.ndarray:
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        x1 = np.minimum(boxes[:, 0], boxes[:, 2])
        x2 = np.maximum(boxes[:, 0], boxes[:, 2])
        y1 = np.minimum(boxes[:, 1], boxes[:, 3])
        y2 = np.maximum(boxes[:, 1], boxes[:, 3])

        x1 = np.clip(x1, 0, width - 1)
        y1 = np.clip(y1, 0, height - 1)
        x2 = np.clip(x2, 0, width - 1)
        y2 = np.clip(y2, 0, height - 1)

        valid = (x2 - x1) >= 1
        valid &= (y2 - y1) >= 1
        return np.stack([x1, y1, x2, y2], axis=1)[valid]


def collate_fn(batch):
    images, proposals, targets = zip(*batch)
    return list(images), [p for p in proposals], list(targets)

