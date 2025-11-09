import json
import os
import pickle
from typing import Dict, List, Optional

import numpy as np  # pyright: ignore[reportMissingImports]
from PIL import Image  # pyright: ignore[reportMissingImports]
from torch.utils.data import Dataset  # pyright: ignore[reportMissingImports]
from torchvision import transforms  # pyright: ignore[reportMissingImports]

class_names = ["__background__", "person", "dog", "cat", "car", "bicycle"]
name_to_idx = {name: idx for idx, name in enumerate(class_names)}


class COCOSubsetDataset(Dataset):
    """轻量化 COCO 子集数据集，提供图像、候选框与标注信息。"""

    def __init__(self, image_dir, ann_file, proposal_dir, transform=None, max_proposals=1500):
        self.image_dir = image_dir
        self.proposal_dir = proposal_dir
        self.max_proposals = max_proposals

        with open(ann_file, "r") as f:
            data = json.load(f)
        self.images = data["images"]
        anns = data["annotations"]
        self.image_to_anns: Dict[int, List[dict]] = {}
        for ann in anns:
            image_id = ann["image_id"]
            self.image_to_anns.setdefault(image_id, []).append(ann)

        self.id_to_filename = {img["id"]: img["file_name"] for img in self.images}
        self.categories = data["categories"]
        self.cat_id_to_class_idx = {
            cat["id"]: name_to_idx[cat["name"]] for cat in self.categories if cat["name"] in name_to_idx
        }

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((227, 227)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        file_name = img_info["file_name"]
        img_id = img_info["id"]
        img_path = os.path.join(self.image_dir, file_name)
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        proposal_path = os.path.join(self.proposal_dir, f"{os.path.splitext(file_name)[0]}.pkl")
        with open(proposal_path, "rb") as fp:
            proposals = np.array(pickle.load(fp)[: self.max_proposals], dtype=np.float32)
        proposals = self._sanitize_boxes(proposals, width, height)

        annotations = []
        for ann in self.image_to_anns.get(img_id, []):
            if ann["category_id"] not in self.cat_id_to_class_idx:
                continue
            bbox = ann["bbox"]  # COCO: [x, y, w, h]
            x, y, w, h = bbox
            clean_box = self._sanitize_box(np.array([x, y, x + w, y + h], dtype=np.float32), width, height)
            if clean_box is None:
                continue
            annotations.append(
                {
                    "bbox": clean_box,
                    "category_id": ann["category_id"],
                    "label": self.cat_id_to_class_idx[ann["category_id"]],
                }
            )

        return {
            "image": image,
            "image_id": img_id,
            "file_name": file_name,
            "path": img_path,
            "proposals": proposals,
            "annotations": annotations,
        }

    def _sanitize_box(self, box: np.ndarray, width: int, height: int) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = box.tolist()
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        x1 = float(np.clip(x1, 0, width - 1))
        y1 = float(np.clip(y1, 0, height - 1))
        x2 = float(np.clip(x2, 0, width - 1))
        y2 = float(np.clip(y2, 0, height - 1))
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _sanitize_boxes(self, boxes: np.ndarray, width: int, height: int) -> np.ndarray:
        if boxes.size == 0:
            return boxes.reshape(-1, 4)
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
        sanitized = np.stack([x1, y1, x2, y2], axis=1)
        if not np.any(valid):
            return np.empty((0, 4), dtype=np.float32)
        return sanitized[valid]