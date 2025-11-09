import argparse
import os
import random
import sys
from typing import List, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]
from PIL import Image  # pyright: ignore[reportMissingImports]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TORCH_CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "torch")
os.makedirs(TORCH_CACHE_DIR, exist_ok=True)
os.environ.setdefault("TORCH_HOME", TORCH_CACHE_DIR)

# 将项目根目录加入 pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
import torch.optim as optim  # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader, Dataset  # pyright: ignore[reportMissingImports]

try:
    torch.hub.set_dir(TORCH_CACHE_DIR)
except Exception:
    pass

from datasets.coco_dataset import COCOSubsetDataset, class_names  # noqa: E402
from models.feature_extractor import AlexNetExtractor  # noqa: E402
from utils.roi_ops import crop_and_resize, compute_iou  # noqa: E402


class RCNNTrainDataset(Dataset):
    """根据 proposals 生成 ROI 样本用于微调 CNN。"""

    def __init__(
        self,
        base_dataset: COCOSubsetDataset,
        max_proposals: int = 800,
        samples_per_image: int = 128,
        positive_iou: float = 0.5,
        negative_iou: float = 0.3,
        foreground_fraction: float = 0.25,
    ) -> None:
        self.transform = base_dataset.transform
        self.samples: List[Tuple[str, np.ndarray, int]] = []

        rng = random.Random(42)
        for idx in range(len(base_dataset)):
            item = base_dataset[idx]
            proposals = item["proposals"][:max_proposals]
            annotations = item["annotations"]
            if len(proposals) == 0 or len(annotations) == 0:
                continue

            gt_boxes = np.stack([ann["bbox"] for ann in annotations])
            gt_labels = np.array([ann["label"] for ann in annotations])

            positives: List[Tuple[str, np.ndarray, int]] = []
            negatives: List[Tuple[str, np.ndarray, int]] = []

            for box in proposals:
                overlaps = compute_iou(box, gt_boxes)
                max_iou = float(overlaps.max()) if overlaps.size else 0.0
                if max_iou >= positive_iou:
                    label = int(gt_labels[int(overlaps.argmax())])
                    positives.append((item["path"], box, label))
                elif max_iou < negative_iou:
                    negatives.append((item["path"], box, 0))

            if not positives and not negatives:
                continue

            max_pos = max(1, int(samples_per_image * foreground_fraction))
            max_neg = max(samples_per_image - max_pos, 1)

            rng.shuffle(positives)
            rng.shuffle(negatives)

            positives = positives[:max_pos]
            negatives = negatives[:max_neg]

            self.samples.extend(positives)
            self.samples.extend(negatives)

        if not self.samples:
            raise RuntimeError("未从数据集中采集到有效样本，请检查 proposals 或标注。")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, box, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        crop = crop_and_resize(image, box)
        tensor = self.transform(crop)
        return tensor, label


def train_one_epoch(loader, feature_model, classifier, criterion, optimizer, device):
    feature_model.train()
    classifier.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        features = feature_model(images)
        logits = classifier(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def main(args):
    torch.backends.cudnn.benchmark = True

    base_dataset = COCOSubsetDataset(
        image_dir=args.image_dir,
        ann_file=args.ann_file,
        proposal_dir=args.proposal_dir,
        max_proposals=args.max_proposals,
    )
    train_dataset = RCNNTrainDataset(
        base_dataset=base_dataset,
        max_proposals=args.max_proposals,
        samples_per_image=args.samples_per_image,
        positive_iou=args.positive_iou,
        negative_iou=args.negative_iou,
        foreground_fraction=args.foreground_fraction,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_model = AlexNetExtractor(pretrained=True).to(device)
    classifier = nn.Linear(4096, len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        list(feature_model.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    os.makedirs(os.path.dirname(args.feature_ckpt), exist_ok=True)

    for epoch in range(args.epochs):
        loss, acc = train_one_epoch(loader, feature_model, classifier, criterion, optimizer, device)
        print(f"[Epoch {epoch + 1:02d}] loss={loss:.4f} acc={acc:.4f}")

    torch.save(feature_model.state_dict(), args.feature_ckpt)
    torch.save(classifier.state_dict(), args.classifier_ckpt)
    print(f"特征网络已保存至 {args.feature_ckpt}")
    print(f"分类器已保存至 {args.classifier_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="COCO 子集图片目录")
    parser.add_argument("--ann-file", required=True, help="COCO 子集标注 json")
    parser.add_argument("--proposal-dir", required=True, help="预生成 proposals 的目录")
    parser.add_argument("--feature-ckpt", default="results/rcnn/feature.pth")
    parser.add_argument("--classifier-ckpt", default="results/rcnn/classifier.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-proposals", type=int, default=800)
    parser.add_argument("--samples-per-image", type=int, default=128)
    parser.add_argument("--positive-iou", type=float, default=0.5)
    parser.add_argument("--negative-iou", type=float, default=0.3)
    parser.add_argument("--foreground-fraction", type=float, default=0.25)
    args = parser.parse_args()
    main(args)

