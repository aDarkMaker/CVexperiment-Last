import argparse
import datetime
import os
import sys
import time
from typing import List

import torch  # pyright: ignore[reportMissingImports]
import torch.optim as optim  # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader  # pyright: ignore[reportMissingImports]
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


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int, pretrained: bool = True):
    if pretrained:
        model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    else:
        model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    # 替换 box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 替换 mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20, scaler=None):
    model.train()
    total_batches = len(data_loader)
    running_loss = 0.0
    batch_start = time.perf_counter()

    for batch_idx, (images, targets, _) in enumerate(data_loader, start=1):
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets
        ]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        loss_value = losses.item()
        running_loss += loss_value

        if batch_idx % print_freq == 0 or batch_idx == total_batches:
            elapsed = time.perf_counter() - batch_start
            avg_loss = running_loss / batch_idx
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[train] batch {batch_idx}/{total_batches}, loss={avg_loss:.4f}, lr={lr:.5f}, elapsed={elapsed:.1f}s",
                flush=True,
            )
            batch_start = time.perf_counter()

    return running_loss / max(total_batches, 1)


def save_checkpoint(model, optimizer, epoch, out_dir, prefix="mask_rcnn"):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{prefix}_epoch{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        ckpt_path,
    )
    print(f"[Checkpoint] Saved {ckpt_path}")
    return ckpt_path


def parse_args():
    parser = argparse.ArgumentParser(description="Mask R-CNN 轻量复现训练脚本")
    parser.add_argument("--image-dir", required=True, help="训练图像目录")
    parser.add_argument("--ann-file", required=True, help="COCO 风格标注 JSON")
    parser.add_argument("--categories", nargs="*", default=None, help="参与训练的类别名")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "results", "mask_rcnn"))
    parser.add_argument("--resume", default=None, help="继续训练的 checkpoint 路径")
    parser.add_argument("--no-pretrained", action="store_true", help="不加载预训练权重")
    parser.add_argument("--use-amp", action="store_true", help="启用混合精度训练")
    parser.add_argument("--eval-every", type=int, default=0, help="暂未实现")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"[INFO] Using device: {device}")

    dataset = COCOSubsetInstanceDataset(
        image_dir=args.image_dir,
        ann_file=args.ann_file,
        categories=args.categories,
    )
    print(f"[INFO] Categories: {dataset.category_names}")
    print(f"[INFO] Training images: {len(dataset)}")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    num_classes = len(dataset.category_names) + 1
    model = build_model(num_classes=num_classes, pretrained=not args.no_pretrained)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"[INFO] Resumed from {args.resume} at epoch {start_epoch}")

    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == "cuda" else None

    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=scaler)
        print(f"[Epoch {epoch:02d}] loss={avg_loss:.4f}")
        lr_scheduler.step()
        save_checkpoint(model, optimizer, epoch, args.output_dir)

    final_path = os.path.join(args.output_dir, "mask_rcnn_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Final model weights saved to {final_path}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"[INFO] Training completed at {timestamp}")


if __name__ == "__main__":
    main()

