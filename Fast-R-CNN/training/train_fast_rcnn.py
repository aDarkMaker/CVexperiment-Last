import argparse
import os
import random
import time
from typing import List, Tuple

import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
import torch.optim as optim  # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader  # pyright: ignore[reportMissingImports]
from torchvision.ops import box_iou, nms  # pyright: ignore[reportMissingImports]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "torch")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("TORCH_HOME", CACHE_DIR)

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from datasets.coco_dataset import COCOSubsetFastDataset, collate_fn, class_names  # noqa: E402
from models.fast_rcnn import FastRCNN, bbox_transform_targets, smooth_l1_loss  # noqa: E402


def sample_proposals(
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    fg_thresh: float = 0.5,
    bg_thresh_hi: float = 0.5,
    bg_thresh_lo: float = 0.1,
    batch_size: int = 128,
    fg_fraction: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if gt_boxes.numel() == 0:
        idx = torch.arange(min(batch_size, proposals.size(0)))
        labels = torch.zeros(idx.size(0), dtype=torch.int64)
        return proposals[idx], labels, torch.zeros((idx.size(0), 4), dtype=torch.float32)

    overlaps = box_iou(proposals, gt_boxes)
    max_overlaps, gt_assignment = overlaps.max(dim=1)

    fg_inds = torch.nonzero(max_overlaps >= fg_thresh).squeeze(1)
    bg_inds = torch.nonzero((max_overlaps < bg_thresh_hi) & (max_overlaps >= bg_thresh_lo)).squeeze(1)

    num_fg = int(fg_fraction * batch_size)
    num_fg = min(num_fg, fg_inds.numel())
    if num_fg > 0:
        fg_inds = fg_inds[torch.randperm(fg_inds.numel())[:num_fg]]
    num_bg = batch_size - num_fg
    num_bg = min(num_bg, bg_inds.numel())
    if num_bg > 0:
        bg_inds = bg_inds[torch.randperm(bg_inds.numel())[:num_bg]]

    keep_inds = torch.cat([fg_inds, bg_inds], dim=0)
    sampled_rois = proposals[keep_inds]

    labels = gt_labels[gt_assignment[keep_inds]]
    labels[num_fg:] = 0  # background label is 0

    bbox_targets = torch.zeros((keep_inds.size(0), 4), dtype=torch.float32)
    if num_fg > 0:
        bbox_targets[:num_fg] = bbox_transform_targets(sampled_rois[:num_fg], gt_boxes[gt_assignment[fg_inds]])
    return sampled_rois, labels, bbox_targets


def compute_bbox_loss(pred_deltas, labels, bbox_targets, num_classes, beta=1.0):
    if bbox_targets.numel() == 0:
        return torch.tensor(0.0, device=pred_deltas.device)

    pred_deltas = pred_deltas.view(-1, num_classes, 4)
    positive = torch.nonzero(labels > 0).squeeze(1)
    if positive.numel() == 0:
        return torch.tensor(0.0, device=pred_deltas.device)

    pos_labels = labels[positive]
    pred_pos = pred_deltas[positive, pos_labels]
    target_pos = bbox_targets[positive].to(pred_pos.device)
    return smooth_l1_loss(pred_pos, target_pos, beta=beta)


def train_one_epoch(
    model: FastRCNN,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    print_freq: int = 10,
):
    model.train()
    running_loss = 0.0
    total_batches = len(data_loader)
    batch_time = time.perf_counter()

    for batch_idx, (images, proposals_list, targets) in enumerate(data_loader, start=1):
        images = torch.stack(images).to(device)
        proposals = proposals_list[0].to(device)
        target = targets[0]
        gt_boxes = target["boxes"].to(device)
        gt_labels = target["labels"].to(device)

        sampled_rois, labels, bbox_targets = sample_proposals(
            proposals,
            gt_boxes,
            gt_labels,
        )
        if sampled_rois.numel() == 0:
            print(f"[WARN] batch {batch_idx}: 没有采样到有效 ROI，跳过本批次")
            continue

        sampled_rois = sampled_rois.to(device)
        zeros = torch.zeros((sampled_rois.size(0), 1), device=device)
        rois = torch.cat([zeros, sampled_rois], dim=1)
        scores, bbox_deltas = model(images, rois)
        labels = labels.to(device)
        bbox_targets = bbox_targets.to(device)

        cls_loss = nn.CrossEntropyLoss()(scores, labels)
        bbox_loss = compute_bbox_loss(bbox_deltas, labels, bbox_targets, model.num_classes)
        loss = cls_loss + bbox_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % print_freq == 0 or batch_idx == total_batches:
            elapsed = time.perf_counter() - batch_time
            avg_loss = running_loss / batch_idx
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[train] batch {batch_idx}/{total_batches}, loss={avg_loss:.4f}, lr={lr:.5f}, elapsed={elapsed:.1f}s",
                flush=True,
            )
            batch_time = time.perf_counter()

    return running_loss / max(total_batches, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Fast R-CNN 训练脚本")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--proposal-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "results", "fast_rcnn"))
    parser.add_argument("--max-proposals", type=int, default=1000)
    parser.add_argument("--print-freq", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    dataset = COCOSubsetFastDataset(
        image_dir=args.image_dir,
        ann_file=args.ann_file,
        proposal_dir=args.proposal_dir,
        max_proposals=args.max_proposals,
    )
    print(f"[INFO] Dataset loaded: {len(dataset)} images")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    print(f"[INFO] DataLoader ready: batch_size={args.batch_size}, num_workers={args.num_workers}, batches={len(data_loader)}")

    model = FastRCNN(num_classes=len(class_names)).to(device)
    print("[INFO] Fast R-CNN model initialized（ResNet50 backbone + ROIAlign）")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    print(f"[INFO] Optimizer: SGD lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay}")

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print(f"[INFO] Checkpoints will be saved to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        print(f"[INFO] ===== Start Epoch {epoch}/{args.epochs} =====")
        avg_loss = train_one_epoch(model, data_loader, optimizer, device, epoch, print_freq=args.print_freq)
        print(f"[INFO] Epoch {epoch:02d} finished: loss={avg_loss:.4f}")
        lr_scheduler.step()
        ckpt_path = os.path.join(args.output_dir, f"fast_rcnn_epoch{epoch}.pth")
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, ckpt_path)
        print(f"[Checkpoint] Saved {ckpt_path}")

    final_path = os.path.join(args.output_dir, "fast_rcnn_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Final weights saved to {final_path}")
    print("[INFO] Training completed")


if __name__ == "__main__":
    main()

