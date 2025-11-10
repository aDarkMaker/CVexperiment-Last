from typing import List, Tuple

import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
import torch.nn.functional as F  # pyright: ignore[reportMissingImports]
from torchvision import models  # pyright: ignore[reportMissingImports]
from torchvision.ops import roi_align  # pyright: ignore[reportMissingImports]


class FastRCNN(nn.Module):
    def __init__(self, num_classes: int, roi_output_size: int = 7, backbone: str = "resnet50"):
        super().__init__()
        if backbone == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            modules = list(resnet.children())[:-2]
            self.backbone = nn.Sequential(*modules)
            self.out_channels = 2048
            self.stride = 32
        else:
            raise ValueError("Unsupported backbone")

        self.roi_output_size = roi_output_size
        flattened_dim = self.out_channels * roi_output_size * roi_output_size

        self.fc1 = nn.Linear(flattened_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

        for layer in [self.fc1, self.fc2, self.cls_score, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

        self.num_classes = num_classes

    def forward(self, images: torch.Tensor, proposals: torch.Tensor):
        """
        images: (N,3,H,W)
        proposals: (M,5) -> [batch_idx, x1, y1, x2, y2]
        """
        feature_map = self.backbone(images)
        pooled = roi_align(
            feature_map,
            proposals,
            output_size=(self.roi_output_size, self.roi_output_size),
            spatial_scale=1.0 / self.stride,
            sampling_ratio=0,
        )
        pooled_flat = pooled.view(pooled.size(0), -1)
        x = F.relu(self.fc1(pooled_flat))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


def bbox_transform(boxes: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes


def bbox_transform_targets(ex_rois: torch.Tensor, gt_rois: torch.Tensor) -> torch.Tensor:
    widths = ex_rois[:, 2] - ex_rois[:, 0]
    heights = ex_rois[:, 3] - ex_rois[:, 1]
    ctr_x = ex_rois[:, 0] + 0.5 * widths
    ctr_y = ex_rois[:, 1] + 0.5 * heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0]
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1]
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets = torch.zeros_like(ex_rois)
    targets[:, 0] = (gt_ctr_x - ctr_x) / widths
    targets[:, 1] = (gt_ctr_y - ctr_y) / heights
    targets[:, 2] = torch.log(gt_widths / widths)
    targets[:, 3] = torch.log(gt_heights / heights)
    return targets


def smooth_l1_loss(input: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean()

