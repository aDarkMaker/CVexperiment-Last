## Mask R-CNN 轻量化复现指南

本目录基于 `torchvision` 官方 `maskrcnn_resnet50_fpn` 模型，结合前面生成的 COCO 子集，实现快速训练、推理与评估流程，满足“得到数值与可视化结果”的需求。

### 1. 依赖与环境
- 复用项目根目录的虚拟环境（建议执行 `source scripts/venv.active.sh`）。
- 关键依赖：`torch`, `torchvision`, `pycocotools`, `opencv-python`, `matplotlib`, `numpy`, `tqdm`。
- 训练脚本会将预训练权重缓存到 `Last/.cache/torch/` 目录，满足“下载内容集中在项目内”的要求。

### 2. 数据集准备
沿用 `Last/data/raw/coco_subset/` 与 `Last/data/proposals/coco_subset/`，确保标注 JSON 中包含 `segmentation`（COCO 官方 `segmentation` 字段默认保留）。

若还未生成子集，可在项目根目录执行：
```bash
python Mask-R-CNN/../R-CNN/datasets/coco_subset.py \
  --image-dir data/raw/coco2017/val2017 \
  --ann-file data/raw/coco2017/annotations/instances_val2017.json \
  --out-dir data/raw/coco_subset
```

### 3. 训练 Mask R-CNN
```bash
export PYTHONPATH=Mask-R-CNN
python Mask-R-CNN/training/train_mask_rcnn.py \
  --image-dir data/raw/coco_subset/images \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --categories person dog cat car bicycle \
  --epochs 10 \
  --batch-size 2 \
  --output-dir results/mask_rcnn
```

说明：
- `--categories` 对应子集类别，可与 R-CNN 保持一致。
- 默认使用预训练权重；如需从头训练可加 `--no-pretrained`。
- 训练会保存每个 epoch 的 checkpoint（`mask_rcnn_epoch*.pth`）及最终权重 `mask_rcnn_final.pth`。
- 如在 MPS 设备上（Mac），`pin_memory` 会自动关闭，日志中的警告可忽略。

### 4. 推理与结果导出
训练完成后执行：
```bash
python Mask-R-CNN/interface/predict.py \
  --image-dir data/raw/coco_subset/images \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --weights results/mask_rcnn/mask_rcnn_final.pth \
  --categories person dog cat car bicycle \
  --score-thresh 0.5 \
  --out-json results/mask_rcnn/detections.json
```

参数说明：
- `--score-thresh` 控制置信度阈值。
- `--limit` 可限制推理张数，调试时很有用。
- `--visualize-dir`（可选）若指定，将输出叠加 Mask 的可视化图像。

输出的 `detections.json` 兼容 COCO 格式，包含 `bbox` 与 `segmentation`（RLE 编码）。

### 5. 评估指标
使用 COCO 官方评价（支持 `bbox` 与 `segm`）：
```bash
python Mask-R-CNN/interface/evaluate.py \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --det-file results/mask_rcnn/detections.json \
  --types bbox segm
```
将输出 AP、AP50、AP75 等关键指标，可直接用于与 R-CNN、Fast/Faster R-CNN 对比。

### 6. 可视化建议
- 推理脚本 `--visualize-dir` 会输出叠加 Mask 的 PNG。
- 也可以在 Notebook 中读取 `detections.json`，使用 `pycocotools` 的 `mask.decode` 与 `matplotlib` 绘制。

### 7. 结果整理与对比
建议把训练日志、评估结果、可视化截图集中到：
- `report/experiment_logs/mask_rcnn.txt`
- `report/figures/mask_rcnn/`
- `results/mask_rcnn/`（保存权重与 JSON）

