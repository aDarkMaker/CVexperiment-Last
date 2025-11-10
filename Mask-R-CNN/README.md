## Mask R-CNN 轻量化复现指南

基于 `torchvision` 官方 `maskrcnn_resnet50_fpn`，结合 COCO 轻量子集，实现训练 / 推理 / 评估流程。以下命令同时提供 **macOS/Linux（Bash）** 与 **Windows PowerShell** 版本。

### 1. 依赖与环境
- 建议在仓库根目录激活虚拟环境（Bash: `source scripts/venv.active.sh`，PowerShell: `.\scripts\venv.active.ps1` 或自行创建）。  
- 关键依赖：`torch`, `torchvision`, `pycocotools`, `opencv-python`, `matplotlib`, `numpy`, `tqdm`。  
- 预训练权重默认缓存到 `Last/.cache/torch/`，满足“下载集中在项目文件夹”要求。

### 2. 数据准备
沿用 `data/raw/coco_subset/`（轻量子集）与 `data/proposals/coco_subset/`（Selective Search），确保 JSON 中保留 `segmentation` 字段。

若尚未生成子集：

**Bash**
```bash
python R-CNN/datasets/coco_subset.py \
  --image-dir data/raw/coco2017/val2017 \
  --ann-file data/raw/coco2017/annotations/instances_val2017.json \
  --out-dir data/raw/coco_subset
```

**PowerShell**
```powershell
python R-CNN/datasets/coco_subset.py `
  --image-dir data/raw/coco2017/val2017 `
  --ann-file data/raw/coco2017/annotations/instances_val2017.json `
  --out-dir data/raw/coco_subset
```

### 3. 训练 Mask R-CNN

**Bash**
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

**PowerShell**
```powershell
$env:PYTHONPATH = "Mask-R-CNN"
python Mask-R-CNN/training/train_mask_rcnn.py `
  --image-dir data/raw/coco_subset/images `
  --ann-file data/raw/coco_subset/instances_subset.json `
  --categories person dog cat car bicycle `
  --epochs 10 `
  --batch-size 2 `
  --output-dir results/mask_rcnn
```

说明：
- `--categories` 指定参与训练的类别，可与 R-CNN/Fast R-CNN 对齐。  
- 默认加载预训练权重；若需从头训练添加 `--no-pretrained`。  
- 每个 epoch 生成 checkpoint；最终保存 `mask_rcnn_final.pth`。  
- 若在 MPS 环境运行，首次迭代较慢属正常，可将 `--num-workers` 调为 0。

### 4. 推理导出

**Bash**
```bash
python Mask-R-CNN/interface/predict.py \
  --image-dir data/raw/coco_subset/images \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --weights results/mask_rcnn/mask_rcnn_final.pth \
  --categories person dog cat car bicycle \
  --score-thresh 0.5 \
  --out-json results/mask_rcnn/detections.json
```

**PowerShell**
```powershell
python Mask-R-CNN/interface/predict.py `
  --image-dir data/raw/coco_subset/images `
  --ann-file data/raw/coco_subset/instances_subset.json `
  --weights results/mask_rcnn/mask_rcnn_final.pth `
  --categories person dog cat car bicycle `
  --score-thresh 0.5 `
  --out-json results/mask_rcnn/detections.json
```

可选参数：`--limit` 限制推理图像数，`--visualize-dir` 输出叠加 mask 的 PNG。

### 5. 评估指标

**Bash**
```bash
python Mask-R-CNN/interface/evaluate.py \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --det-file results/mask_rcnn/detections.json \
  --types bbox segm
```

**PowerShell**
```powershell
python Mask-R-CNN/interface/evaluate.py `
  --ann-file data/raw/coco_subset/instances_subset.json `
  --det-file results/mask_rcnn/detections.json `
  --types bbox segm
```

输出 COCO 官方的 AP、AP50、AP75，可与 R-CNN / Fast R-CNN / Faster R-CNN 对比。

### 6. 可视化建议
- 在推理命令中指定 `--visualize-dir` 生成覆盖 mask 的示意图。  
- 或使用 `utils/visualization.py`/Jupyter Notebook 读取 `detections.json` 与原图结合展示。

### 7. 结果整理
- 日志与指标：`report/experiment_logs/mask_rcnn.txt`  
- 图像可视化：`report/figures/mask_rcnn/`  
- 权重与检测结果：`results/mask_rcnn/`

