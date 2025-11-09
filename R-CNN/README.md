## R-CNN 复现说明（COCO 子集）

该目录提供轻量化 R-CNN 复现方案，基于 COCO 2017 的精简子集（5 个类别），适合在本地 GPU 环境快速跑通“候选框 + CNN + SVM + BBox 回归”完整流程。

### 1. 数据准备

1. 下载 COCO 2017 `val2017` 图像与 `annotations_trainval2017` 标注（可放在 `data/raw/coco2017/`）。
2. 生成轻量子集（默认每类 200 张图像）：

```bash
python R-CNN/datasets/coco_subset.py \
  --image-dir data/raw/coco2017/val2017 \
  --ann-file data/raw/coco2017/annotations/instances_val2017.json \
  --out-dir data/raw/coco_subset
```

3. 对子集生成 Selective Search proposals：

```bash
python R-CNN/datasets/generate_proposals.py \
  --image-dir data/raw/coco_subset/images \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --out-dir data/proposals/coco_subset
```

> ⚠️ `generate_proposals.py` 基于 `opencv-contrib-python`，若报错请确认依赖已安装。

### 2. 微调 CNN（AlexNet）

```bash
export PYTHONPATH=R-CNN  # 或在脚本顶部手动追加 sys.path

python R-CNN/training/finetune_cnn.py \
  --image-dir data/raw/coco_subset/images \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --proposal-dir data/proposals/coco_subset \
  --epochs 5 \
  --batch-size 64 \
  --feature-ckpt results/rcnn/feature.pth \
  --classifier-ckpt results/rcnn/classifier.pth
```

脚本会基于 ROI 样本（IoU≥0.5 判为前景，IoU<0.3 判为背景）微调 `fc6/fc7`，并保存特征网络与分类器权重。

### 3. 缓存 ROI 特征

```bash
python R-CNN/training/extract_features.py \
  --image-dir data/raw/coco_subset/images \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --proposal-dir data/proposals/coco_subset \
  --feature-ckpt results/rcnn/feature.pth \
  --out-dir data/cache/coco_subset \
  --max-proposals 800
```

输出：
- `*_features.npy`：每个候选框的 4096 维 fc7 特征
- `*_proposals.pkl`：对应的候选框坐标

### 4. 训练 SVM 分类器与 BBox 回归器

```bash
# 线性 SVM
python R-CNN/training/train_svm.py \
  --cache-dir data/cache/coco_subset \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --out-path results/rcnn/svm_models.pkl

# 边框回归
python R-CNN/training/train_bbox_reg.py \
  --cache-dir data/cache/coco_subset \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --out-path results/rcnn/bbox_reg.pkl
```

### 5. 推理与可视化/评估

1. 生成检测结果：

```bash
python R-CNN/inference/predict.py \
  --image-dir data/raw/coco_subset/images \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --proposal-dir data/proposals/coco_subset \
  --feature-ckpt results/rcnn/feature.pth \
  --svm-model results/rcnn/svm_models.pkl \
  --bbox-reg results/rcnn/bbox_reg.pkl \
  --out-json results/rcnn/detections.json
```

2. 使用 COCO 官方评估：

```bash
python R-CNN/inference/evaluate.py \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --det-file results/rcnn/detections.json
```

如需仅评估部分类别，可追加 `--categories person dog`。

> 可视化功能可在 `output/` 目录自行扩展，例如编写 notebook 读取 `detections.json` 并绘制检测框。

### 6. 目录说明

- `datasets/`：COCO 子集生成、数据集封装、proposal 脚本
- `models/`：特征提取骨干（AlexNet fc6/fc7）
- `training/`：微调、特征缓存、SVM、BBox 回归等脚本
- `inference/`：推理与评估脚本
- `results/rcnn/`：训练输出（权重、预测、评估）
- `data/`：原始数据、proposals、缓存特征（请根据磁盘情况选择 symlink）

### 7. 依赖

确保虚拟环境安装以下关键包（详见仓库根目录 `requirements.txt`）：

- `torch`, `torchvision`
- `scikit-learn`
- `opencv-contrib-python`
- `pycocotools`
- `numpy`, `scipy`, `tqdm`

执行 `scripts/updateRequir.sh` 可同步最新依赖到 `requirements.txt`。

