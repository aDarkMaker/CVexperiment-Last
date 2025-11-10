## R-CNN 复现说明（COCO 子集）

该目录提供轻量化 R-CNN 复现方案，基于 COCO 2017 的精简子集（5 个类别），适合在本地 GPU 环境快速跑通“候选框 + CNN + SVM + BBox 回归”完整流程。以下所有命令均提供 **macOS/Linux（Bash）** 与 **Windows PowerShell** 两种写法，可根据系统选择执行。

---

### 1. 数据准备

> 前置：请先参照项目根目录 `README.md` 下载 COCO `val2017` 与标注文件到 `data/raw/coco2017/`。

#### 1.1 生成 COCO 轻量子集

**Bash**（在项目根目录执行）
```bash
python R-CNN/datasets/coco_subset.py \
  --image-dir data/raw/coco2017/val2017 \
  --ann-file data/raw/coco2017/annotations/instances_val2017.json \
  --out-dir data/raw/coco_subset
```

**PowerShell**（在项目根目录执行）
```powershell
python R-CNN/datasets/coco_subset.py `
  --image-dir data/raw/coco2017/val2017 `
  --ann-file data/raw/coco2017/annotations/instances_val2017.json `
  --out-dir data/raw/coco_subset
```

#### 1.2 生成 Selective Search proposals

**Bash**
```bash
python R-CNN/datasets/generate_proposals.py \
  --image-dir data/raw/coco_subset/images \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --out-dir data/proposals/coco_subset
```

**PowerShell**
```powershell
python R-CNN/datasets/generate_proposals.py `
  --image-dir data/raw/coco_subset/images `
  --ann-file data/raw/coco_subset/instances_subset.json `
  --out-dir data/proposals/coco_subset
```

> ⚠️ `generate_proposals.py` 依赖 `opencv-contrib-python`。若提示缺少 `ximgproc`，请确认依赖已安装。

---

### 2. 微调 CNN（AlexNet）

**Bash**
```bash
export PYTHONPATH=R-CNN  # 也可在脚本内部 append sys.path
python R-CNN/training/finetune_cnn.py \
  --image-dir data/raw/coco_subset/images \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --proposal-dir data/proposals/coco_subset \
  --epochs 5 \
  --batch-size 64 \
  --feature-ckpt results/rcnn/feature.pth \
  --classifier-ckpt results/rcnn/classifier.pth
```

**PowerShell**
```powershell
$env:PYTHONPATH = "R-CNN"
python R-CNN/training/finetune_cnn.py `
  --image-dir data/raw/coco_subset/images `
  --ann-file data/raw/coco_subset/instances_subset.json `
  --proposal-dir data/proposals/coco_subset `
  --epochs 5 `
  --batch-size 64 `
  --feature-ckpt results/rcnn/feature.pth `
  --classifier-ckpt results/rcnn/classifier.pth
```

脚本会根据 ROI 正负样本（IoU ≥ 0.5 为正、IoU < 0.3 为负）微调 `fc6/fc7`，并保存特征网络与分类器权重。

---

### 3. 缓存 ROI 特征

**Bash**
```bash
python R-CNN/training/extract_features.py \
  --image-dir data/raw/coco_subset/images \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --proposal-dir data/proposals/coco_subset \
  --feature-ckpt results/rcnn/feature.pth \
  --out-dir data/cache/coco_subset \
  --max-proposals 800
```

**PowerShell**
```powershell
python R-CNN/training/extract_features.py `
  --image-dir data/raw/coco_subset/images `
  --ann-file data/raw/coco_subset/instances_subset.json `
  --proposal-dir data/proposals/coco_subset `
  --feature-ckpt results/rcnn/feature.pth `
  --out-dir data/cache/coco_subset `
  --max-proposals 800
```

输出：
- `*_features.npy`：每个候选框的 4096 维 fc7 特征  
- `*_proposals.pkl`：对应候选框坐标（与特征文件配套）

---

### 4. 训练 SVM 分类器与 BBox 回归器

**Bash**
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

**PowerShell**
```powershell
python R-CNN/training/train_svm.py `
  --cache-dir data/cache/coco_subset `
  --ann-file data/raw/coco_subset/instances_subset.json `
  --out-path results/rcnn/svm_models.pkl

python R-CNN/training/train_bbox_reg.py `
  --cache-dir data/cache/coco_subset `
  --ann-file data/raw/coco_subset/instances_subset.json `
  --out-path results/rcnn/bbox_reg.pkl
```

---

### 5. 推理与评估

#### 5.1 生成检测结果

**Bash**
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

**PowerShell**
```powershell
python R-CNN/inference/predict.py `
  --image-dir data/raw/coco_subset/images `
  --ann-file data/raw/coco_subset/instances_subset.json `
  --proposal-dir data/proposals/coco_subset `
  --feature-ckpt results/rcnn/feature.pth `
  --svm-model results/rcnn/svm_models.pkl `
  --bbox-reg results/rcnn/bbox_reg.pkl `
  --out-json results/rcnn/detections.json
```

#### 5.2 COCO 指标评估

**Bash**
```bash
python R-CNN/inference/evaluate.py \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --det-file results/rcnn/detections.json
```

**PowerShell**
```powershell
python R-CNN/inference/evaluate.py `
  --ann-file data/raw/coco_subset/instances_subset.json `
  --det-file results/rcnn/detections.json
```

如需仅评估部分类别，可追加 `--categories person dog`。

> 可视化可使用 `utils/visualization.py`（或自建 Notebook）读取 `detections.json` 与原图叠加展示。

---

### 6. 目录说明

- `datasets/`：COCO 子集生成、数据集封装、proposal 脚本  
- `models/`：特征提取骨干（AlexNet fc6/fc7）  
- `training/`：微调 CNN / 缓存特征 / SVM / BBox 回归脚本  
- `inference/`：推理与评估脚本  
- `results/rcnn/`：训练输出（权重、预测、评估）  
- `data/`：原始数据、proposals、缓存特征

---

### 7. 依赖

关键依赖如下（完整列表见仓库根目录 `requirements.txt`）：

- `torch`, `torchvision`
- `scikit-learn`
- `opencv-contrib-python`
- `pycocotools`
- `numpy`, `scipy`, `tqdm`

macOS/Linux 可执行 `./scripts/updateRequir.sh` 同步依赖；Windows 可手动 `pip install -r requirements.txt`。

