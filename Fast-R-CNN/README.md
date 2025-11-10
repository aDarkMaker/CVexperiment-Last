## Fast R-CNN 复现指南

基于预先生成的 COCO 子集与 Selective Search proposals，使用 PyTorch 手工实现 Fast R-CNN（共享卷积特征 + ROI Pooling）训练、推理与评估流程。

> **前置准备**  
> - 已按照根目录 `README.md` 下载 COCO 2017 `val2017` 及标注。  
> - 已运行 `R-CNN/datasets/coco_subset.py` 与 `R-CNN/datasets/generate_proposals.py` 生成 `data/raw/coco_subset/` 与 `data/proposals/coco_subset/`。  
> - 激活虚拟环境（macOS/Linux: `source scripts/venv.active.sh`；Windows PowerShell: `.\scripts\venv.active.ps1` 若已自建）。

---

### 1. 训练 Fast R-CNN

**macOS / Linux（Bash）**
```bash
python Fast-R-CNN/training/train_fast_rcnn.py \
  --image-dir data/raw/coco_subset/images \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --proposal-dir data/proposals/coco_subset \
  --epochs 10 \
  --batch-size 1 \
  --output-dir results/fast_rcnn \
  --print-freq 10
```

**Windows PowerShell**（在项目根目录执行）
```powershell
$env:PYTHONPATH = "Fast-R-CNN"
python Fast-R-CNN/training/train_fast_rcnn.py `
  --image-dir data/raw/coco_subset/images `
  --ann-file data/raw/coco_subset/instances_subset.json `
  --proposal-dir data/proposals/coco_subset `
  --epochs 10 `
  --batch-size 1 `
  --output-dir results/fast_rcnn `
  --print-freq 10
```

训练脚本会每个 epoch 输出 `[Epoch xx] loss=...` 和 checkpoint (`fast_rcnn_epoch*.pth`)，最终保存 `fast_rcnn_final.pth`。

---

### 2. 推理生成检测结果

**Bash**
```bash
export PYTHONPATH=Fast-R-CNN
python Fast-R-CNN/interface/predict.py \
  --image-dir data/raw/coco_subset/images \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --proposal-dir data/proposals/coco_subset \
  --weights results/fast_rcnn/fast_rcnn_final.pth \
  --score-thresh 0.05 \
  --out-json results/fast_rcnn/detections.json
```

**PowerShell**（在项目根目录执行）
```powershell
$env:PYTHONPATH = "Fast-R-CNN"
python Fast-R-CNN/interface/predict.py `
  --image-dir data/raw/coco_subset/images `
  --ann-file data/raw/coco_subset/instances_subset.json `
  --proposal-dir data/proposals/coco_subset `
  --weights results/fast_rcnn/fast_rcnn_final.pth `
  --score-thresh 0.05 `
  --out-json results/fast_rcnn/detections.json
```

`detections.json` 为 COCO 兼容格式，可直接用于评估或可视化。

---

### 3. 评估 mAP

**Bash**
```bash
python Fast-R-CNN/interface/evaluate.py \
  --ann-file data/raw/coco_subset/instances_subset.json \
  --det-file results/fast_rcnn/detections.json
```

**PowerShell**
```powershell
python Fast-R-CNN/interface/evaluate.py `
  --ann-file data/raw/coco_subset/instances_subset.json `
  --det-file results/fast_rcnn/detections.json
```

输出与 COCO 官方脚本一致的 AP / AP50 / AP75 等指标，用于与 R-CNN、Faster/Mask R-CNN 对比。

---

### 4. 常见参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 1 | 当前实现仅支持单图训练；可在 GPU 环境尝试更大值 |
| `--max-proposals` | 1000 | 每张图使用的候选框数量 |
| `--print-freq` | 10 | 日志打印间隔（批次） |
| `--score-thresh` | 0.05 | 推理阶段最低置信度 |
| `--nms-thresh` | 0.3 | 推理阶段 NMS 阈值 |

---

### 5. 结果整理建议
- 训练日志与指标写入 `report/experiment_logs/fast_rcnn.txt`。
- 可视化图片（若使用 `utils/visualization.py`）放在 `report/figures/fast_rcnn/`。
- `results/fast_rcnn/` 中保存训练权重、`detections.json` 与评估输出。

> 同一批实验可使用 `utils/visualization.py` 将 Fast R-CNN 与其他模型预测叠加对比，便于撰写报告中“效果&效率”分析章节。

