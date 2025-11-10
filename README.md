目标检测是计算机视觉的基本任务之一，旨在从一副图像中用矩形框标注物体。典型方法包括 R-CNN、Fast R-CNN、Faster R-CNN、Mask R-CNN。本项目将复现其中三种并对比其优缺点。

---

## 数据集准备（COCO 2017 val）

### macOS / Linux（Bash）
```bash
# 在项目根目录执行
mkdir -p data/raw/coco2017
cd data/raw/coco2017

curl -L -o val2017.zip http://images.cocodataset.org/zips/val2017.zip
unzip -q val2017.zip

curl -L -o annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q annotations_trainval2017.zip
```

### Windows PowerShell
```powershell
# 在项目根目录执行
New-Item -ItemType Directory -Force -Path data/raw/coco2017 | Out-Null
Set-Location data/raw/coco2017

Invoke-WebRequest "http://images.cocodataset.org/zips/val2017.zip" -OutFile "val2017.zip"
Expand-Archive -Path "val2017.zip" -DestinationPath .

Invoke-WebRequest "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" -OutFile "annotations_trainval2017.zip"
Expand-Archive -Path "annotations_trainval2017.zip" -DestinationPath .
```

> 下载完成后，后续各子目录（`R-CNN`、`Fast-R-CNN`、`Mask-R-CNN` 等）将在 `data/raw/coco_subset/` 与 `data/proposals/coco_subset/` 上继续处理。

---

## 子项目指引

| 模型 | 说明 |
|------|------|
| `R-CNN/README.md` | 轻量化 R-CNN 复现流程（微调 AlexNet + SVM + BBox 回归）。已提供 Bash & PowerShell 指令。 |
| `Fast-R-CNN/README.md` | 手写 Fast R-CNN（共享卷积 + ROI Pooling），含训练、推理、评估指令。 |
| `Mask-R-CNN/README.md` | 基于 `torchvision` Mask R-CNN，提供训练、推理、COCO 评估步骤。 |

每个目录 README 中都列出了 macOS/Linux 与 Windows PowerShell 两套命令，可直接对照执行。

---

## 其他
- `scripts/updateRequir.sh`：更新 `requirements.txt` 的辅助脚本（Bash）。  
- `utils/visualization.py`：对比不同模型检测结果的绘图工具。  
- `report/`：建议将实验日志、图像、最终对比报告集中存放。  
- `results/`：各模型训练输出（权重、预测 JSON、评估指标）。

完成复现后，可按课程/任务要求撰写实验分析，比较精度、速度、实例可视化效果等。