目标检测是计算机视觉的基本任务之一，旨在从一副图像上用矩形框把一些物体框定出来。在目标检测领域有几个典型网络，分别是R-CNN、Fast R-CNN、Faster R-CNN以及Mask R-CNN。请你阅读上述四个网络对应的原始论文，并要求：
1.从上述方法中任选三种进行复现（也可运行开源代码），得到目标检测的数值和可视化结果；
2.对比三种方法的目标检测结果，分析各自的优缺点；

[Datasets]
- COCO 2017
  - mkdir -p data/raw/coco2017
  - cd data/raw/coco2017
  - ``` wget http://images.cocodataset.org/zips/val2017.zip ```
  - ``` unzip val2017.zip ```
  - ``` wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip ```
  - ``` unzip annotations_trainval2017.zip ```
- R-CNN
  - ``` python R-CNN/datasets/coco_subset.py \--image-dir data/raw/coco2017/val2017 \--ann-file data/raw/coco2017/annotations/instances_val2017.json \--out-dir data/raw/coco_subset ```
  - ``` python R-CNN/datasets/generate_proposals.py \--image-dir data/raw/coco_subset/images \--ann-file data/raw/coco_subset/instances_subset.json \--out-dir data/proposals/coco_subset ```
  - Train
    - ``` python R-CNN/training/extract_features.py \--image-dir data/raw/coco_subset/images \--ann-file data/raw/coco_subset/instances_subset.json \--proposal-dir data/proposals/coco_subset \--feature-ckpt results/rcnn/feature.pth \--out-dir data/cache/coco_subset \--max-proposals 800 ```

[Ultralytics]
- 框架