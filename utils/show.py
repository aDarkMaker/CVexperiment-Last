import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.visualization import visualize_models

visualize_models(
    coco_ann_file="data/raw/coco_subset/instances_subset.json",
    image_root="data/raw/coco_subset/images",
    model_results={
        "R-CNN": "results/rcnn/detections.json",
        "Fast R-CNN": "results/fast_rcnn/detections.json",
        "Mask R-CNN": "results/mask_rcnn/detections.json",
    },
    image_ids=[724, 87038], 
    score_thresh=0.6,
    show_gt=True,
    save_dir="report/figures/model_compare"
)