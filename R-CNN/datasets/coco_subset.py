import argparse
import json
import os
import random
from collections import defaultdict
from shutil import copy2

TARGET_CLASSES = ["person", "dog", "cat", "car", "bicycle"]
SAMPLES_PER_CLASS = 200

def main(args):
    random.seed(args.seed)
    with open(args.ann_file, "r") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    target_cat_ids = [cid for cid, name in cat_id_to_name.items() if name in TARGET_CLASSES]

    img_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    selected_images = set()
    class_counts = {name: 0 for name in TARGET_CLASSES}

    for img in coco["images"]:
        anns = img_to_anns.get(img["id"], [])
        present_classes = set(cat_id_to_name[a["category_id"]] for a in anns if a["category_id"] in target_cat_ids)
        if not present_classes:
            continue
        keep = False
        for cls in present_classes:
            if class_counts[cls] < SAMPLES_PER_CLASS:
                keep = True
                break
        if keep:
            selected_images.add(img["id"])
            for cls in present_classes:
                if class_counts[cls] < SAMPLES_PER_CLASS:
                    class_counts[cls] += 1

        if all(count >= SAMPLES_PER_CLASS for count in class_counts.values()):
            break

    subset_images = [img for img in coco["images"] if img["id"] in selected_images]
    subset_annotations = [ann for ann in coco["annotations"] if ann["image_id"] in selected_images and ann["category_id"] in target_cat_ids]
    subset_categories = [cat for cat in coco["categories"] if cat["id"] in target_cat_ids]

    os.makedirs(args.out_dir, exist_ok=True)
    subset_json = {
        "images": subset_images,
        "annotations": subset_annotations,
        "categories": subset_categories
    }
    out_ann = os.path.join(args.out_dir, "instances_subset.json")
    with open(out_ann, "w") as f:
        json.dump(subset_json, f)

    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    for img in subset_images:
        src = os.path.join(args.image_dir, img["file_name"])
        dst = os.path.join(args.out_dir, "images", img["file_name"])
        copy2(src, dst)

    print(f"子集生成完成，共 {len(subset_images)} 张图像")
    print(f"各类别数量：{class_counts}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--out-dir", default="data/raw/coco_subset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)