import argparse
import os
import pickle
import sys
from typing import List

import numpy as np  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader  # pyright: ignore[reportMissingImports]

# 将项目根目录加入搜索路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from datasets.coco_dataset import COCOSubsetDataset  # noqa: E402
from models.feature_extractor import AlexNetExtractor  # noqa: E402
from utils.roi_ops import crop_and_resize  # noqa: E402


def preprocess_crops(image, proposals, transform) -> List[torch.Tensor]:
    crops = []
    for box in proposals:
        crop = crop_and_resize(image, box)
        crops.append(transform(crop))
    return crops


def main(args):
    dataset = COCOSubsetDataset(
        image_dir=args.image_dir,
        ann_file=args.ann_file,
        proposal_dir=args.proposal_dir,
        max_proposals=args.max_proposals,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_model = AlexNetExtractor(pretrained=False)
    feature_model.load_state_dict(torch.load(args.feature_ckpt, map_location=device))
    feature_model = feature_model.to(device)
    feature_model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            sample = batch[0]
            file_name = sample["file_name"]
            image = sample["image"]
            proposals = sample["proposals"]

            crops = preprocess_crops(image, proposals, dataset.transform)
            if not crops:
                continue
            crop_tensor = torch.stack(crops).to(device)
            features = feature_model(crop_tensor).cpu().numpy()

            base = os.path.splitext(file_name)[0]
            np.save(os.path.join(args.out_dir, f"{base}_features.npy"), features)
            with open(os.path.join(args.out_dir, f"{base}_proposals.pkl"), "wb") as fp:
                pickle.dump(proposals, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--proposal-dir", required=True)
    parser.add_argument("--feature-ckpt", required=True)
    parser.add_argument("--out-dir", default="data/cache/coco_subset")
    parser.add_argument("--max-proposals", type=int, default=800)
    args = parser.parse_args()
    main(args)