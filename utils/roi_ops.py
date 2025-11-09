import numpy as np  # pyright: ignore[reportMissingImports]
from PIL import Image  # pyright: ignore[reportMissingImports]


def _sanitize_single_box(box, width: int, height: int):
    x1 = float(min(box[0], box[2]))
    x2 = float(max(box[0], box[2]))
    y1 = float(min(box[1], box[3]))
    y2 = float(max(box[1], box[3]))

    x1 = float(np.clip(x1, 0, max(width - 1, 0)))
    x2 = float(np.clip(x2, 0, max(width - 1, 0)))
    y1 = float(np.clip(y1, 0, max(height - 1, 0)))
    y2 = float(np.clip(y2, 0, max(height - 1, 0)))

    if x2 <= x1:
        if width <= 1:
            x1, x2 = 0.0, 1.0
        elif x1 >= width - 1:
            x1 = max(0.0, width - 2.0)
            x2 = float(width - 1)
        else:
            x2 = min(float(width - 1), x1 + 1.0)

    if y2 <= y1:
        if height <= 1:
            y1, y2 = 0.0, 1.0
        elif y1 >= height - 1:
            y1 = max(0.0, height - 2.0)
            y2 = float(height - 1)
        else:
            y2 = min(float(height - 1), y1 + 1.0)

    return x1, y1, x2, y2


def crop_and_resize(image: Image.Image, box, size=(227, 227)):
    width, height = image.size
    x1, y1, x2, y2 = _sanitize_single_box(box, width, height)

    left = int(np.floor(x1))
    upper = int(np.floor(y1))
    right = int(np.ceil(x2))
    lower = int(np.ceil(y2))

    # 再次确保右下角比左上角大
    if right <= left:
        right = min(width, left + 1)
    if lower <= upper:
        lower = min(height, upper + 1)

    crop = image.crop((left, upper, right, lower))
    return crop.resize(size, Image.BILINEAR)


def compute_iou(box, boxes):
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1 + 1)
    inter_h = np.maximum(0.0, y2 - y1 + 1)
    inter = inter_w * inter_h

    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    boxes_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    union = box_area + boxes_area - inter
    union = np.maximum(union, 1e-6)
    return inter / union
