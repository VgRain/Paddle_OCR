import os
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
import torch


class YOLOv8_DualAugDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.images = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
        self.img_size = img_size

        # Original transform
        self.t_original = A.Compose([
            A.Resize(img_size, img_size)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        # Blurred transform
        self.t_blur = A.Compose([
            A.Downscale(scale_min=0.5, scale_max=0.8, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.Resize(img_size, img_size)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        # 90-degree rotation
        self.t_rotate = A.Compose([
            A.Rotate(limit=(90, 90), p=1.0),
            A.Resize(img_size, img_size)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        # Blur + Rotate
        self.t_blur_rotate = A.Compose([
            A.Rotate(limit=(90, 90), p=1.0),
            A.Downscale(scale_min=0.5, scale_max=0.8, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.Resize(img_size, img_size)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __len__(self):
        return len(self.images)

    def load_label(self, label_path):
        boxes, cls_ids = [], []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    c, x, y, w, h = map(float, line.split())
                    cls_ids.append(int(c))
                    boxes.append([x, y, w, h])
        return boxes, cls_ids

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.rsplit(".", 1)[0] + ".txt")

        # Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load Labels
        boxes, cls_ids = self.load_label(label_path)

        # Generate 4 versions
        o = self.t_original(image=image, bboxes=boxes, class_labels=cls_ids)
        b = self.t_blur(image=image, bboxes=boxes, class_labels=cls_ids)
        r = self.t_rotate(image=image, bboxes=boxes, class_labels=cls_ids)
        br = self.t_blur_rotate(image=image, bboxes=boxes, class_labels=cls_ids)

        # Convert to YOLOv8 format
        def convert(t):
            img = torch.tensor(t["image"]).permute(2, 0, 1) / 255.0
            labels = torch.tensor([[cls] + list(bb) for cls, bb in zip(t["class_labels"], t["bboxes"])])
            return img, labels

        img_o, lab_o = convert(o)
        img_b, lab_b = convert(b)
        img_r, lab_r = convert(r)
        img_br, lab_br = convert(br)

        # Return 4x images + labels
        return (
            torch.stack([img_o, img_b, img_r, img_br]),   # shape: [4, 3, 640, 640]
            [lab_o, lab_b, lab_r, lab_br]
        )
