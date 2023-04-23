import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor
from image_target_resize import ImageTargetResize
import json


class ObjectDetectionDataset(Dataset):
    def __init__(self, image_dir, annotations_file_path, split="train"):
        assert split in ("train", "validation"), "Invalid split type. Use 'train' or 'validation'."
        self.split = split

        self.coco_annotations = self.read_annotations(annotations_file_path)
        self.annotations = self.coco_annotations["annotations"]
        self.image_paths = self.get_image_paths(image_dir, self.coco_annotations)
        self.image_target_resize = ImageTargetResize(1824, 2736)
        self.category_label_map = {1: "Plush", 2: "Pillow"}

    def read_annotations(self, annotations_path):
        # Load the annotations from the JSON file
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        return annotations

    def get_image_paths(self, image_dir, annotations):
        image_paths = []
        for image_details in annotations["images"]:
            image_path = os.path.join(image_dir, image_details["file_name"])
            image_paths.append(image_path)
        return image_paths

    def get_item(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image = to_tensor(image)

        bbox = torch.tensor(self.annotations[idx]["bbox"], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.annotations[idx]["category_id"], dtype=torch.long).unsqueeze(0)
        # img, bbox = self.image_target_resize(image, bbox)
        x_min, y_min, width, height = torch.unbind(bbox, 1)
        x_max = x_min + width
        y_max = y_min + height
        bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32).unsqueeze(0)
        target = {"boxes": bbox,
                  "labels": label
                  }
        return image, target

    def get_item_normal_augs(self, idx):
        # Add any other augs if you want here
        return self.get_item(idx)

    def get_item_validation(self, idx):
        return self.get_item(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Mosaic
        if self.split == "train":
            return self.get_item_normal_augs(idx)
        else:
            return self.get_item_validation(idx)
