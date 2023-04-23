import json
import os
import shutil
import random
from collections import defaultdict

# Set the path to your directories
root = "Week 3"
image_dir = 'Week 3/object_photos'
annotation_dir = 'Week 3/annotations'
annotation_file = 'Week 3/annotations/annotations.json'

# Load the COCO-style annotations
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Create train and val directories if they don't exist
image_train_dest = os.path.join(root, 'train/images')
image_val_dest = os.path.join(root, 'val/images')

os.makedirs(image_train_dest, exist_ok=True)
os.makedirs(image_val_dest, exist_ok=True)

# Set the split ratio for train and validation
train_split = 0.8

# Group image IDs by class
class_to_image_ids = defaultdict(list)

for ann in annotations['annotations']:
    image_id = ann['image_id']
    category_id = ann['category_id']
    class_to_image_ids[category_id].append(image_id)

# Shuffle the images and split them into train and validation sets
train_image_ids = set()
val_image_ids = set()

for category_id, image_ids in class_to_image_ids.items():
    random.shuffle(image_ids)
    num_train = int(len(image_ids) * train_split)
    train_image_ids.update(image_ids[:num_train])
    val_image_ids.update(image_ids[num_train:])

# Copy images and filter annotations for train and validation sets
train_annotations = {key: [] for key in annotations.keys()}
val_annotations = {key: [] for key in annotations.keys()}

for image in annotations['images']:
    image_id = image['id']
    image_path = os.path.join(image_dir, image['file_name'])

    if image_id in train_image_ids:
        train_annotations['images'].append(image)
        shutil.copy(image_path, image_train_dest)
    else:
        val_annotations['images'].append(image)
        shutil.copy(image_path, image_val_dest)

for ann in annotations['annotations']:
    image_id = ann['image_id']

    if image_id in train_image_ids:
        train_annotations['annotations'].append(ann)
    else:
        val_annotations['annotations'].append(ann)

for key in annotations.keys():
    if key not in ['images', 'annotations']:
        train_annotations[key] = annotations[key]
        val_annotations[key] = annotations[key]

# Save the split annotations in COCO format
with open(os.path.join(root, 'train/annotations_train.json'), 'w') as f:
    json.dump(train_annotations, f)

with open(os.path.join(root, 'val/annotations_val.json'), 'w') as f:
    json.dump(val_annotations, f)
