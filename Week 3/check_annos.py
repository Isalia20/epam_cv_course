import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# Load the annotations from the JSON file
with open('Week 3/annotations/annotations.json', 'r') as f:
    annotations = json.load(f)

# Define the path to the image folder
image_folder = 'object_photos/'

# Iterate over the images in the annotations
for image in annotations['images']:
    # Open the image
    img = Image.open(os.path.join(image_folder, image['file_name']))

    # Create a figure
    fig, ax = plt.subplots(1)

    # Show the image
    ax.imshow(img)

    # Find the annotations for this image
    image_annotations = [a for a in annotations['annotations'] if a['image_id'] == image['id']]

    # Add rectangles for each annotation
    for ann in image_annotations:
        bbox = ann['bbox']
        x, y, w, h = bbox
        rect = Rectangle((x, y), w, h, fill=False, edgecolor='red')
        ax.add_patch(rect)

    # Show the plot
    plt.show()

with open("/Users/iraklisalia/Desktop/epam_cv_course/epam_cv_course/Week 3/val/annotations_val.json", 'r') as f:
    annotations = json.load(f)
