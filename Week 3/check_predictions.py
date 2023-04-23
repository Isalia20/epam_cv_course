from train import Model
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

def load_model(model_checkpoint_path):
    model = Model(1)
    model.model.eval()
    checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = F.to_tensor(image)
    return image

def plot_predictions(image, detections, scores, threshold=0.4):
    plt.imshow(image)
    ax = plt.gca()

    for i in range(len(detections)):
        if scores[i] > threshold:
            x0, y0, x1, y1 = detections[i][:4]
            w = x1 - x0
            h = y1 - y0
            rect = plt.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.show()


def main():
    model = load_model('checkpoints/test_v4/epoch=14-map=0.723.ckpt')
    image_path = 'Week 3/val/images/IMG_1884.png'
    image_tensor = preprocess_image(image_path)

    with torch.inference_mode():
        detections = model.model([image_tensor])

    image = cv2.imread(image_path)
    boxes = detections[0]["boxes"]
    scores = detections[0]["scores"]
    labels = detections[0]["labels"]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plot_predictions(image, boxes, scores, threshold=0.6)


if __name__ == '__main__':
    main()
