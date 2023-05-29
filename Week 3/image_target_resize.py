import torch
from torchvision.transforms import RandomResizedCrop, Resize


class ImageTargetResize:
    def __init__(self, resized_image_height, resized_image_width):
        self.resize = Resize(size=(resized_image_height, resized_image_width))  # 1020, 1980
        self.resized_image_height = resized_image_height
        self.resized_image_width = resized_image_width

    def __call__(self, image, target):
        image_height, image_width = image.shape[1], image.shape[2]
        height_ratio = self.resized_image_height / image_height
        width_ratio = self.resized_image_width / image_width
        image = self.resize(image)
        ys = target[:, [1, 3]] * height_ratio
        xs = target[:, [0, 2]] * width_ratio
        targets = torch.vstack([xs[:, 0], ys[:, 0], xs[:, 1], ys[:, 1]])
        targets = targets.swapaxes(1, 0)
        return image, targets
