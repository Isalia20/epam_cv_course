import pytorch_lightning as pl
from dataset import ObjectDetectionDataset
# from collate_batch import make_data_loader
import torch

image_dir_train = "Week 3/train/images"
image_dir_val = "Week 3/val/images"
annot_file_train ="Week 3/train/annotations_train.json"
annot_file_val = "Week 3/val/annotations_val.json"


class BatchCollator:
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """
    def __init__(self):
        super().__init__()

    def __call__(self, batch):
        images = [item[0] for item in batch]
        target_dicts = [item[1] for item in batch]
        return images, target_dicts


def make_data_loader(dataset, phase, batch_size, num_workers):

    collator = BatchCollator()

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
        persistent_workers=False, # phase != "train",
        shuffle=phase == "train",
    )
    return data_loader


class ObjectDetectionDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.data_train = ObjectDetectionDataset(image_dir_train, annot_file_train, split="train")
        self.data_val = ObjectDetectionDataset(image_dir_val, annot_file_val, split="validation")

    def train_dataloader(self):
        return make_data_loader(self.data_train, "train", batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return make_data_loader(self.data_val, "val", batch_size=self.batch_size, num_workers=2)
