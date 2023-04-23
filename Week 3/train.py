import os
import torch

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from data_module import ObjectDetectionDataModule
from torch.optim.lr_scheduler import _LRScheduler
from math import cos, pi
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models import ResNet50_Weights
from pytorch_lightning.utilities.seed import seed_everything

# Set the seed for everything
seed_value = 42
seed_everything(seed_value)


class CosineAnnealingWarmup(_LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            min_lr: float = 0.00001,
            warmup_steps: int = 0,
            decay_steps: int = 300,
            last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.last_epoch = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            mult = self.last_epoch / self.warmup_steps
        else:
            mult = 0.5 * (
                    1 + cos(pi * (self.last_epoch - self.warmup_steps) / self.decay_steps)
            )
        return [
            self.min_lr + (base_lr - self.min_lr) * mult for base_lr in self.base_lrs
        ]


class Model(LightningModule):
    def __init__(self, batch_size) -> None:
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn_v2(weights_backbone=ResNet50_Weights.DEFAULT, num_classes=3)
        self.model = self.model.train()
        self.batch_size = batch_size
        self.metric = MeanAveragePrecision(num_classes=3)

    def forward(self, images, targets):
        loss_dict = self.model.forward(images, targets)
        return loss_dict

    def on_validation_epoch_start(self) -> None:
        self.model.train()

    def training_step(self, batch, _):
        images, targets = batch
        loss_dict = self.forward(images, targets)
        total_loss = sum(loss_dict[key] for key in loss_dict)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.00005)
        scheduler = CosineAnnealingWarmup(optimizer)
        return [optimizer], [scheduler]

    def validation_step(self, batch, _):
        images, targets = batch
        loss_dict = self.forward(images, targets)

        total_loss = sum(loss_dict[key] for key in loss_dict)

        self.log("val_loss", total_loss, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.model.eval()
        # Get the model predictions
        predictions = self.model(images)
        self.model.train()
        # Convert the predictions and targets into the format expected by the metric
        preds = []
        tgts = []
        for pred, tgt in zip(predictions, targets):
            pred_boxes = pred['boxes'].tolist()
            pred_labels = pred['labels'].tolist()
            pred_scores = pred['scores'].tolist()
            preds.append({"boxes": torch.tensor(pred_boxes),
                          "labels": torch.tensor(pred_labels),
                          "scores": torch.tensor(pred_scores)})

            tgt_boxes = tgt['boxes'].tolist()
            tgt_labels = tgt['labels'].tolist()
            tgts.append({"boxes": torch.tensor(tgt_boxes),
                         "labels": torch.tensor(tgt_labels)})

        # Update the metric
        if len(preds) != 0:
            print("Calculating metric")
            self.metric.update(preds, tgts)
            map_value = self.metric.compute()
            self.log_dict(map_value, on_epoch=True)
            # Reset the metric for the next epoch
            self.metric.reset()

        return {"loss": total_loss}

    def on_validation_epoch_end(self) -> None:
        self.model.train()


def main(model_name, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare train dataset and validation dataset
    quad_data_module = ObjectDetectionDataModule(batch_size=batch_size)

    fcos_model = Model(batch_size=batch_size)
    fcos_model = fcos_model.to(device)

    output_folder = f"checkpoints/{model_name}"
    if not os.path.exists(output_folder + "/"):
        os.makedirs(output_folder + "/")

    callbacks = [
        ModelCheckpoint(
            dirpath=output_folder + "/",
            filename="{epoch}-{map:.3f}",
            monitor="map",
            mode="max",
            every_n_epochs=1,
            save_top_k=3,
        ),
    ]
    trainer = Trainer(
        accelerator="cpu",
        max_epochs=20,
        # max_steps=20,
        precision=32,  # Can't do 16 on cpu due to some error
        benchmark=True,
        callbacks=callbacks,
    )
    trainer.fit(
        fcos_model,
        quad_data_module,
    )


if __name__ == "__main__":
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "fasterrcnn_model"
    main(model_name, batch_size=4)
