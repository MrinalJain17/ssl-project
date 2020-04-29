import os
import random
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from data_helper import LabeledDataset
from denoising_autoencoder import DenoisingAutoencoder
from helper import collate_fn

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

FEATURE_EXTRACTOR_PATH = "./saved_models/denoising_autoencoder.ckpt"
IMAGE_FOLDER = "../data"
ANNOTATION_CSV = "../data/annotation.csv"


class RoadMapNetwork(pl.LightningModule):
    """TODO
    """

    def __init__(self, hparams):
        super(RoadMapNetwork, self).__init__()
        self.hparams = hparams

        self.feature_extractor = DenoisingAutoencoder.load_from_checkpoint(
            FEATURE_EXTRACTOR_PATH
        )
        self.feature_extractor.freeze()
        self.classifier = nn.Sequential(
            nn.Conv2d(192, 64, 3, 1), nn.ReLU(), nn.Conv2d(64, 1, 3, 1)
        )

    def forward(self, x):
        stacked = self._stack_features(x)
        stacked = F.interpolate(stacked, size=804, mode="bilinear", align_corners=False)
        stacked = self.classifier(stacked)

        return torch.squeeze(stacked, 1)  # Output size -> (None, 800, 800)

    def _stack_features(self, x):
        temp = []
        for idx in range(6):
            features = self.feature_extractor(x[:, idx, :, :, :])
            temp.append(features)

        return torch.cat(temp, 1)

    def training_step(self, batch, batch_idx):
        sample, _, road_image = batch
        sample = torch.stack(sample)
        road_image = torch.stack(road_image).float()
        predicted_road_image = self.forward(sample)
        loss = F.binary_cross_entropy_with_logits(predicted_road_image, road_image)

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        sample, _, road_image = batch
        sample = torch.stack(sample)
        road_image = torch.stack(road_image).float()
        predicted_road_image = self.forward(sample)
        loss = F.binary_cross_entropy_with_logits(predicted_road_image, road_image)

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()

        logs = {"val_loss": val_loss_mean}
        return {"val_loss": val_loss_mean, "log": logs}

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.LEARNING_RATE,
            weight_decay=self.hparams.L2_PENALTY,
        )

    def prepare_data(self):
        # The scenes from 106 - 133 are labeled
        labeled_scene_index = np.arange(106, 134)

        # Keeping aside 8 scenes for validation
        # np.random.shuffle(labeled_scene_index)
        self._train_labeled_scene_index = labeled_scene_index[8:]
        self._valid_labeled_scene_index = labeled_scene_index[:8]
        self._static_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.54, 0.60, 0.63), (0.34, 0.34, 0.34)
                ),
            ]
        )

        self.labeled_trainset = LabeledDataset(
            image_folder=IMAGE_FOLDER,
            annotation_file=ANNOTATION_CSV,
            scene_index=self._train_labeled_scene_index,
            transform=self._static_transform,
            extra_info=False,
        )

        self.labeled_validset = LabeledDataset(
            image_folder=IMAGE_FOLDER,
            annotation_file=ANNOTATION_CSV,
            scene_index=self._valid_labeled_scene_index,
            transform=self._static_transform,
            extra_info=False,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.labeled_trainset,
            batch_size=self.hparams.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.labeled_validset,
            batch_size=self.hparams.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )


def main(args):
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=None,  # To prevent from using the slurm job id
        name="lightning_logs_road_map",
    )

    model = RoadMapNetwork(hparams=args)
    trainer = Trainer(gpus=1, max_epochs=args.EPOCHS, logger=logger)
    trainer.fit(model)
    trainer.save_checkpoint("saved_models/road_map.ckpt")


if __name__ == "__main__":
    parser = ArgumentParser()

    # parametrize the network
    parser.add_argument("--BATCH_SIZE", type=int, default=8)
    parser.add_argument("--EPOCHS", type=int, default=50)
    parser.add_argument("--LEARNING_RATE", type=float, default=1e-4)
    parser.add_argument("--L2_PENALTY", type=float, default=1e-5)

    args = parser.parse_args()

    # train
    main(args)
