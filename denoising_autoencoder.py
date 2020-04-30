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

from data_helper import AddGaussianNoise, CorruptedUnlabeledDataset

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

IMAGE_FOLDER = "../data"


class DenoisingAutoencoder(pl.LightningModule):
    """A denoising autoencoder module

    The trained autoencoder is later used as a feature extractor for the
    labeled data.

    Parameters
    ----------

    hparams : argparse.Namespace
        A namespace containing the required hyperparameters. In particular, the
        code expects `hparams` to have the following keys:
        1. BATCH_SIZE
        2. LEARNING_RATE
        3. L2_PENALTY
        4. EPOCHS
    """

    def __init__(self, hparams):
        super(DenoisingAutoencoder, self).__init__()
        self.hparams = hparams
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 3, 2),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 3, 2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, 2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, 2),
            nn.LeakyReLU(),
        )  # Output size -> (None, 32, 13, 13)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 3, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 128, 3, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 256, 3, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 3, 3, 2, output_padding=1),
            nn.Tanh(),
        )  # Output size -> (None, 3, 224, 224)

    def forward(self, x):
        features = self.encoder(x)
        return features

    def training_step(self, batch, batch_idx):
        input_, target_ = batch
        features = self.forward(input_)
        reconstruction = self.decoder(features)
        loss = F.mse_loss(reconstruction, target_)

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        input_, target_ = batch
        reconstruction = self.decoder(self.forward(input_))
        loss = F.mse_loss(reconstruction, target_)

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
        # The first 106 scenes are unlabeled
        unlabeled_scene_index = np.arange(106)

        # Keeping aside 6 scenes for validation
        # np.random.shuffle(unlabeled_scene_index)
        self._train_unlabeled_scene_index = unlabeled_scene_index[:100]
        self._valid_unlabeled_scene_index = unlabeled_scene_index[100:]
        self._static_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.54, 0.60, 0.63), (0.34, 0.34, 0.34)
                ),
            ]
        )
        self._noise = AddGaussianNoise(mean=0.0, std=0.25)

        self.unlabeled_trainset = CorruptedUnlabeledDataset(
            image_folder=IMAGE_FOLDER,
            scene_index=self._train_unlabeled_scene_index,
            transform=self._static_transform,
            noise=self._noise,
        )

        self.unlabeled_validset = CorruptedUnlabeledDataset(
            image_folder=IMAGE_FOLDER,
            scene_index=self._valid_unlabeled_scene_index,
            transform=self._static_transform,
            noise=self._noise,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.unlabeled_trainset,
            batch_size=self.hparams.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.unlabeled_validset,
            batch_size=self.hparams.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
        )


def main(args):
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=None,  # To prevent from using the slurm job id
        name="lightning_logs",
    )

    model = DenoisingAutoencoder(hparams=args)
    trainer = Trainer(gpus=1, max_epochs=args.EPOCHS, logger=logger)
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()

    # parametrize the network
    parser.add_argument("--BATCH_SIZE", type=int, default=128)
    parser.add_argument("--EPOCHS", type=int, default=50)
    parser.add_argument("--LEARNING_RATE", type=float, default=1e-3)
    parser.add_argument("--L2_PENALTY", type=float, default=1e-5)

    args = parser.parse_args()

    # train
    main(args)
