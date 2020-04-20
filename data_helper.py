import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

from helper import convert_map_to_lane_map, convert_map_to_road_map

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6
image_names = [
    "CAM_FRONT_LEFT.jpeg",
    "CAM_FRONT.jpeg",
    "CAM_FRONT_RIGHT.jpeg",
    "CAM_BACK_LEFT.jpeg",
    "CAM_BACK.jpeg",
    "CAM_BACK_RIGHT.jpeg",
]


# The dataset class for unlabeled data.
class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, scene_index, first_dim, transform):
        """
        Args:
            image_folder (string): the location of the image folder
            scene_index (list): a list of scene indices for the unlabeled data
            first_dim ({'sample', 'image'}):
                'sample' will return [batch_size, NUM_IMAGE_PER_SAMPLE, 3, H, W]
                'image' will return [batch_size, 3, H, W] and the index of the camera [0 - 5]
                    CAM_FRONT_LEFT: 0
                    CAM_FRONT: 1
                    CAM_FRONT_RIGHT: 2
                    CAM_BACK_LEFT: 3
                    CAM_BACK.jpeg: 4
                    CAM_BACK_RIGHT: 5
            transform (Transform): The function to process the image
        """

        self.image_folder = image_folder
        self.scene_index = scene_index
        self.transform = transform

        assert first_dim in ["sample", "image"]
        self.first_dim = first_dim

    def __len__(self):
        if self.first_dim == "sample":
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE
        elif self.first_dim == "image":
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE

    def __getitem__(self, index):
        if self.first_dim == "sample":
            scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
            sample_id = index % NUM_SAMPLE_PER_SCENE
            sample_path = os.path.join(
                self.image_folder, f"scene_{scene_id}", f"sample_{sample_id}"
            )

            images = []
            for image_name in image_names:
                image_path = os.path.join(sample_path, image_name)
                image = Image.open(image_path)
                images.append(self.transform(image))
            image_tensor = torch.stack(images)

            return image_tensor

        elif self.first_dim == "image":
            scene_id = self.scene_index[
                index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)
            ]
            sample_id = (
                index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)
            ) // NUM_IMAGE_PER_SAMPLE
            image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

            image_path = os.path.join(
                self.image_folder,
                f"scene_{scene_id}",
                f"sample_{sample_id}",
                image_name,
            )

            image = Image.open(image_path)

            return self.transform(image), index % NUM_IMAGE_PER_SAMPLE


# The dataset class for labeled data.
class LabeledDataset(torch.utils.data.Dataset):
    def __init__(
        self, image_folder, annotation_file, scene_index, transform, extra_info=True
    ):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """

        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info

    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):
        scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
        sample_id = index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(
            self.image_folder, f"scene_{scene_id}", f"sample_{sample_id}"
        )

        images = []
        for image_name in image_names:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path)
            images.append(self.transform(image))
        image_tensor = torch.stack(images)

        data_entries = self.annotation_dataframe[
            (self.annotation_dataframe["scene"] == scene_id)
            & (self.annotation_dataframe["sample"] == sample_id)
        ]
        corners = data_entries[
            ["fl_x", "fr_x", "bl_x", "br_x", "fl_y", "fr_y", "bl_y", "br_y"]
        ].to_numpy()
        categories = data_entries.category_id.to_numpy()

        ego_path = os.path.join(sample_path, "ego.png")
        ego_image = Image.open(ego_path)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = convert_map_to_road_map(ego_image)

        target = {}
        target["bounding_box"] = torch.as_tensor(corners).view(-1, 2, 4)
        target["category"] = torch.as_tensor(categories)

        if self.extra_info:
            actions = data_entries.action_id.to_numpy()
            # You can change the binary_lane to False to get a lane with
            lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)

            extra = {}
            extra["action"] = torch.as_tensor(actions)
            extra["ego_image"] = ego_image
            extra["lane_image"] = lane_image

            return image_tensor, target, road_image, extra

        else:
            return image_tensor, target, road_image


# The dataset class for unlabeled data.
class CustomUnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, scene_index):
        """
        Args:
            image_folder (string): the location of the image folder
            scene_index (list): a list of scene indices for the unlabeled data
        """

        self.image_folder = image_folder
        self.scene_index = scene_index
        self.first_dim = "image"
        self.permutations = self.__retrive_permutations(1000)
        self.__image_transformer = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256, Image.BILINEAR),
                torchvision.transforms.CenterCrop(255),
            ]
        )
        self.__augment_tile = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(64),
                torchvision.transforms.Resize((75, 75), Image.BILINEAR),
                torchvision.transforms.Lambda(rgb_jittering),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE

    def __getitem__(self, index):
        scene_id = self.scene_index[
            index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)
        ]
        sample_id = (
            index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)
        ) // NUM_IMAGE_PER_SAMPLE
        image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

        image_path = os.path.join(
            self.image_folder, f"scene_{scene_id}", f"sample_{sample_id}", image_name,
        )

        img = Image.open(image_path).convert("RGB")
        if np.random.rand() < 0.30:
            img = img.convert("LA").convert("RGB")

        if img.size[0] != 255:
            img = self.__image_transformer(img)

        s = float(img.size[0]) / 3
        a = s / 2
        tiles = [None] * 9
        for n in range(9):
            i = n / 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            tile = self.__augment_tile(tile)
            # Normalize the patches indipendently to avoid low level features shortcut
            m, s = (
                tile.view(3, -1).mean(dim=1).numpy(),
                tile.view(3, -1).std(dim=1).numpy(),
            )
            s[s == 0] = 1
            norm = torchvision.transforms.Normalize(mean=m.tolist(), std=s.tolist())
            tile = norm(tile)
            tiles[n] = tile

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data = torch.stack(data, 0)

        return data, int(order), tiles

    def __retrive_permutations(self, classes):
        all_perm = np.load("permutations_1000.npy")
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm


def rgb_jittering(im):
    im = np.array(im, "int32")
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype("uint8")
