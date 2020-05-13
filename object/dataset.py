import os
from PIL import Image

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset

from helper import convert_map_to_lane_map, convert_map_to_road_map

from utils import transform

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6
image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
    ]
    
  #Do this as a temporary trick. Make this work later.  
#image_names = ['CAM_FRONT_LEFT.jpeg']

# The dataset class for labeled data.
class LabeledDataset(Dataset):    

    def __init__(self, image_folder, annotation_file, scene_index,extra_info, split):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data 
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}
        
        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.extra_info = extra_info
    
    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):
        #print(index)
        scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
        sample_id = index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}') 

        data_entries = self.annotation_dataframe[(self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        #print(data_entries)
        
        #corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']].to_numpy()
        data_entries['min_x']= data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x']].min(axis=1)
        data_entries['min_y']= data_entries[['fl_y', 'fr_y','bl_y', 'br_y']].min(axis=1)
        data_entries['max_x']= data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x']].max(axis=1)
        data_entries['max_y']= data_entries[['fl_y', 'fr_y','bl_y', 'br_y']].max(axis=1)
        #print(data_entries)
        
        corners = data_entries[['min_x', 'min_y','max_x','max_y']].to_numpy()
        #print(corners)
        categories = data_entries.category_id.to_numpy()
        
        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = convert_map_to_road_map(ego_image)
        
        target = {}
        #target['bounding_box'] = torch.as_tensor(corners).view(-1, 2, 4)
        #target['bounding_box'] = torch.as_tensor(corners).view(-1, 2, 2) #<--------------Look into this later
        target['bounding_box'] = torch.as_tensor(corners).view(-1, 4)
        target['category'] = torch.as_tensor(categories)
        
        
        images = []
        for image_name in image_names:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path)
            new_image, new_box,new_label = transform(image,target,split=self.split)
            images.append(new_image)
        #<-------------------------Uncomment after we use all data!  
        image_tensor = torch.stack(images)
        target['bounding_box'] = new_box
        target['category'] = new_label

        if self.extra_info:
            actions = data_entries.action_id.to_numpy()
            # You can change the binary_lane to False to get a lane with 
            lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)
            
            extra = {}
            extra['action'] = torch.as_tensor(actions)
            extra['ego_image'] = ego_image
            extra['lane_image'] = lane_image

            return image_tensor, target, road_image, extra
        
        else:
            image_tensor = image_tensor.squeeze(0)
            boxes = target['bounding_box'].type(torch.FloatTensor)  # (n_objects, 4)
            labels = target['category'].type(torch.LongTensor)
            
            #return image_tensor,target['bounding_box'],target['category']
            return image_tensor, boxes,labels
            
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            #difficulties.append(b[3])

        images = torch.stack(images, dim=0)
        return images, boxes, labels
        
class CorruptedUnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, scene_index, transform, noise=None):
        """
        Args:
            image_folder (string): the location of the image folder
            scene_index (list): a list of scene indices for the unlabeled data
            transform (Transform): The function to process the image
        """

        self.image_folder = image_folder
        self.scene_index = scene_index
        self.transform = transform
        self.noise = noise
        self.first_dim = "image"

    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE

    def __getitem__(self, index):
        scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
        sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
        image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

        image_path = os.path.join(self.image_folder, f"scene_{scene_id}", f"sample_{sample_id}", image_name)

        image = Image.open(image_path)

        target_ = self.transform(image)
        if self.noise is not None:
            input_ = self.noise(target_)
        else:
            input_ = target_.clone()

        return input_, target_


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"

    