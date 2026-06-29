import os
import torch
import torch.nn as nn
from typing import Any, Callable, Optional, Tuple
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import VisionDataset
from PIL import Image
import numpy as np
from PIL import Image


class PUFdataset(VisionDataset):
    def __init__(self, 
                 root_dir, 
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):        
        super().__init__(root=root_dir, transform=transform, target_transform=None)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.labels = []
        self.label_to_index = {}
        index = 0
        for category in os.listdir(root_dir):
            if category not in self.label_to_index:
                self.label_to_index[category] = index
                index += 1
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                for image_name in os.listdir(category_path):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(category_path, image_name))
                        self.labels.append(self.label_to_index[category])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            try:
                label = self.target_transform(label)
            except Exception as e:
                print(f"Error applying target_transform: {e}")
        return image, label





class PUFdataset_vis(VisionDataset):
 
    def __init__(self,
                 root_dir,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(root=root_dir, transform=transform, target_transform=None)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.labels = []
        self.label_to_index = {}
        # ---- visualization metadata (aligned with self.images) ----
        self.devices = []        # individual PUF device = folder name
        self.technologies = []   # 'Arbiter' / 'SRAM' / 'Other'

        index = 0
        for category in os.listdir(root_dir):
            if category not in self.label_to_index:
                self.label_to_index[category] = index
                index += 1
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                tech = self._parse_technology(category)
                for image_name in os.listdir(category_path):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(category_path, image_name))
                        self.labels.append(self.label_to_index[category])
                        self.devices.append(category)
                        self.technologies.append(tech)
 
        self.image_paths = self.images  # alias
 
    @staticmethod
    def _parse_technology(category):
        """'puf1_b' -> 'Arbiter' ; 'Board 0028' -> 'SRAM'.
        Adjust the prefix rules here if your naming differs."""
        low = category.lower()
        if low.startswith('puf'):
            return 'Arbiter'
        elif low.startswith('board'):
            return 'SRAM'
        return 'Other'
 
    def __len__(self):
        return len(self.images)
 
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            try:
                label = self.target_transform(label)
            except Exception as e:
                print(f"Error applying target_transform: {e}")
        return image, label