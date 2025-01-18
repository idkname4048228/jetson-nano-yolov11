import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image

class ImageLoader:
    def __init__(self, dataset_dir, batch_size=32, random_seed=42):
        train_ratio, val_ratio = 0.7, 0.15

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.transform = transforms.Compose([
            transforms.Resize((600, 600)),
            transforms.ToTensor()
        ])

        self.classes = ['crease', 'dusty_break', 'dusty_inside', 'tin', 'OK']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.dataset = self.create_custom_dataset(dataset_dir)

        n_total = len(self.dataset)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        self.train_set, self.val_set, self.test_set = random_split(self.dataset, [n_train, n_val, n_test])

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False)

    def create_custom_dataset(self, dataset_dir):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(dataset_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if os.path.isfile(img_path) and self.is_valid_file(img_path):
                        samples.append((img_path, self.class_to_idx[class_name]))
        return CustomImageDataset(samples, transform=self.transform)

    @staticmethod
    def is_valid_file(filepath):
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        return any(filepath.endswith(ext) for ext in valid_extensions)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target
