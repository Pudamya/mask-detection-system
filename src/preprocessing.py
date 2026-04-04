# Importing libraries
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import random
from collections import Counter

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Handles all data loading, splitting, and transformation.
class BasicPreprocessing:
    def __init__(self, data_dir='data', img_size=128, batch_size=32, seed=42):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.seed = seed
        self.classes = ['with_mask', 'without_mask']
        self.results_dir = 'results'

        os.makedirs(self.results_dir, exist_ok=True)
        set_seed(self.seed)

    def import_dataset(self):
        image_paths = []
        labels = []

        for label_idx, class_name in enumerate(self.classes):
            class_folder = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_folder):
                raise FileNotFoundError(f"Folder not found: {class_folder}")

            for img_file in os.listdir(class_folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_folder, img_file))
                    labels.append(label_idx)

        print(f"Total images found: {len(image_paths)}")
        print(f"  With mask:    {labels.count(0)}")
        print(f"  Without mask: {labels.count(1)}")

        return image_paths, labels

    def split_dataset(self, image_paths, labels):
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            image_paths, labels,
            test_size=0.3,
            random_state=42,
            stratify=labels
        )

        # Second split: val vs test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            random_state=42,
            stratify=y_temp
        )

        print(f"\nDataset Split:")
        print(f"  Train:      {len(X_train)} images")
        print(f"  Validation: {len(X_val)} images")
        print(f"  Test:       {len(X_test)} images")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_transforms(self):
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),       # 50% chance flip
            transforms.RandomRotation(degrees=15),         # rotate up to 15°
            transforms.ColorJitter(brightness=0.3,
                                   contrast=0.3,
                                   saturation=0.2),        # vary lighting
            transforms.ToTensor(),                          # converts to [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) # standardize
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        return train_transform, val_test_transform

    def get_dataloaders(self):
        image_paths, labels = self.import_dataset()
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(
            image_paths, labels
        )
        train_transform, val_test_transform = self.get_transforms()

        train_dataset = MaskDataset(X_train, y_train, train_transform)
        val_dataset   = MaskDataset(X_val,   y_val,   val_test_transform)
        test_dataset  = MaskDataset(X_test,  y_test,  val_test_transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_dataset,   batch_size=self.batch_size,
                                  shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_dataset,  batch_size=self.batch_size,
                                  shuffle=False, num_workers=0)

        return train_loader, val_loader, test_loader

    def visualize_samples(self, image_paths, labels, n=8):
        """Plots sample images from the dataset."""
        fig, axes = plt.subplots(2, n // 2, figsize=(15, 6))
        axes = axes.flatten()

        indices = np.random.choice(len(image_paths), n, replace=False)
        for i, idx in enumerate(indices):
            img = cv2.imread(image_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            axes[i].imshow(img)
            axes[i].set_title(self.classes[labels[idx]])
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig('results/sample_images.png')
        plt.show()
        print("Sample images saved to results/sample_images.png")


class MaskDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image using PIL
        image = Image.open(self.image_paths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label