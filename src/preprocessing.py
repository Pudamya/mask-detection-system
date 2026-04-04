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
        self.summarize_dataset(labels)
        return image_paths, labels

    def summarize_dataset(self, labels):
        class_counts = Counter(labels)
        total = len(labels)

        print("\nDataset Summary")
        for idx, class_name in enumerate(self.classes):
            count = class_counts.get(idx, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"{class_name:15s}: {count:4d} images ({percentage:.2f}%)")
        print(f"{'total':15s}: {total:4d} images")

    def split_dataset(self, image_paths, labels):
        X_train, X_temp, y_train, y_temp = train_test_split(
            image_paths,
            labels,
            test_size=0.30,
            random_state=self.seed,
            stratify=labels
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=self.seed,
            stratify=y_temp
        )

        print("\nDataset Split")
        print(f"Train      : {len(X_train)}")
        print(f"Validation : {len(X_val)}")
        print(f"Test       : {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_transforms(self):
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.15,
                hue=0.02
            ),
            transforms.RandomPerspective(distortion_scale=0.10, p=0.20),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return train_transform, val_test_transform

    def get_dataloaders(self):
        image_paths, labels = self.import_dataset()
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(image_paths, labels)
        train_transform, val_test_transform = self.get_transforms()

        train_dataset = MaskDataset(X_train, y_train, train_transform)
        val_dataset = MaskDataset(X_val, y_val, val_test_transform)
        test_dataset = MaskDataset(X_test, y_test, val_test_transform)

        pin_memory = torch.cuda.is_available()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory
        )

        return train_loader, val_loader, test_loader

    def visualize_samples(self, image_paths, labels, n=8):
        fig, axes = plt.subplots(2, n // 2, figsize=(15, 6))
        axes = axes.flatten()

        indices = np.random.choice(len(image_paths), n, replace=False)

        for i, idx in enumerate(indices):
            img = cv2.imread(image_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            axes[i].imshow(img)
            axes[i].set_title(self.classes[labels[idx]])
            axes[i].axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'sample_images.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Sample images saved to {save_path}")


class MaskDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label