import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.block(x)
    
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights    

# Custom CNN architecture for mask classification
class ModelDevelopment(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super(ModelDevelopment, self).__init__()

        self.block1 = ConvBlock(3, 32, dropout=0.10)
        self.block2 = ConvBlock(32, 64, dropout=0.15)
        self.block3 = ConvBlock(64, 128, dropout=0.20)
        self.block4 = ConvBlock(128, 256, dropout=0.25)

        self.attention = SEBlock(256)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
    )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x

    def get_architecture_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'architecture': str(self)
        }

# Handles the full training loop, validation, and saving the best model.
class ModelTrainer:
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device

        # CrossEntropyLoss = Softmax + NLLLoss
        self.criterion = nn.CrossEntropyLoss()

        # Adam optimizer: adaptive learning rates per parameter
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                                    weight_decay=1e-4)  # L2 regularization

        # Reduce LR by factor 0.5 if val_loss doesn't improve for 3 epochs
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                   patience=3, factor=0.5)

        self.history = {'train_loss': [], 'val_loss': [],
                        'train_acc': [],  'val_acc': []}
        self.best_val_loss = float('inf')

    def train_one_epoch(self, train_loader):
        self.model.train()  # Enable dropout & batch norm training mode
        total_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()        
            outputs = self.model(images)      
            loss = self.criterion(outputs, labels)  
            loss.backward()                   
            self.optimizer.step()             

            total_loss += loss.item()
            _, predicted = outputs.max(1)     
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        return total_loss / len(train_loader), 100. * correct / total

    # Evaluate on validation set
    def validate(self, val_loader):
        self.model.eval()  # Disable dropout, use running stats for BN
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():  
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return total_loss / len(val_loader), 100. * correct / total

    def train(self, train_loader, val_loader, epochs=30, save_path='models/best_model.pth'):
        print(f"\nTraining on: {self.device}")
        print(f"Epochs: {epochs}\n")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc     = self.validate(val_loader)

            # Step the scheduler based on validation loss
            self.scheduler.step(val_loss)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            print(f"Epoch [{epoch:3d}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Save best model (based on validation loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f" Best model saved (val_loss: {val_loss:.4f})")

        print("\nTraining complete!")
        return self.history

    def plot_history(self, save_dir='results'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(self.history['train_loss']) + 1)

        ax1.plot(epochs, self.history['train_loss'], 'b-o', label='Train Loss', markersize=3)
        ax1.plot(epochs, self.history['val_loss'],   'r-o', label='Val Loss',   markersize=3)
        ax1.set_title('Loss per Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, self.history['train_acc'], 'b-o', label='Train Acc', markersize=3)
        ax2.plot(epochs, self.history['val_acc'],   'r-o', label='Val Acc',   markersize=3)
        ax2.set_title('Accuracy per Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
        plt.show()
        print("Training curves saved.")