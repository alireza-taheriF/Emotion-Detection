# src/train_model.py

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision import transforms
from model import EmotionCNN

# Paths and hyperparameters
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'fer2013')
train_csv = os.path.join(data_dir, 'train.csv')
test_csv  = os.path.join(data_dir, 'test.csv')
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'custom_emotion_cnn.pth')
batch_size = 64
epochs = 30
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FEROneHotDataset(Dataset):
    """
    Dataset for FER2013 one-hot CSV without header.
    Assumes each row: [pixel0, pixel1, ..., pixel2303, label0, ..., label6]
    """
    def __init__(self, csv_file, transform=None, num_labels=7):
        # load all data as numpy array
        data = np.loadtxt(csv_file, delimiter=',')
        # split pixels and labels
        self.pixels = data[:, :-num_labels].astype(np.uint8)
        self.labels = data[:, -num_labels:].astype(int)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # get pixel vector and reshape to image
        pixel_vec = self.pixels[idx]
        img = pixel_vec.reshape(48, 48).astype(np.uint8)
        img = img[:, :, np.newaxis]  # add channel dim
        if self.transform:
            img = self.transform(img)
        # convert one-hot to index
        label = int(np.argmax(self.labels[idx]))
        return img, label


def train():
    # Transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Datasets & loaders
    train_ds = FEROneHotDataset(train_csv, transform)
    test_ds  = FEROneHotDataset(test_csv, transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    # Model, loss, optimizer
    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            # Debug first batch of first epoch
            if epoch == 1 and batch_idx == 0:
                print(f"DEBUG: imgs batch shape = {imgs.shape}, labels batch shape = {labels.shape}")

            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch}/{epochs} — Loss: {running_loss/len(train_loader):.4f} — Test Acc: {acc:.4f}")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

    print(f"Training complete. Best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train()
