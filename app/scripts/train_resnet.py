import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from tqdm import tqdm

# --- データ拡張のための関数群 ---
def add_noise(signal, noise_level=0.05): return signal + np.random.normal(0, noise_level, signal.shape)
def scale_amplitude(signal, scale_factor_range=(0.9, 1.1)): return signal * np.random.uniform(scale_factor_range[0], scale_factor_range[1])
def time_shift(signal, max_shift=10): return np.roll(signal, np.random.randint(-max_shift, max_shift))

class ECGAugmentations:
    def __init__(self, probability=0.5): self.probability = probability
    def __call__(self, signal):
        signal = np.copy(signal)
        if np.random.rand() < self.probability: signal = add_noise(signal)
        if np.random.rand() < self.probability: signal = scale_amplitude(signal)
        if np.random.rand() < self.probability: signal = time_shift(signal)
        return signal

from torch.utils.data import DataLoader, Dataset
class ECGDataset(Dataset):
    def __init__(self, data, labels, transform=None): self.data = np.expand_dims(data, 1).astype(np.float32); self.labels = labels.astype(np.int64); self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        signal, label = self.data[idx], self.labels[idx]
        if self.transform:
            signal_augmented = self.transform(signal.squeeze(0))
            return torch.tensor(signal_augmented, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

FIGURE_SAVE_PATH = './figures/'
def evaluate(model, test_loader, device, figure_savename, model_name):
    print(f"\n--- Evaluating {model_name} on Test Set ---")
    model.eval(); all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device)); _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    class_names = ['N', 'S', 'V', 'Q/F']; print("\nClassification Report:"); print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    cm = confusion_matrix(all_labels, all_preds); plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names); plt.title(f'Confusion Matrix ({model_name})'); plt.ylabel('Actual'); plt.xlabel('Predicted')
    if not os.path.exists(FIGURE_SAVE_PATH): os.makedirs(FIGURE_SAVE_PATH)
    plt.savefig(os.path.join(FIGURE_SAVE_PATH, figure_savename)); print(f"Confusion matrix saved to {os.path.join(FIGURE_SAVE_PATH, figure_savename)}")

from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd
PROCESSED_DATA_DIR = './data/processed/'
BATCH_SIZE = 256
def prepare_pretrain_loader():
    print("--- Loading data for Pre-training Phase ---"); X_ptbxl = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_pretrain_ptbxl.npy')); y_ptbxl = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_pretrain_ptbxl.npy')); y_ptbxl_mapped = np.where(y_ptbxl == 0, 0, 3); X_mit_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy')); y_mit_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy')); X_pretrain = np.concatenate((X_ptbxl, X_mit_train), axis=0); y_pretrain = np.concatenate((y_ptbxl_mapped, y_mit_train), axis=0)
    pretrain_dataset = ECGDataset(X_pretrain, y_pretrain, transform=ECGAugmentations()); pretrain_loader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0); return pretrain_loader
def prepare_finetune_loader():
    print("\n--- Loading data for Fine-tuning Phase ---"); X_mit_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy')); y_mit_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy')); n_samples, n_timesteps = X_mit_train.shape; X_train_reshaped = X_mit_train.reshape(n_samples, n_timesteps); smote = SMOTE(random_state=42); X_resampled, y_resampled = smote.fit_resample(X_train_reshaped, y_mit_train); print(f"Resampled MIT-BIH training distribution: {sorted(Counter(y_resampled).items())}")
    finetune_dataset = ECGDataset(X_resampled, y_resampled, transform=ECGAugmentations()); finetune_loader = DataLoader(finetune_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0); return finetune_loader


# --- グローバル設定 & ハイパーパラメータ ---
MODEL_SAVE_PATH = './models/'
INPUT_SIZE, NUM_CLASSES = 288, 4
PRETRAIN_EPOCHS = 5
FROZEN_FINETUNE_EPOCHS = 20
UNFROZEN_FINETUNE_EPOCHS = 10
PRETRAIN_LR = 0.001
FROZEN_FINETUNE_LR = 0.0001
UNFROZEN_FINETUNE_LR = 1e-6

# --- ResNetのコアとなるResidual Block ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# --- ResNetを組み込んだ新しいモデル ---
class ECG_ResNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ECG_ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=16, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.layer1 = self._make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, kernel_size=7, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def run_resnet_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 事前学習フェーズ ---
    pretrain_loader = prepare_pretrain_loader()
    model = ECG_ResNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=PRETRAIN_LR)

    print("\n--- Phase 1: Pre-training ResNet ---")
    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        progress_bar = tqdm(pretrain_loader, desc=f"Pre-train Epoch {epoch+1}/{PRETRAIN_EPOCHS}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    # --- 2. 凍結ファインチューニングフェーズ ---
    finetune_loader = prepare_finetune_loader()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FROZEN_FINETUNE_LR)

    print("\n--- Phase 2: Frozen Fine-tuning ResNet ---")
    for epoch in range(FROZEN_FINETUNE_EPOCHS):
        model.train()
        progress_bar = tqdm(finetune_loader, desc=f"Frozen FT Epoch {epoch+1}/{FROZEN_FINETUNE_EPOCHS}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_ft.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_ft.step()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    # --- 3. 全層ファインチューニングフェーズ ---
    for param in model.parameters():
        param.requires_grad = True
    optimizer_full = optim.Adam(model.parameters(), lr=UNFROZEN_FINETUNE_LR)

    print("\n--- Phase 3: Unfrozen Full Fine-tuning ResNet ---")
    for epoch in range(UNFROZEN_FINETUNE_EPOCHS):
        model.train()
        progress_bar = tqdm(finetune_loader, desc=f"Full FT Epoch {epoch+1}/{UNFROZEN_FINETUNE_EPOCHS}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_full.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_full.step()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    print("All training phases finished.")

    # --- 最終評価 ---
    test_loader = DataLoader(ECGDataset(np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy')), np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))), batch_size=BATCH_SIZE, num_workers=0)
    
    model_savename = 'best_model_resnet.pth'
    figure_savename = 'confusion_matrix_resnet.png'
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, model_savename))
    
    evaluate(model, test_loader, device, figure_savename, "ResNet Model")

if __name__ == '__main__':
    run_resnet_training()