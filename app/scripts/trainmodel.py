import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
from tqdm import tqdm

# --- グローバル設定 & ハイパーパラメータ ---
PROCESSED_DATA_DIR = './data/processed/'
MODEL_SAVE_PATH = './models/'
FIGURE_SAVE_PATH = './figures/'
INPUT_SIZE = 288
NUM_CLASSES = 4
BATCH_SIZE = 256
PRETRAIN_EPOCHS = 5
FROZEN_FINETUNE_EPOCHS = 20
UNFROZEN_FINETUNE_EPOCHS = 10
PRETRAIN_LR = 0.001
FROZEN_FINETUNE_LR = 0.0001
UNFROZEN_FINETUNE_LR = 1e-6 # 全層ファインチューニングでは、非常に小さい学習率を使う

# --- データ拡張のための関数群 ---
def add_noise(signal, noise_level=0.05):
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise
def scale_amplitude(signal, scale_factor_range=(0.9, 1.1)):
    scale_factor = np.random.uniform(scale_factor_range[0], scale_factor_range[1])
    return signal * scale_factor
def time_shift(signal, max_shift=10):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(signal, shift)

# --- データ拡張をまとめるTransformクラス ---
class ECGAugmentations:
    def __init__(self, probability=0.5):
        self.probability = probability
    def __call__(self, signal):
        signal = np.copy(signal)
        if np.random.rand() < self.probability: signal = add_noise(signal)
        if np.random.rand() < self.probability: signal = scale_amplitude(signal)
        if np.random.rand() < self.probability: signal = time_shift(signal)
        return signal

# --- カスタムDatasetクラス ---
class ECGDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = np.expand_dims(data, 1).astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            signal_augmented = self.transform(signal.squeeze(0))
            signal_tensor = torch.tensor(signal_augmented, dtype=torch.float32).unsqueeze(0)
            return signal_tensor, torch.tensor(label, dtype=torch.long)
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# --- 1D-CNNモデル定義 ---
class ECG_CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(ECG_CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, 16, 1, 8), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 128, 8, 1, 4), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(128, 256, 4, 1, 2), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * (INPUT_SIZE // 8), 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# --- 評価用関数 ---
def evaluate(model, test_loader, device, figure_savename):
    print("\n--- Evaluating Model on Test Set ---")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    class_names = ['N', 'S', 'V', 'Q/F']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Ultimate Model)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    if not os.path.exists(FIGURE_SAVE_PATH): os.makedirs(FIGURE_SAVE_PATH)
    plt.savefig(os.path.join(FIGURE_SAVE_PATH, figure_savename))
    print(f"Confusion matrix saved to {os.path.join(FIGURE_SAVE_PATH, figure_savename)}")


def prepare_pretrain_loader():
    print("--- Loading data for Pre-training Phase ---")
    X_ptbxl = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_pretrain_ptbxl.npy'))
    y_ptbxl = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_pretrain_ptbxl.npy'))
    y_ptbxl_mapped = np.where(y_ptbxl == 0, 0, 3)
    X_mit_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
    y_mit_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
    X_pretrain = np.concatenate((X_ptbxl, X_mit_train), axis=0)
    y_pretrain = np.concatenate((y_ptbxl_mapped, y_mit_train), axis=0)
    print(f"Total pre-training samples: {len(X_pretrain)}")
    pretrain_dataset = ECGDataset(X_pretrain, y_pretrain, transform=ECGAugmentations())
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return pretrain_loader

def prepare_finetune_loader():
    print("\n--- Loading data for Fine-tuning Phase ---")
    X_mit_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
    y_mit_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
    n_samples, n_timesteps = X_mit_train.shape
    X_train_reshaped = X_mit_train.reshape(n_samples, n_timesteps)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_reshaped, y_mit_train)
    print(f"Resampled MIT-BIH training distribution: {sorted(Counter(y_resampled).items())}")
    finetune_dataset = ECGDataset(X_resampled, y_resampled, transform=ECGAugmentations())
    finetune_loader = DataLoader(finetune_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return finetune_loader

def run_ultimate_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 事前学習フェーズ ---
    pretrain_loader = prepare_pretrain_loader()
    model = ECG_CNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=PRETRAIN_LR)

    print("\n--- Phase 1: Pre-training on PTB-XL + MIT-BIH data ---")
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
    
    print("Pre-training finished.")

    # --- 2. 凍結ファインチューニングフェーズ ---
    finetune_loader = prepare_finetune_loader()
    for param in model.conv_layers.parameters():
        param.requires_grad = False
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FROZEN_FINETUNE_LR)

    print("\n--- Phase 2: Frozen Fine-tuning ---")
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

    print("\n--- Phase 3: Unfrozen Full Fine-tuning ---")
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
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
    test_dataset = ECGDataset(X_test, y_test, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model_savename = 'ultimate_model.pth'
    figure_savename = 'confusion_matrix_ultimate.png'
    if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, model_savename))
    
    evaluate(model, test_loader, device, figure_savename)


if __name__ == '__main__':
    run_ultimate_training()