import os
import numpy as np
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

from .step4d_train_smote import ECGDataset, ECGAugmentations, ECG_CNN, validate, evaluate

# --- 設定 ---
PROCESSED_DATA_DIR = './data/processed/'
MODEL_SAVE_PATH = './models/'
FIGURE_SAVE_PATH = './figures/'

# --- ハイパーパラメータ ---
BATCH_SIZE = 256
PRETRAIN_EPOCHS = 5
FINETUNE_EPOCHS = 20
PRETRAIN_LR = 0.001
FINETUNE_LR = 0.0001

# (これ以降の関数の内容は変更ありません)
def prepare_pretrain_loader():
    """事前学習用データ（PTB-XL + MIT-BIH）を準備する"""
    print("--- Loading data for Pre-training Phase ---")
    
    X_ptbxl = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_pretrain_ptbxl.npy'))
    y_ptbxl = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_pretrain_ptbxl.npy'))
    y_ptbxl_mapped = np.where(y_ptbxl == 0, 0, 3)

    X_mit_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
    y_mit_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))

    X_pretrain = np.concatenate((X_ptbxl, X_mit_train), axis=0)
    y_pretrain = np.concatenate((y_ptbxl_mapped, y_mit_train), axis=0)
    
    print(f"Total pre-training samples: {len(X_pretrain)}")
    print(f"Pre-training data distribution: {sorted(Counter(y_pretrain).items())}")

    pretrain_dataset = ECGDataset(X_pretrain, y_pretrain, transform=ECGAugmentations())
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return pretrain_loader

def prepare_finetune_loader():
    """ファインチューニング用データ（SMOTE適用済みMIT-BIH）を準備する"""
    print("\n--- Loading data for Fine-tuning Phase ---")
    
    X_mit_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
    y_mit_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))

    print(f"Original MIT-BIH training distribution: {sorted(Counter(y_mit_train).items())}")
    n_samples, n_timesteps = X_mit_train.shape
    X_train_reshaped = X_mit_train.reshape(n_samples, n_timesteps)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_reshaped, y_mit_train)
    print(f"Resampled MIT-BIH training distribution: {sorted(Counter(y_resampled).items())}")

    finetune_dataset = ECGDataset(X_resampled, y_resampled, transform=ECGAugmentations())
    finetune_loader = DataLoader(finetune_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return finetune_loader

def main_transfer_learning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 事前学習フェーズ ---
    pretrain_loader = prepare_pretrain_loader()
    model = ECG_CNN(num_classes=4).to(device)
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

    # --- 2. ファインチューニングフェーズ ---
    finetune_loader = prepare_finetune_loader()
    for param in model.conv_layers.parameters():
        param.requires_grad = False
    optimizer_ft = optim.Adam(model.fc_layers.parameters(), lr=FINETUNE_LR)

    print("\n--- Phase 2: Fine-tuning on SMOTE'd MIT-BIH data ---")
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        progress_bar = tqdm(finetune_loader, desc=f"Fine-tune Epoch {epoch+1}/{FINETUNE_EPOCHS}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_ft.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_ft.step()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    print("Fine-tuning finished.")

    # --- 3. 最終評価 ---
    for param in model.parameters():
        param.requires_grad = True
        
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
    test_dataset = ECGDataset(X_test, y_test, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model_savename = 'best_model_transfer_learning.pth'
    figure_savename = 'confusion_matrix_transfer_learning.png'
    if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, model_savename))
    
    print("\n--- Final Evaluation on MIT-BIH Test Set ---")
    # 評価ロジックを再記述
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
    cm