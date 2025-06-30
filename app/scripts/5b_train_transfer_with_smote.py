import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
from tqdm import tqdm

# --- スクリプトのインポート ---
# 以前のスクリプトで定義したクラスや関数を再利用
from scripts.4d_train_with_smote import ECGDataset, ECGAugmentations, ECG_CNN, validate, evaluate

# --- 設定 ---
PROCESSED_DATA_DIR = './data/processed/'
MODEL_SAVE_PATH = './models/'

# --- ハイパーパラメータ ---
BATCH_SIZE = 256
# 事前学習フェーズ
PRETRAIN_EPOCHS = 5
PRETRAIN_LR = 0.001
# ファインチューニングフェーズ
FINETUNE_EPOCHS = 15
FINETUNE_LR = 0.0001 # ファインチューニングでは学習率を小さくする

def prepare_pretrain_loader(device):
    """事前学習用データ（PTB-XL + MIT-BIH）を準備する"""
    print("--- Loading data for Pre-training Phase ---")
    
    # PTB-XLデータの読み込み
    try:
        X_ptbxl = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_pretrain_ptbxl.npy'))
        y_ptbxl = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_pretrain_ptbxl.npy'))
    except FileNotFoundError:
        print("Error: PTB-XL preprocessed data not found. Please run '3b_preprocess_ptbxl.py' first.")
        return None
    
    # PTB-XLのラベルをMIT-BIHのクラス体系にマッピング (正常:0, 異常:3)
    # N -> 0, Abnormal -> 3 (Q/F class)
    y_ptbxl_mapped = np.where(y_ptbxl == 0, 0, 3)
    
    # MIT-BIH訓練データの読み込み
    X_mit_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
    y_mit_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))

    # 全ての事前学習データを結合
    X_pretrain = np.concatenate((X_ptbxl, X_mit_train), axis=0)
    y_pretrain = np.concatenate((y_ptbxl_mapped, y_mit_train), axis=0)
    
    print(f"Total pre-training samples: {len(X_pretrain)}")

    # データ拡張を適用したDatasetを作成
    pretrain_dataset = ECGDataset(X_pretrain, y_pretrain, transform=ECGAugmentations())
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    return pretrain_loader

def prepare_finetune_loader(device):
    """ファインチューニング用データ（SMOTE適用済みMIT-BIH）を準備する"""
    print("\n--- Loading data for Fine-tuning Phase ---")
    
    X_mit_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
    y_mit_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))

    # SMOTEでMIT-BIH訓練データを均衡化
    print(f"Original MIT-BIH training distribution: {sorted(Counter(y_mit_train).items())}")
    n_samples, n_timesteps = X_mit_train.shape
    X_train_reshaped = X_mit_train.reshape(n_samples, n_timesteps)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_reshaped, y_mit_train)
    print(f"Resampled MIT-BIH training distribution: {sorted(Counter(y_resampled).items())}")

    # SMOTE適用後のデータでDatasetを作成（データ拡張も適用）
    finetune_dataset = ECGDataset(X_resampled, y_resampled, transform=ECGAugmentations())
    finetune_loader = DataLoader(finetune_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    return finetune_loader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 事前学習フェーズ ---
    pretrain_loader = prepare_pretrain_loader()
    if pretrain_loader is None: return

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
    
    # 特徴抽出層（conv_layers）の重みを凍結
    for param in model.conv_layers.parameters():
        param.requires_grad = False
    
    # 分類層（fc_layers）のみを学習対象とする新しいオプティマイザ
    optimizer_ft = optim.Adam(model.fc_layers.parameters(), lr=FINETUNE_LR)

    print("\n--- Phase 2: Fine-tuning on SMOTE'd MIT-BIH data ---")
    for epoch in range(FINETUNE_EPOCHS):
        model.train() # fc_layersのみが学習される
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
    # 評価のために全ての層の凍結を解除
    for param in model.parameters():
        param.requires_grad = True
        
    # MIT-BIHのテストデータで評価
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
    test_dataset = ECGDataset(X_test, y_test, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 保存ファイル名を変更
    model_savename = 'best_model_transfer_learning.pth'
    if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, model_savename))
    
    print("\n--- Final Evaluation on MIT-BIH Test Set ---")
    # 以前作成したevaluate関数を呼び出す
    evaluate(model, test_loader, device)


if __name__ == '__main__':
    # このスクリプトは他のスクリプトから関数をインポートするため、
    # 循環インポートを避けるため直接実行する
    main()