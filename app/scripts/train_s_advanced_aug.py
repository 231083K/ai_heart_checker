import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from tsaug import AddNoise, Quantize, Drift, TimeWarp

# --- データ拡張の定義 (ユーザー様発見の正しい形式) ---
# 複数の拡張処理からランダムに1つを選択して実行するパイプライン
augmenter = (
    TimeWarp(n_speed_change=5, max_speed_ratio=3, prob=1.0) +
    AddNoise(scale=0.05, prob=1.0) +
    Quantize(n_levels=20, prob=1.0) +
    Drift(max_drift=0.1, prob=1.0)
)

# --- カスタムDatasetクラス (augment呼び出しを修正) ---
class ECGDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            signal_3d = signal.reshape(1, -1, 1)
            # ★★★ ここが重要な修正点：不要な 'y' 引数を削除 ★★★
            signal_aug = self.transform.augment(signal_3d)
            signal = signal_aug[0, :, 0]
        return torch.tensor(signal, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)

# --- CNN-LSTMモデル定義 (変更なし) ---
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_LSTM, self).__init__(); self.conv1 = nn.Sequential(nn.Conv1d(1, 64, 16, 2, 7), nn.BatchNorm1d(64), nn.ReLU()); self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 8, 2, 3), nn.BatchNorm1d(128), nn.ReLU()); self.lstm = nn.LSTM(input_size=128, hidden_size=100, num_layers=2, batch_first=True, bidirectional=True); self.fc = nn.Linear(100 * 2, num_classes)
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = x.permute(0, 2, 1); x, _ = self.lstm(x); x = self.fc(x[:, -1, :]); return x

# --- 設定 (変更なし) ---
PROCESSED_DATA_DIR = './data/processed/'; MODEL_SAVE_PATH = './models/'
BATCH_SIZE, NUM_EPOCHS, MAX_LR = 128, 30, 0.001

# --- メイン実行関数 (変更なし) ---
def train_s_model_advanced():
    print("--- Training S-specialist model with Advanced Augmentation + One-Cycle LR ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train_svdb_NS.npy')); y = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train_svdb_NS.npy'))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_dataset = ECGDataset(X_train, y_train, transform=augmenter)
    val_dataset = ECGDataset(X_val, y_val, transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)
    
    model = CNN_LSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MAX_LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader))

    for epoch in range(NUM_EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step(); scheduler.step()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

    if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 's_specialist_model_adv_aug.pth'))
    print("\nAdvanced training complete and model saved.")

    print("\n--- Evaluating model on validation set ---")
    model.eval(); all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.to(device)); _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    class_names = ['N (Normal)', 'S (Supraventricular)']; print("\nValidation Classification Report:"); print(classification_report(all_preds, all_labels, target_names=class_names, zero_division=0))

if __name__ == '__main__':
    train_s_model_advanced()