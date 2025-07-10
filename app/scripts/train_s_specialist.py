import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report # 評価レポート用
from tqdm import tqdm

# --- 設定 ---
PROCESSED_DATA_DIR = './data/processed/'
MODEL_SAVE_PATH = './models/'
INPUT_SIZE, BATCH_SIZE, NUM_EPOCHS, LR = 288, 128, 20, 0.001

# --- CNN-LSTM ハイブリッドモデル定義 (変更なし) ---
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=2): # NとSの2クラス分類
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1, 64, 16, 2, 7), nn.BatchNorm1d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 8, 2, 3), nn.BatchNorm1d(128), nn.ReLU())
        self.lstm = nn.LSTM(input_size=128, hidden_size=100, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(100 * 2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

def train_s_model():
    print("--- Training S-class specialist model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        X = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train_svdb_NS.npy'))
        y = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train_svdb_NS.npy'))
    except FileNotFoundError:
        print(f"Error: Preprocessed SVDB data not found. Please run 'step3e_preprocess_svdb_for_S.py' first.")
        return

    X = np.expand_dims(X, 1)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=BATCH_SIZE)
    
    model = CNN_LSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step(); progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 's_specialist_model.pth'))
    print("\nS-specialist model training complete and saved.")

    # --- ▼▼▼ ここからが追加された評価ロジック ▼▼▼ ---
    print("\n--- Evaluating S-specialist model on validation set ---")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    class_names = ['N (Normal)', 'S (Supraventricular)']
    print("\nValidation Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    # --- ▲▲▲ 評価ロジックここまで ▲▲▲ ---

if __name__ == '__main__':
    train_s_model()