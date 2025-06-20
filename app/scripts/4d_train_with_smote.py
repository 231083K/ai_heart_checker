import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE # SMOTEをインポート
from collections import Counter

# --- グローバル設定 & ハイパーパラメータ ---
PROCESSED_DATA_DIR = './data/processed/'
MODEL_SAVE_PATH = './models/'
FIGURE_SAVE_PATH = './figures/'
INPUT_SIZE = 288
NUM_CLASSES = 4
BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 0.001

# --- データ拡張のための関数群 (変更なし) ---
def add_noise(signal, noise_level=0.05):
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise
def scale_amplitude(signal, scale_factor_range=(0.9, 1.1)):
    scale_factor = np.random.uniform(scale_factor_range[0], scale_factor_range[1])
    return signal * scale_factor
def time_shift(signal, max_shift=10):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(signal, shift)

class ECGAugmentations:
    def __init__(self, probability=0.5):
        self.probability = probability
    def __call__(self, signal):
        signal = np.copy(signal)
        if np.random.rand() < self.probability: signal = add_noise(signal)
        if np.random.rand() < self.probability: signal = scale_amplitude(signal)
        if np.random.rand() < self.probability: signal = time_shift(signal)
        return signal

# --- カスタムDatasetクラス (変更なし) ---
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

# --- モデル定義 (変更なし) ---
class ECG_CNN(nn.Module):
    def __init__(self, num_classes):
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

# --- データ読み込み部分の修正 (SMOTEを適用) ---
def prepare_dataloaders(device):
    print("Loading preprocessed data...")
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'))
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
    
    # --- ここからがSMOTEの処理 ---
    print("\nApplying SMOTE to the training data...")
    print(f"Original training data distribution: {sorted(Counter(y_train).items())}")
    
    # SMOTEは2Dデータを要求するため、(N, L)の形状に一旦変形
    n_samples, n_timesteps = X_train.shape
    X_train_reshaped = X_train.reshape(n_samples, n_timesteps)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train)
    
    print(f"Resampled training data distribution: {sorted(Counter(y_train_resampled).items())}")
    # --- SMOTE処理ここまで ---
    
    # データセット作成にはリサンプリング後のデータを使用
    train_dataset = ECGDataset(X_train_resampled, y_train_resampled, transform=ECGAugmentations())
    val_dataset = ECGDataset(X_val, y_val, transform=None)
    test_dataset = ECGDataset(X_test, y_test, transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # SMOTEでデータが均衡になったため、クラス重みは不要
    print("\nData loading complete. SMOTE and Augmentation enabled for training data.")
    return train_loader, val_loader, test_loader

# --- 学習・評価ループ (損失関数から重みを削除) ---
def train(model, train_loader, val_loader, criterion, optimizer, device):
    # (この関数の中身はほぼ変更なし)
    best_val_loss = float('inf')
    print("\n--- Starting Model Training with SMOTE + Augmentation ---")
    for epoch in range(NUM_EPOCHS):
        # ... (ループ処理は前回のスクリプトと同じ) ...
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'best_model_smote.pth'))
            print(f"  -> Model saved with improved validation loss: {best_val_loss:.4f}")
    print("--- Training Finished ---")

def validate(model, loader, criterion, device):
    # (この関数の中身は変更なし)
    model.eval()
    # ... (処理は前回のスクリプトと同じ) ...
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / total, correct / total

def evaluate(model, test_loader, device):
    # (この関数の中身は変更なし、ファイル名のみ変更)
    print("\n--- Evaluating Model on Test Set ---")
    model_path = os.path.join(MODEL_SAVE_PATH, 'best_model_smote.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    # ... (処理は前回のスクリプトと同じ) ...
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
    plt.title('Confusion Matrix (with SMOTE + Augmentation)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    if not os.path.exists(FIGURE_SAVE_PATH): os.makedirs(FIGURE_SAVE_PATH)
    plt.savefig(os.path.join(FIGURE_SAVE_PATH, 'confusion_matrix_smote.png'))
    print(f"Confusion matrix saved to {FIGURE_SAVE_PATH}confusion_matrix_smote.png")


# --- メイン実行ブロック (損失関数から重みを削除) ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # prepare_dataloadersはもう重みを返さない
    train_loader, val_loader, test_loader = prepare_dataloaders(device)
    
    model = ECG_CNN(num_classes=NUM_CLASSES).to(device)
    
    # SMOTEで訓練データが均衡になったため、クラス重みは不要
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train(model, train_loader, val_loader, criterion, optimizer, device)
    
    evaluate(model, test_loader, device)