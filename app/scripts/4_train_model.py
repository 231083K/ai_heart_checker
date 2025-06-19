import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- グローバル設定 & ハイパーパラメータ ---
PROCESSED_DATA_DIR = './data/processed/'
MODEL_SAVE_PATH = './models/'
FIGURE_SAVE_PATH = './figures/'

# モデル設定
INPUT_SIZE = 288  # セグメントの長さ (フェーズ2で設定)
NUM_CLASSES = 4   # クラス数 (N, S, V, Q/F)

# 学習設定
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# --- 1. 1D-CNNモデルの定義 ---
class ECG_CNN(nn.Module):
    def __init__(self, num_classes):
        super(ECG_CNN, self).__init__()
        # 特徴抽出層 (Convolutional Layers)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # 分類層 (Fully Connected Layers)
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * (INPUT_SIZE // 8), 512), # MaxPool1d 3回で 2^3=8 で割る
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

# --- 2. データ読み込みとDataLoaderの準備 ---
def prepare_dataloaders(device):
    print("Loading preprocessed data...")
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'))
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))

    # Conv1dのためにチャネル次元を追加 (N, L) -> (N, C, L)
    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)
    X_test = np.expand_dims(X_test, 1)
    
    # Numpy配列をPyTorchテンソルに変換
    train_tensors = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_tensors = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_tensors = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    # DataLoaderを作成
    train_loader = DataLoader(train_tensors, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_tensors, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_tensors, batch_size=BATCH_SIZE, shuffle=False)
    
    # クラス不均衡対策の重みを計算
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("Data loading complete.")
    return train_loader, val_loader, test_loader, class_weights_tensor

# --- 3. 学習ループ ---
def train(model, train_loader, val_loader, criterion, optimizer, device):
    best_val_loss = float('inf')
    print("\n--- Starting Model Training ---")

    for epoch in range(NUM_EPOCHS):
        model.train() # 学習モード
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

        # 検証
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 最良モデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'best_model.pth'))
            print(f"  -> Model saved with improved validation loss: {best_val_loss:.4f}")

    print("--- Training Finished ---")

def validate(model, loader, criterion, device):
    model.eval() # 評価モード
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

# --- 4. 評価 ---
def evaluate(model, test_loader, device):
    print("\n--- Evaluating Model on Test Set ---")
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, 'best_model.pth')))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 評価レポート
    class_names = ['N', 'S', 'V', 'Q/F']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 混同行列
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    if not os.path.exists(FIGURE_SAVE_PATH): os.makedirs(FIGURE_SAVE_PATH)
    plt.savefig(os.path.join(FIGURE_SAVE_PATH, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {FIGURE_SAVE_PATH}confusion_matrix.png")

# --- メイン実行ブロック ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_weights = prepare_dataloaders(device)
    
    model = ECG_CNN(num_classes=NUM_CLASSES).to(device)
    
    # 損失関数にクラス重みを設定
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train(model, train_loader, val_loader, criterion, optimizer, device)
    
    evaluate(model, test_loader, device)