import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# モデルA (シンプルなCNN)
class ECG_CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(ECG_CNN, self).__init__()
        self.conv_layers = nn.Sequential(nn.Conv1d(1, 64, 16, 1, 8), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2), nn.Conv1d(64, 128, 8, 1, 4), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2, 2), nn.Conv1d(128, 256, 4, 1, 2), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2, 2))
        self.fc_layers = nn.Sequential(nn.Linear(256 * (288 // 8), 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes))
    def forward(self, x):
        x = self.conv_layers(x); x = x.view(x.size(0), -1); x = self.fc_layers(x)
        return x

# モデルB (ResNet)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ResidualBlock, self).__init__(); self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False); self.bn1 = nn.BatchNorm1d(out_channels); self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2, bias=False); self.bn2 = nn.BatchNorm1d(out_channels); self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels: self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm1d(out_channels))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))); out = self.bn2(self.conv2(out)); out += self.shortcut(x); out = F.relu(out)
        return out
class ECG_ResNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ECG_ResNet, self).__init__(); self.in_channels = 64; self.conv1 = nn.Conv1d(1, 64, 16, 2, 7, bias=False); self.bn1 = nn.BatchNorm1d(64); self.layer1 = self._make_layer(ResidualBlock, 64, 2, stride=1); self.layer2 = self._make_layer(ResidualBlock, 128, 2, stride=2); self.layer3 = self._make_layer(ResidualBlock, 256, 2, stride=2); self.layer4 = self._make_layer(ResidualBlock, 512, 2, stride=2); self.avg_pool = nn.AdaptiveAvgPool1d(1); self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1); layers = []
        for s in strides: layers.append(block(self.in_channels, out_channels, kernel_size=7, stride=s)); self.in_channels = out_channels
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))); out = self.layer1(out); out = self.layer2(out); out = self.layer3(out); out = self.layer4(out); out = self.avg_pool(out); out = out.view(out.size(0), -1); out = self.fc(out)
        return out
        
# --- データセット定義 ---
class ECGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = np.expand_dims(data, 1).astype(np.float32)
        self.labels = labels.astype(np.int64)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def evaluate_ensemble_model():
    print("--- Evaluating Ensemble Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 2つのモデルをロード ---
    print("Loading models...")
    # モデル1: 転移学習モデル
    model_transfer = ECG_ResNet().to(device) 
    model_transfer.load_state_dict(torch.load('./models/best_model_resnet.pth', map_location=device))
    model_transfer.eval()
    print(" -> ResNet Transfer Learning model loaded.")

    # モデル2: SMOTE特化モデル
    model_smote = ECG_CNN().to(device)
    model_smote.load_state_dict(torch.load('./models/best_model_smote.pth', map_location=device))
    model_smote.eval()
    print(" -> SMOTE-specialist model loaded.")

    # --- 2. テストデータをロード ---
    X_test = np.load('./data/processed/X_test.npy')
    y_test = np.load('./data/processed/y_test.npy')
    test_dataset = ECGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    print("Test data loaded.")

    # --- 3. アンサンブルで予測 ---
    print("Running ensemble prediction...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            
            # 各モデルで確率を計算
            probs_transfer = F.softmax(model_transfer(inputs), dim=1)
            probs_smote = F.softmax(model_smote(inputs), dim=1)
            
            # 確率を平均化
            avg_probs = (probs_transfer + probs_smote) / 2
            
            _, final_preds = torch.max(avg_probs, 1)
            
            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 4. 最終評価 ---
    class_names = ['N', 'S', 'V', 'Q/F']
    print("\n--- Final Ensemble Model Performance ---")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Ensemble Model)')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    if not os.path.exists('./figures/'): os.makedirs('./figures/')
    plt.savefig('./figures/confusion_matrix_ensemble.png')
    print("Ensemble confusion matrix saved to './figures/confusion_matrix_ensemble.png'")

if __name__ == '__main__':
    evaluate_ensemble_model()