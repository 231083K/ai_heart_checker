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



class ECG_CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(ECG_CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, 16, 1, 8), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 128, 8, 1, 4), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(128, 256, 4, 1, 2), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * (288 // 8), 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x); x = x.view(x.size(0), -1); x = self.fc_layers(x)
        return x

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
        out = F.relu(self.bn1(self.conv1(x))); out = self.bn2(self.conv2(out))
        out += self.shortcut(x); out = F.relu(out)
        return out
class ECG_ResNet(nn.Module):
    def __init__(self, num_classes=4):
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
        strides = [stride] + [1]*(num_blocks-1); layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, kernel_size=7, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))); out = self.layer1(out); out = self.layer2(out)
        out = self.layer3(out); out = self.layer4(out); out = self.avg_pool(out)
        out = out.view(out.size(0), -1); out = self.fc(out)
        return out

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1, 64, 16, 2, 7), nn.BatchNorm1d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 8, 2, 3), nn.BatchNorm1d(128), nn.ReLU())
        self.lstm = nn.LSTM(input_size=128, hidden_size=100, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(100 * 2, num_classes)
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = x.permute(0, 2, 1)
        x, _ = self.lstm(x); x = self.fc(x[:, -1, :])
        return x
        
# --- データセット定義 ---
class ECGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = np.expand_dims(data, 1).astype(np.float32)
        self.labels = labels.astype(np.int64)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def evaluate_final_ensemble():
    print("--- Evaluating Final 3-Model Ensemble (Hierarchical Logic) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 3つのモデルをロード ---
    print("Loading all models...")
    model_smote = ECG_CNN().to(device)
    model_smote.load_state_dict(torch.load('./models/best_model_smote.pth', map_location=device))
    model_smote.eval()
    print(" -> SMOTE-specialist model loaded.")

    model_resnet = ECG_ResNet().to(device)
    model_resnet.load_state_dict(torch.load('./models/best_model_resnet.pth', map_location=device))
    model_resnet.eval()
    print(" -> ResNet Transfer Learning model loaded.")
    
    model_s_specialist = CNN_LSTM().to(device)
    model_s_specialist.load_state_dict(torch.load('./models/best_model_S.pth', map_location=device))
    model_s_specialist.eval()
    print(" -> S-specialist model loaded.")


    # --- 2. テストデータをロード ---
    X_test = np.load('./data/processed/X_test.npy')
    y_test = np.load('./data/processed/y_test.npy')
    test_dataset = ECGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    print("Test data loaded.")

    # --- 3. アンサンブルで予測 ---
    print("Running hierarchical ensemble prediction...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            
            # 1. 総合医チーム（2つの4クラスモデル）による一次診断
            probs_smote = F.softmax(model_smote(inputs), dim=1)
            probs_resnet = F.softmax(model_resnet(inputs), dim=1)
            avg_probs_4_class = (probs_smote + probs_resnet) / 2
            _, initial_predictions = torch.max(avg_probs_4_class, 1)
            
            # 2. Sクラス専門医による二次診断の準備
            outputs_s = model_s_specialist(inputs)
            _, predictions_s_specialist = torch.max(outputs_s, 1) # 0がN, 1がS
            
            # 3. 最終的な予測を決定
            final_predictions = initial_predictions.clone()
            for i in range(len(initial_predictions)):
                # 一次診断がN(0)かS(1)だった場合、S専門医の意見を最終判断とする
                if initial_predictions[i] == 0 or initial_predictions[i] == 1:
                    final_predictions[i] = predictions_s_specialist[i]
            
            all_preds.extend(final_predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 4. 最終評価レポート ---
    class_names = ['N', 'S', 'V', 'Q/F']
    print("\n--- Final 3-Model Ensemble Performance ---")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Final Ensemble Model)')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    if not os.path.exists('./figures/'): os.makedirs('./figures/')
    plt.savefig('./figures/confusion_matrix_final_ensemble.png')
    print("Final ensemble confusion matrix saved to './figures/confusion_matrix_final_ensemble.png'")

if __name__ == '__main__':
    evaluate_final_ensemble()