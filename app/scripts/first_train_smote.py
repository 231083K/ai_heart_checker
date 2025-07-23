import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter
from tqdm import tqdm

PROCESSED_DATA_DIR = './data/processed/'
MODEL_SAVE_PATH = './models/'
INPUT_SIZE, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE = 288, 4, 128, 30, 0.001

class ECGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = np.expand_dims(data, 1).astype(np.float32)
        self.labels = labels.astype(np.int64)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class ECG_CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ECG_CNN, self).__init__(); self.conv_layers = nn.Sequential(nn.Conv1d(1, 64, 16, 1, 8), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2), nn.Conv1d(64, 128, 8, 1, 4), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2, 2), nn.Conv1d(128, 256, 4, 1, 2), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2, 2)); self.fc_layers = nn.Sequential(nn.Linear(256 * (INPUT_SIZE // 8), 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes))
    def forward(self, x): x = self.conv_layers(x); x = x.view(x.size(0), -1); x = self.fc_layers(x); return x

def train_smote_model():
    print("--- Training SMOTE-specialist model (for Ensemble) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
    
    print(f"Original MIT-BIH training distribution: {sorted(Counter(y_train).items())}")
    n_samples, n_timesteps = X_train.shape
    X_train_reshaped = X_train.reshape(n_samples, n_timesteps)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_reshaped, y_train)
    print(f"Resampled MIT-BIH training distribution: {sorted(Counter(y_resampled).items())}")
    
    train_dataset = ECGDataset(X_resampled, y_resampled)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = ECG_CNN().to(device)
    criterion = nn.CrossEntropyLoss() # データは均衡なので重みは不要
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'best_model_smote.pth'))
    print("\nSMOTE-specialist model training complete and saved as 'best_model_smote.pth'.")

if __name__ == '__main__':
    train_smote_model()