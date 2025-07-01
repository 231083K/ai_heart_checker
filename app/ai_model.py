import os
import numpy as np
import torch
import torch.nn as nn
import wfdb
from scipy.signal import butter, filtfilt, resample
import matplotlib.pyplot as plt
import io
import base64

# =============================================================================
# ▼▼▼ 外部スクリプトからインポートする代わりに、ここに必要な関数を全て定義 ▼▼▼
# =============================================================================

def bandpass_filter(data, lowcut=5.0, highcut=15.0, fs=360, order=1):
    """バンドパスフィルタ（Pan-Tompkins法の一部）"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def pan_tompkins_detect(sig, fs):
    """Pan-Tompkins法に基づくR波検出アルゴリズムの簡易実装"""
    filtered_sig = bandpass_filter(sig, fs=fs)
    diff_sig = np.diff(filtered_sig)
    squared_sig = diff_sig**2
    window_size = int(0.150 * fs)
    integrated_sig = np.convolve(squared_sig, np.ones(window_size)/window_size, mode='same')
    
    qrs_peaks_indices = []
    search_radius = int(0.2 * fs)
    
    noise_peak, signal_peak, threshold = 0.0, 0.0, 0.0
    
    for i in range(len(integrated_sig)):
        if i > 0 and i < len(integrated_sig) - 1:
            if integrated_sig[i-1] < integrated_sig[i] and integrated_sig[i+1] < integrated_sig[i]:
                peak_val = integrated_sig[i]
                threshold = 0.125 * signal_peak + 0.875 * noise_peak
                if peak_val > threshold:
                    qrs_peaks_indices.append(i)
                    signal_peak = 0.125 * peak_val + 0.875 * signal_peak
                else:
                    noise_peak = 0.125 * peak_val + 0.875 * noise_peak
    
    if not qrs_peaks_indices: return np.array([])
        
    final_peaks = [qrs_peaks_indices[0]]
    for i in range(1, len(qrs_peaks_indices)):
        if qrs_peaks_indices[i] - final_peaks[-1] > search_radius:
            final_peaks.append(qrs_peaks_indices[i])

    return np.array(final_peaks)

# =============================================================================
# ▲▲▲ ここまでがインポートの代わりの定義部分 ▲▲▲
# =============================================================================


# --- モデル定義 (変更なし) ---
INPUT_SIZE = 288
NUM_CLASSES = 4
class ECG_CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
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

# --- グローバル変数としてモデルを一度だけロード (変更なし) ---
device = torch.device("cpu")
model = ECG_CNN(num_classes=NUM_CLASSES).to(device)
model_path = './models/ultimate_model.pth'
if not os.path.exists(model_path):
    model_path = './models/best_model_transfer_learning.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded model from {model_path}")
else:
    print(f"Error: Model file not found at {model_path}.")
model.eval()

# --- 推論（診断）を実行するメイン関数 (変更なし) ---
def diagnose_ecg_record(record_path):
    TARGET_FS = 360
    SEG_PRE = 108
    SEG_POST = 180
    
    try:
        signal, fields = wfdb.rdsamp(record_path)
    except Exception as e:
        return {"error": f"Could not read record file: {e}"}

    ecg_signal = signal[:, 0].astype(np.float64)
    fs = fields['fs']
    if fs != TARGET_FS:
        num_samples_360hz = int(ecg_signal.shape[0] * TARGET_FS / fs)
        ecg_signal = resample(ecg_signal, num_samples_360hz)
    
    r_peaks = pan_tompkins_detect(sig=ecg_signal, fs=TARGET_FS)
    
    if r_peaks.size == 0:
        return {"error": "No heartbeats could be detected in the signal."}
    
    segments, r_peak_locations = [], []
    for r_peak in r_peaks:
        if r_peak - SEG_PRE < 0 or r_peak + SEG_POST >= len(ecg_signal):
            continue
        segment = ecg_signal[r_peak - SEG_PRE : r_peak + SEG_POST]
        if np.std(segment) > 1e-6:
            segment = (segment - np.mean(segment)) / np.std(segment)
            segments.append(segment)
            r_peak_locations.append(r_peak)

    if not segments:
        return {"error": "No valid heartbeat segments could be extracted."}
    
    segments_np = np.expand_dims(np.array(segments, dtype=np.float32), 1)
    segments_tensor = torch.tensor(segments_np, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(segments_tensor.to(device))
        _, predictions = torch.max(outputs, 1)
    
    predictions = predictions.cpu().numpy()

    class_names = ['N (正常)', 'S (上室性)', 'V (心室性)', 'Q/F (その他)']
    counts = {name: 0 for name in class_names}
    for pred in predictions:
        counts[class_names[pred]] += 1
    
    total_beats = len(predictions)
    normal_beats = counts.get('N (正常)', 0)
    abnormal_beats = total_beats - normal_beats
    abnormal_percentage = (abnormal_beats / total_beats) * 100 if total_beats > 0 else 0
    
    s_beats = counts.get('S (上室性)', 0)
    v_beats = counts.get('V (心室性)', 0)
    
    if v_beats >= 5:
        risk_level = "high"
        summary_text = "注意: 心室性の異常な拍動が複数検出されました。専門医への相談を推奨します。"
    elif s_beats >= 10 or abnormal_percentage >= 20:
        risk_level = "medium"
        summary_text = "中程度のリスク: 正常ではない可能性のある心拍が一定数見られます。経過観察が推奨されます。"
    elif abnormal_beats > 0:
        risk_level = "low"
        summary_text = "低リスク: いくつか正常ではない拍動が見られますが、割合は低いです。"
    else:
        risk_level = "normal"
        summary_text = "診断結果: 検出された心拍は主に正常な範囲内です。"

    plt.figure(figsize=(15, 5))
    plot_duration_samples = min(len(ecg_signal), 3600)
    time_axis = np.arange(plot_duration_samples) / TARGET_FS
    plt.plot(time_axis, ecg_signal[:plot_duration_samples])
    
    colors = {1: 'orange', 2: 'red', 3: 'green'}
    added_labels = set()
    for i, pred_class in enumerate(predictions):
        r_peak_pos = r_peak_locations[i]
        if r_peak_pos < plot_duration_samples and pred_class in colors:
            label = f'Anomaly ({class_names[pred_class]})'
            if label not in added_labels:
                plt.axvline(x=r_peak_pos/TARGET_FS, color=colors[pred_class], linestyle='--', label=label)
                added_labels.add(label)
            else:
                plt.axvline(x=r_peak_pos/TARGET_FS, color=colors[pred_class], linestyle='--')

    plt.title(f'ECG Waveform (First {plot_duration_samples/TARGET_FS:.1f}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    if added_labels: plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        "counts": counts, 
        "plot_base64": plot_base64,
        "summary": {
            "abnormal_percentage": abnormal_percentage,
            "text": summary_text,
            "level": risk_level
        }
    }