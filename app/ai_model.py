# app/ai_model.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from scipy.signal import butter, filtfilt, resample
import matplotlib.pyplot as plt
import io
import base64

# --- Matplotlibの日本語設定 ---
import matplotlib
matplotlib.use('Agg') # GUIバックエンド以外を使用する設定
plt.rcParams['font.family'] = 'IPAPGothic'

# --- 自前のR波検出アルゴリズム ---
def bandpass_filter(data, lowcut=5.0, highcut=15.0, fs=360, order=1):
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def pan_tompkins_detect(sig, fs):
    filtered_sig = bandpass_filter(sig, fs=fs)
    diff_sig = np.diff(filtered_sig)
    squared_sig = diff_sig**2
    window_size = int(0.150 * fs)
    integrated_sig = np.convolve(squared_sig, np.ones(window_size)/window_size, mode='same')
    
    qrs_peaks, search_radius = [], int(0.2 * fs)
    noise_peak, signal_peak, threshold = 0.0, 0.0, 0.0
    
    for i in range(len(integrated_sig)):
        if i > 0 and i < len(integrated_sig) - 1 and integrated_sig[i-1] < integrated_sig[i] and integrated_sig[i+1] < integrated_sig[i]:
            peak_val = integrated_sig[i]
            threshold = 0.125 * signal_peak + 0.875 * noise_peak
            if peak_val > threshold:
                qrs_peaks.append(i)
                signal_peak = 0.125 * peak_val + 0.875 * signal_peak
            else:
                noise_peak = 0.125 * peak_val + 0.875 * noise_peak
    
    if not qrs_peaks: return np.array([])
    final_peaks = [qrs_peaks[0]]
    for i in range(1, len(qrs_peaks)):
        if qrs_peaks[i] - final_peaks[-1] > search_radius:
            final_peaks.append(qrs_peaks[i])
    return np.array(final_peaks)

# --- モデル定義とロード ---
INPUT_SIZE, NUM_CLASSES = 288, 4
class ECG_CNN(nn.Module):
    # (中身は変更なし)
    def __init__(self, num_classes=NUM_CLASSES):
        super(ECG_CNN, self).__init__(); self.conv_layers = nn.Sequential(nn.Conv1d(1, 64, 16, 1, 8), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2), nn.Conv1d(64, 128, 8, 1, 4), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2, 2), nn.Conv1d(128, 256, 4, 1, 2), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2, 2)); self.fc_layers = nn.Sequential(nn.Linear(256 * (INPUT_SIZE // 8), 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes))
    def forward(self, x):
        x = self.conv_layers(x); x = x.view(x.size(0), -1); x = self.fc_layers(x); return x
device = torch.device("cpu"); model = ECG_CNN().to(device)
model_path = './models/ultimate_model.pth' # 'ultimate_model.pth' を優先
if not os.path.exists(model_path): model_path = './models/best_model_transfer_learning.pth'
if os.path.exists(model_path): model.load_state_dict(torch.load(model_path, map_location=device)); print(f"Successfully loaded model from {model_path}")
else: print(f"Error: Model file not found at {model_path}.")
model.eval()

# --- コア診断関数 ---
def run_diagnosis_pipeline(ecg_signal, fs):
    TARGET_FS, SEG_PRE, SEG_POST = 360, 108, 180
    if fs != TARGET_FS: ecg_signal = resample(ecg_signal, int(len(ecg_signal) * TARGET_FS / fs))
    r_peaks = pan_tompkins_detect(sig=ecg_signal.astype(np.float64), fs=TARGET_FS)
    if r_peaks.size == 0: return {"error": "心拍を検出できませんでした。"}
    segments, r_peak_locs = [], []; 
    for r_peak in r_peaks:
        if r_peak - SEG_PRE < 0 or r_peak + SEG_POST >= len(ecg_signal): continue
        segment = ecg_signal[r_peak - SEG_PRE : r_peak + SEG_POST]
        if np.std(segment) > 1e-6: segments.append((segment - np.mean(segment)) / np.std(segment)); r_peak_locs.append(r_peak)
    if not segments: return {"error": "有効な心拍セグメントを抽出できませんでした。"}
    with torch.no_grad(): _, predictions = torch.max(model(torch.tensor(np.expand_dims(np.array(segments, dtype=np.float32), 1), dtype=torch.float32).to(device)), 1)
    predictions = predictions.cpu().numpy()
    class_names = ['N (正常)', 'S (上室性)', 'V (心室性)', 'Q/F (その他)']; counts = {name: 0 for name in class_names}
    for pred in predictions: counts[class_names[pred]] += 1
    total_beats, abnormal_beats = len(predictions), len(predictions) - counts.get('N (正常)', 0)
    abnormal_percentage = (abnormal_beats / total_beats) * 100 if total_beats > 0 else 0
    s_beats, v_beats = counts.get('S (上室性)', 0), counts.get('V (心室性)', 0)
    if v_beats >= 5: risk_level, summary_text = "high", "注意: 心室性の異常な拍動が複数検出されました。専門医への相談を推奨します。"
    elif s_beats >= 10 or abnormal_percentage >= 20: risk_level, summary_text = "medium", "中程度のリスク: 正常ではない可能性のある心拍が一定数見られます。経過観察が推奨されます。"
    elif abnormal_beats > 0: risk_level, summary_text = "low", "低リスク: いくつか正常ではない拍動が見られますが、割合は低いです。"
    else: risk_level, summary_text = "normal", "診断結果: 検出された心拍は主に正常な範囲内です。"
    plt.figure(figsize=(15, 5)); plot_samples = min(len(ecg_signal), 3600); time_axis = np.arange(plot_samples) / TARGET_FS
    plt.plot(time_axis, ecg_signal[:plot_samples]); colors = {1: 'orange', 2: 'red', 3: 'green'}; added_labels = set()
    for i, p_cls in enumerate(predictions):
        r_pos = r_peak_locs[i]
        if r_pos < plot_samples and p_cls in colors:
            label = f'異常 ({class_names[p_cls].split(" ")[0]})'; 
            if label not in added_labels: plt.axvline(x=r_pos/TARGET_FS, color=colors[p_cls], linestyle='--', label=label); added_labels.add(label)
            else: plt.axvline(x=r_pos/TARGET_FS, color=colors[p_cls], linestyle='--')
    plt.title(f'心電図波形（最初の{plot_samples/TARGET_FS:.1f}秒）とAIによる異常検出'); plt.xlabel('時間 (秒)'); plt.ylabel('振幅')
    if added_labels: plt.legend()
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8'); plt.close()
    return {"counts": counts, "plot_base64": plot_base64, "summary": {"abnormal_percentage": abnormal_percentage, "text": summary_text, "level": risk_level}}

# --- ラッパー関数 ---
def diagnose_wfdb_record(record_path):
    try:
        signal, fields = wfdb.rdsamp(record_path); return run_diagnosis_pipeline(signal[:, 0], fields['fs'])
    except Exception as e: return {"error": f"WFDBファイルの処理エラー: {e}"}

def diagnose_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'signal' not in df.columns: return {"error": "CSVに'signal'列がありません。"}
        return run_diagnosis_pipeline(df['signal'].to_numpy(), fs=360)
    except Exception as e: return {"error": f"CSVファイルの処理エラー: {e}"}