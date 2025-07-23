import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, resample
from tqdm import tqdm

def pan_tompkins_detect(sig, fs):
    # 1. バンドパスフィルタ
    low = 5.0 / (fs / 2)
    high = 15.0 / (fs / 2)
    b, a = butter(1, [low, high], btype='band')
    filtered_sig = filtfilt(b, a, sig)
    
    # 2. 微分
    diff_sig = np.diff(filtered_sig)
    
    # 3. 二乗
    squared_sig = diff_sig**2
    
    # 4. 移動窓積分
    window_size = int(0.150 * fs)
    integrated_sig = np.convolve(squared_sig, np.ones(window_size)/window_size, mode='same')
    
    # 5. ピーク検出
    qrs_peaks_indices = []
    search_radius = int(0.2 * fs)
    
    i = 0
    while i < len(integrated_sig):
        window = integrated_sig[i:i+search_radius]
        if len(window) == 0:
            break
        
        peak_idx = np.argmax(window)
        if window[peak_idx] > 0.5 * np.mean(integrated_sig):
            qrs_peaks_indices.append(i + peak_idx)
            i += search_radius
        else:
            i += 1
            
    return np.array(qrs_peaks_indices)

# --- 設定 ---
PTBXL_DATA_DIR = './data/ptb-xl/'
PROCESSED_DATA_DIR = './data/processed/'
TARGET_FS = 360
SEG_PRE = 108
SEG_POST = 180

def final_preprocess_ptbxl():
    print("--- Preprocessing PTB-XL Dataset (Final Self-Contained Method) ---")
    if not os.path.exists(PROCESSED_DATA_DIR): os.makedirs(PROCESSED_DATA_DIR)

    try:
        metadata = pd.read_csv(os.path.join(PTBXL_DATA_DIR, 'ptbxl_database.csv'), index_col='ecg_id')
    except FileNotFoundError:
        print(f"Error: 'ptbxl_database.csv' not found.")
        return

    all_segments, all_labels = [], []

    for ecg_id, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing PTB-XL"):
        file_path_100hz = os.path.join(PTBXL_DATA_DIR, row['filename_lr'])

        try:
            signal_100hz, fields = wfdb.rdsamp(file_path_100hz)
        except Exception:
            continue
        
        fs = fields['fs']
        num_samples_target = int(signal_100hz.shape[0] * TARGET_FS / fs)
        signal_target_fs = resample(signal_100hz, num_samples_target)
        ecg_signal = signal_target_fs[:, 0].astype(np.float64)

        r_peaks = pan_tompkins_detect(sig=ecg_signal, fs=TARGET_FS)
        
        if len(r_peaks) == 0:
            continue

        is_normal = 'NORM' in str(row['scp_codes'])
        label = 0 if is_normal else 1
        
        for r_peak in r_peaks:
            if r_peak - SEG_PRE < 0 or r_peak + SEG_POST >= len(ecg_signal):
                continue
            
            segment = ecg_signal[r_peak - SEG_PRE : r_peak + SEG_POST]
            if np.std(segment) > 1e-6:
                segment = (segment - np.mean(segment)) / np.std(segment)
                all_segments.append(segment)
                all_labels.append(label)

    X = np.array(all_segments, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    print(f"\nFinished processing PTB-XL.")
    print(f"    Total segments created: {len(X)}")
    
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_pretrain_ptbxl.npy'), X)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_pretrain_ptbxl.npy'), y)
    print(f"    Saved pre-training data to {PROCESSED_DATA_DIR}")

if __name__ == '__main__':
    final_preprocess_ptbxl()