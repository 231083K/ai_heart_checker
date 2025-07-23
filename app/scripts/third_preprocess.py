import wfdb
import os
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, filtfilt, resample
from collections import Counter

def bandpass_filter(data, lowcut=5.0, highcut=15.0, fs=360, order=1):
    nyquist = 0.5 * fs; low = lowcut / nyquist; high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band'); return filtfilt(b, a, data)
def pan_tompkins_detect(sig, fs):
    filtered_sig = bandpass_filter(sig, fs=fs); diff_sig = np.diff(filtered_sig); squared_sig = diff_sig**2
    window_size = int(0.150 * fs); integrated_sig = np.convolve(squared_sig, np.ones(window_size)/window_size, mode='same')
    qrs_peaks, search_radius = [], int(0.2 * fs); noise_peak, signal_peak, threshold = 0.0, 0.0, 0.0
    for i in range(len(integrated_sig)):
        if i > 0 and i < len(integrated_sig) - 1 and integrated_sig[i-1] < integrated_sig[i] and integrated_sig[i+1] < integrated_sig[i]:
            peak_val = integrated_sig[i]; threshold = 0.125 * signal_peak + 0.875 * noise_peak
            if peak_val > threshold: qrs_peaks.append(i); signal_peak = 0.125 * peak_val + 0.875 * signal_peak
            else: noise_peak = 0.125 * peak_val + 0.875 * noise_peak
    if not qrs_peaks: return np.array([])
    final_peaks = [qrs_peaks[0]]
    for i in range(1, len(qrs_peaks)):
        if qrs_peaks[i] - final_peaks[-1] > search_radius: final_peaks.append(qrs_peaks[i])
    return np.array(final_peaks)

# --- 設定 ---
SVDB_DATA_DIR = './data/svdb'
PROCESSED_DATA_DIR = './data/processed'
TARGET_FS = 360
SEG_PRE = 108
SEG_POST = 180

def preprocess_svdb():
    print("--- Preprocessing SVDB for S-class specialist model (Resampling fix) ---")
    if not os.path.exists(PROCESSED_DATA_DIR): os.makedirs(PROCESSED_DATA_DIR)

    try:
        record_list = wfdb.get_record_list('svdb')
    except Exception as e:
        record_list = sorted(list(set([f.split('.')[0] for f in os.listdir(SVDB_DATA_DIR) if f.endswith('.hea')])))
        print(f"get_record_list failed, found {len(record_list)} records by scanning directory.")

    all_segments, all_labels = [], []
    
    for record_name in tqdm(record_list, desc="Processing SVDB"):
        record_path = os.path.join(SVDB_DATA_DIR, record_name)
        try:
            signal, fields = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
        except Exception: continue
        
        ecg_signal_original = signal[:, 0].astype(np.float64)
        original_fs = fields['fs']

        if original_fs != TARGET_FS:
            num_samples_target = int(len(ecg_signal_original) * TARGET_FS / original_fs)
            ecg_signal = resample(ecg_signal_original, num_samples_target)
        else:
            ecg_signal = ecg_signal_original

        r_peaks = pan_tompkins_detect(sig=ecg_signal, fs=TARGET_FS)
        
        for r_peak in r_peaks:
            ann_indices = np.where(np.abs(annotation.sample * (TARGET_FS / original_fs) - r_peak) < TARGET_FS / 2)[0]
            if len(ann_indices) == 0: continue
            symbol = annotation.symbol[ann_indices[0]]
            
            label = -1
            if symbol in ['N', 'L', 'R', 'e', 'j']: label = 0 
            elif symbol in ['A', 'a', 'J', 'S']: label = 1 
            else: continue
            
            if r_peak - SEG_PRE < 0 or r_peak + SEG_POST >= len(ecg_signal): continue
            segment = ecg_signal[r_peak - SEG_PRE : r_peak + SEG_POST]
            if np.std(segment) > 1e-6:
                segment = (segment - np.mean(segment)) / np.std(segment)
                all_segments.append(segment); all_labels.append(label)
    
    if not all_labels:
        print("\nFinished. No segments were created. Please check the source data and script logic.")
        return

    X = np.array(all_segments)
    y = np.array(all_labels)

    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train_svdb_NS.npy'), X)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train_svdb_NS.npy'), y)
    print(f"\nFinished. Created {len(all_segments)} segments for N-S classification.")
    print(f"Class distribution: {sorted(Counter(y).items())}")

if __name__ == '__main__':
    preprocess_svdb()