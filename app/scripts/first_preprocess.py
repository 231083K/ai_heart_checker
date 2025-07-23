import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
from collections import Counter
from sklearn.model_selection import train_test_split

# --- グローバル設定 ---
# データディレクトリ
DATA_DIR = './data/mit-bih-db'
# MIT-BIH Arrhythmia Database の全48レコードのリスト
RECORD_LIST = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]
# 保存先ディレクトリ
PROCESSED_DATA_DIR = './data/processed'

# --- 前処理パラメータ ---
# サンプリング周波数 (MIT-BIHは360Hz)
FS = 360
# R波を中心に切り出すセグメントの、R波より前のサンプル数
SEG_PRE = 108  # 0.3秒
# R波の後のサンプル数
SEG_POST = 180 # 0.5秒
# セグメントの全長
SEG_LEN = SEG_PRE + SEG_POST # 288サンプル (0.8秒)

# --- アノテーションのマッピング ---
# AAMI規格に準拠したクラス分類
# N: Normal, S: Supraventricular, V: Ventricular, F: Fusion, Q: Unknown
# 今回はN, S, V, その他(Q)の4クラスに分類する
AAMI_MAP = {
    'N': ['N', 'L', 'R', 'e', 'j'],
    'S': ['A', 'a', 'J', 'S'],
    'V': ['V', 'E'],
    'F': ['F'],
    'Q': ['/', 'f', 'Q']
}
# マッピング辞書を逆引きしやすい形に変換
SYMBOL_TO_CLASS = {symbol: class_name for class_name, symbols in AAMI_MAP.items() for symbol in symbols}
CLASS_TO_INT = {'N': 0, 'S': 1, 'V': 2, 'Q': 3, 'F': 3} # FもQ(その他)にマージ

# --- フィルタリング関数 ---
def bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=360, order=4):
    """バンドパスフィルタを適用する関数"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# --- メイン処理関数 ---
def preprocess_data():
    """データ前処理のメインパイプライン"""
    print("--- Phase 2: Data Preprocessing & Feature Engineering ---")

    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        print(f"Created directory: {PROCESSED_DATA_DIR}")

    # ステップ1: 患者（レコード）単位でデータを分割
    # これにより、同じ患者のデータが訓練とテストに混入するのを防ぐ
    train_records, test_records = train_test_split(RECORD_LIST, test_size=0.2, random_state=42)
    train_records, val_records = train_test_split(train_records, test_size=0.2, random_state=42)

    print(f"Train records ({len(train_records)}): {train_records}")
    print(f"Validation records ({len(val_records)}): {val_records}")
    print(f"Test records ({len(test_records)}): {test_records}")

    # ステップ2: 各データセット（訓練/検証/テスト）を生成
    for split_name, records in [('train', train_records), ('val', val_records), ('test', test_records)]:
        print(f"\nProcessing {split_name} set...")
        
        all_segments = []
        all_labels = []

        for record_name in records:
            print(f"  - Reading record: {record_name}")
            
            # データの読み込み
            record_path = os.path.join(DATA_DIR, record_name)
            signal_data = wfdb.rdrecord(record_path).p_signal[:, 0] # 最初のチャンネル(MLII)のみ使用
            annotation = wfdb.rdann(record_path, 'atr')

            # ノイズ除去
            filtered_signal = bandpass_filter(signal_data)

            # 心拍セグメントの切り出し
            qrs_indices = annotation.sample
            beat_symbols = annotation.symbol
            
            for i, r_peak in enumerate(qrs_indices):
                # セグメントが信号の範囲外に出ないかチェック
                if r_peak - SEG_PRE < 0 or r_peak + SEG_POST >= len(filtered_signal):
                    continue
                
                # アノテーションが分類対象かチェック
                symbol = beat_symbols[i]
                if symbol not in SYMBOL_TO_CLASS:
                    continue
                
                # セグメントの切り出し
                segment = filtered_signal[r_peak - SEG_PRE : r_peak + SEG_POST]
                
                # Zスコア正規化
                segment = (segment - np.mean(segment)) / np.std(segment)
                
                all_segments.append(segment)
                all_labels.append(CLASS_TO_INT[SYMBOL_TO_CLASS[symbol]])

        # Numpy配列に変換
        X = np.array(all_segments)
        y = np.array(all_labels)
        
        # データの形状とクラス分布を表示
        print(f"  Finished processing {split_name} set.")
        print(f"    X_{split_name} shape: {X.shape}")
        print(f"    y_{split_name} shape: {y.shape}")
        print(f"    Class distribution: {sorted(Counter(y).items())}")
        
        # データをファイルに保存
        np.save(os.path.join(PROCESSED_DATA_DIR, f'X_{split_name}.npy'), X)
        np.save(os.path.join(PROCESSED_DATA_DIR, f'y_{split_name}.npy'), y)
        print(f"    Saved {split_name} data to {PROCESSED_DATA_DIR}")

    print("\n--- Preprocessing complete! ---")

if __name__ == '__main__':
    preprocess_data()