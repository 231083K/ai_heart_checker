import os
import requests
import pandas as pd
from tqdm import tqdm
import time

# --- 設定 ---
BASE_URL = 'https://physionet.org/files/ptb-xl/1.0.3/'
DATA_DIR = './data/ptb-xl/'
METADATA_FILES = ['ptbxl_database.csv', 'scp_statements.csv']

def download_file(url, local_path, pbar):
    """単一のファイルをダウンロードするヘルパー関数"""
    try:
        # 既にファイルが存在し、サイズが0より大きければスキップ
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            pbar.update(1)
            return "skipped"

        response = requests.get(url, timeout=60)
        response.raise_for_status()  # HTTPエラーがあれば例外を発生

        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        # 保存成功を確認
        if not (os.path.exists(local_path) and os.path.getsize(local_path) > 0):
            raise IOError(f"File not written correctly to {local_path}")
        
        pbar.update(1)
        return "success"
    except Exception as e:
        # print(f"\nFailed to download {url}. Reason: {e}")
        pbar.update(1)
        return "failed"

def robust_download_ptbxl():
    """
    PTB-XLデータセットを1ファイルずつ確実にダウンロードする。
    """
    print("--- Starting Robust Download Process for PTB-XL ---")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created local directory: {DATA_DIR}")

    # --- ステージ1: メタデータファイルのダウンロード ---
    print("\nStage 1: Downloading metadata files...")
    for filename in METADATA_FILES:
        url = f"{BASE_URL}{filename}"
        local_path = os.path.join(DATA_DIR, filename)
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(" -> Success.")
        except Exception as e:
            print(f" -> FAILED. Could not download metadata file: {filename}. Error: {e}")
            print("Cannot proceed without metadata. Exiting.")
            return
            
    # --- ステージ2: 全波形・ヘッダーファイルのダウンロード ---
    print("\nStage 2: Downloading all waveform and header files...")
    try:
        metadata_df = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_database.csv'))
    except FileNotFoundError:
        print("Error: ptbxl_database.csv not found. Stage 1 might have failed.")
        return

    # ダウンロードするファイル名のリストを作成 (高解像度と低解像度の両方)
    files_to_download = set(metadata_df['filename_hr'].tolist() + metadata_df['filename_lr'].tolist())
    
    success_count, failure_count, skip_count = 0, 0, 0
    start_time = time.time()

    with tqdm(total=len(files_to_download), unit='file', desc="Downloading ECG data") as pbar:
        for filename in files_to_download:
            url = f"{BASE_URL}{filename}"
            local_path = os.path.join(DATA_DIR, filename)
            
            result = download_file(url, local_path, pbar)
            if result == "success":
                success_count += 1
            elif result == "failed":
                failure_count += 1
            elif result == "skipped":
                skip_count += 1
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n--- PTB-XL Download Process Finished! ---")
    print(f"Total files attempted: {len(files_to_download)}")
    print(f"  Successfully downloaded: {success_count}")
    print(f"  Already existed (skipped): {skip_count}")
    print(f"  Failed to download: {failure_count}")
    print(f"Total time taken: {elapsed_time:.2f} seconds")

    if failure_count > 0:
        print("\nWARNING: Some files failed to download. You may want to run the script again.")
    else:
        print("\nAll files seem to be downloaded successfully!")

if __name__ == '__main__':
    robust_download_ptbxl()