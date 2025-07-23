import os
import requests
import pandas as pd
from tqdm import tqdm
import time

# --- 設定 ---
BASE_URL = 'https://physionet.org/files/ptb-xl/1.0.3/'
DATA_DIR = './data/ptb-xl/'
METADATA_FILES = ['ptbxl_database.csv', 'scp_statements.csv']
RETRY_COUNT = 3
RETRY_DELAY = 3

def download_file(url, local_path, pbar):
    """単一のファイルをダウンロードするヘルパー関数"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        pbar.set_postfix_str(f"{os.path.basename(local_path)} Skipped")
        pbar.update(1)
        return "skipped"

    for attempt in range(RETRY_COUNT):
        try:
            response = requests.get(url, headers=headers, timeout=60, stream=True)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                pbar.set_postfix_str(f"{os.path.basename(local_path)} Success")
                pbar.update(1)
                return "success"
            else:
                raise IOError("File not written correctly")

        except Exception as e:
            if attempt >= RETRY_COUNT - 1:
                pbar.set_postfix_str(f"Error on {os.path.basename(local_path)}")
                pbar.write(f"\nFAILED for {os.path.basename(local_path)} after {RETRY_COUNT} retries. URL: {url}, Error: {type(e).__name__}")
            time.sleep(RETRY_DELAY)
    
    pbar.update(1)
    return "failed"


def final_robust_download_ptbxl():
    """
    CSVから読み込んだベース名に、手動で拡張子を追加してURLを生成する最終版。
    """
    print("--- Starting Final Robust Download Process for PTB-XL ---")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    try:
        metadata_df = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_database.csv'))
    except FileNotFoundError:
        print("Error: 'ptbxl_database.csv' not found.")
        return

    files_to_download = []
    for index, row in metadata_df.iterrows():
        lr_base = row['filename_lr']  
        hr_base = row['filename_hr']  
        
        files_to_download.append(f"{lr_base}.dat")
        files_to_download.append(f"{lr_base}.hea")
        files_to_download.append(f"{hr_base}.dat")
        files_to_download.append(f"{hr_base}.hea")

    files_to_download = sorted(list(set(files_to_download)))
    print(f"Total unique files to download: {len(files_to_download)}")

    with tqdm(total=len(files_to_download), unit='file', desc="Downloading PTB-XL") as pbar:
        for filename in files_to_download:
            url = f"{BASE_URL}{filename}"
            
            dir_part = os.path.dirname(filename)
            local_dir = os.path.join(DATA_DIR, dir_part)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            
            local_path = os.path.join(DATA_DIR, filename)
            
            # ダウンロード実行
            download_file(url, local_path, pbar)

    print("\n--- PTB-XL Download Process Finished! ---")
    print("Please check the 'data/ptb-xl' directory.")

if __name__ == '__main__':
    final_robust_download_ptbxl()