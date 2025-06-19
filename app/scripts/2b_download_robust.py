import os
import requests
import time
from tqdm import tqdm

# --- 設定 ---
# PhysioNet上のデータベースのベースURL
BASE_URL = 'https://physionet.org/files/mitdb/1.0.0/'
# データを保存するローカルディレクトリ
DATA_DIR = './data/mit-bih-db'
# ダウンロード対象のレコードリスト
RECORD_LIST = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]
# 各レコードでダウンロードするファイルの拡張子
EXTENSIONS = ['.atr', '.dat', '.hea']

def robust_download():
    """
    requestsライブラリを使い、1ファイルずつダウンロードと保存確認を行う、
    最も堅牢なダウンロード処理。
    """
    print("--- Starting Robust Download Process ---")

    # データ保存用ディレクトリがなければ作成
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created local directory: {DATA_DIR}")
    
    # ダウンロードするファイルの総数を計算
    total_files = len(RECORD_LIST) * len(EXTENSIONS)
    print(f"Attempting to download {total_files} files for {len(RECORD_LIST)} records.")

    success_count = 0
    failure_count = 0
    skip_count = 0
    start_time = time.time()

    # tqdmを使って進捗バーを表示
    with tqdm(total=total_files, unit='file') as pbar:
        for record_name in RECORD_LIST:
            for ext in EXTENSIONS:
                file_name = f"{record_name}{ext}"
                pbar.set_description(f"Processing {file_name}")

                url = f"{BASE_URL}{file_name}"
                local_path = os.path.join(DATA_DIR, file_name)

                # 既にファイルが存在すればスキップ
                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                    pbar.update(1)
                    skip_count += 1
                    continue
                
                try:
                    # ネットワークからファイルをダウンロード
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()  # 4xx or 5xx エラーの場合は例外を発生

                    # ファイルをディスクに書き込み
                    with open(local_path, 'wb') as f:
                        f.write(response.content)
                    
                    # 保存成功を最終確認
                    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                        success_count += 1
                    else:
                        raise IOError(f"File '{file_name}' was not written to disk correctly.")

                except requests.exceptions.RequestException as e:
                    # ネットワークエラー
                    # print(f"\nFailed to download {file_name}. Reason: {e}")
                    failure_count += 1
                except IOError as e:
                    # ファイル書き込みエラー
                    # print(f"\nFailed to save {file_name}. Reason: {e}")
                    failure_count += 1
                
                pbar.update(1)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n--- Robust Download Process Finished! ---")
    print(f"Total files attempted: {total_files}")
    print(f"  Successfully downloaded: {success_count}")
    print(f"  Already existed (skipped): {skip_count}")
    print(f"  Failed to download/save: {failure_count}")
    print(f"Total time taken: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    robust_download()