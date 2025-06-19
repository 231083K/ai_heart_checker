import wfdb
import os
import time

# データベース名 (PhysioNet上)
DB_NAME = 'mitdb'
# データを保存するローカルディレクトリ
DATA_DIR = './data/mit-bih-db'

# MIT-BIH Arrhythmia Database の全48レコードのリスト
RECORD_LIST = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

def download_all_records():
    """
    os.chdirを使い、カレントディレクトリを変更することで保存場所を制御する、
    最も互換性の高い方法で全レコードをダウンロードする。
    """
    print(f"--- Starting full download for database: {DB_NAME} (os.chdir method) ---")
    print(f"Targeting {len(RECORD_LIST)} records.")

    # データ保存用ディレクトリがなければ作成
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created local directory: {DATA_DIR}")

    success_count = 0
    failure_count = 0
    start_time = time.time()
    
    # 元の作業ディレクトリを記憶
    original_cwd = os.getcwd()

    # 全レコードに対してループ処理
    for i, record_name in enumerate(RECORD_LIST):
        print(f"\nProcessing record: {record_name} ({i + 1}/{len(RECORD_LIST)})...")
        
        try:
            # ダウンロード先ディレクトリに移動
            os.chdir(DATA_DIR)
            
            # レコードとアノテーションを読み込み（引数を最小限にする）
            # この場合、ファイルはカレントディレクトリ（DATA_DIR）にダウンロードされる
            record = wfdb.rdrecord(record_name, pn_dir=DB_NAME)
            annotation = wfdb.rdann(record_name, 'atr', pn_dir=DB_NAME)
            
            # 読み込み成功の確認
            assert len(record.p_signal) > 0
            assert len(annotation.symbol) > 0

            print(f"  -> SUCCESS: Record '{record_name}' downloaded and verified.")
            success_count += 1
        except Exception as e:
            # エラーが発生した場合はスキップ
            print(f"  -> FAILED to process record '{record_name}'. Skipping.")
            print(f"     Error details: {e}")
            failure_count += 1
        finally:
            # 必ず元の作業ディレクトリに戻る
            os.chdir(original_cwd)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n--- Download process finished! ---")
    print(f"Total records attempted: {len(RECORD_LIST)}")
    print(f"  Successfully downloaded: {success_count}")
    print(f"  Failed or skipped: {failure_count}")
    print(f"Total time taken: {elapsed_time:.2f} seconds")
    print(f"All data is located in: {os.path.abspath(DATA_DIR)}")

if __name__ == '__main__':
    download_all_records()