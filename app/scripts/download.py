import wfdb
import matplotlib.pyplot as plt
import os

# データベース名とダウンロード先のディレクトリ
DB_NAME = 'mitdb'
DATA_DIR = './data/mit-bih-db'
RECORDS_TO_CHECK = ['100', '101', '103'] # 確認対象のレコード

def main():
    """
    MIT-BIH Arrhythmia Databaseをダウンロードし、
    指定したレコードの波形とアノテーションをプロットして検証する。
    """
    print("--- Phase 1.3: Data Collection & Preparation ---")

    # 1. データベースのダウンロード
    print(f"Downloading MIT-BIH Arrhythmia Database ('{DB_NAME}') to '{DATA_DIR}'...")
    # データベース全体をダウンロード（既に存在する場合はスキップされる）
    wfdb.dl_database(DB_NAME, DATA_DIR)
    print("Download complete.")

    # 2. レコードの読み込みと可視化による検証
    for record_name in RECORDS_TO_CHECK:
        print(f"\nVerifying record: {record_name}")
        record_path = os.path.join(DATA_DIR, record_name)

        try:
            # レコード（波形データ）を読み込み
            record = wfdb.rdrecord(record_path)
            # アノテーション（心拍の診断情報）を読み込み
            annotation = wfdb.rdann(record_path, 'atr')

            print(f"Record {record_name} loaded successfully.")
            print(f"  - Sampling frequency: {record.fs} Hz")
            print(f"  - Signal length: {record.sig_len} samples")
            print(f"  - Number of annotations: {len(annotation.symbol)}")

            # 波形とアノテーションをプロット
            fig, ax = plt.subplots(figsize=(15, 5))
            wfdb.plot_wfdb(record=record, annotation=annotation,
                           title=f'Record {record_name} from MIT-BIH DB',
                           time_units='seconds', ax=[ax]) # ax=[ax]で描画先を指定

            # プロットを画像ファイルとして保存
            output_filename = f'./data/record_{record_name}_plot.png'
            fig.savefig(output_filename)
            print(f"  - Plot saved to '{output_filename}'")
            plt.close(fig) # メモリ解放

        except Exception as e:
            print(f"Error processing record {record_name}: {e}")

if __name__ == '__main__':
    # dataディレクトリがなければ作成
    if not os.path.exists('./data'):
        os.makedirs('./data')
    main()