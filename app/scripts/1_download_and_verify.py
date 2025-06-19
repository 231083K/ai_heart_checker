import wfdb
import matplotlib.pyplot as plt
import os

# データベース名 (PhysioNet上)
DB_NAME = 'mitdb'
# データを保存するローカルディレクトリ
DATA_DIR = './data/mit-bih-db'
# 今回、処理対象とするレコードのリスト
RECORDS_TO_CHECK = ['100', '101', '103', '200', '201', '203']

def main():
    """
    指定したレコードについて、データを読み込み（なければダウンロードし）、
    波形とアノテーションをプロットして検証する。
    """
    print("--- Phase 1.3: Data Collection & Preparation (Robust Method) ---")

    # データ保存用ディレクトリがなければ作成
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created local directory: {DATA_DIR}")

    # 2. 指定したレコードの読み込みと可視化による検証
    for record_name in RECORDS_TO_CHECK:
        print(f"\nProcessing record: {record_name}")

        try:
            # レコード（波形データ）を読み込み
            # ローカルのDATA_DIRにファイルがなければ、PhysioNetのpn_dirから自動ダウンロードされる
            record = wfdb.rdrecord(record_name, pn_dir=f'{DB_NAME}/1.0.0', local_dir=DATA_DIR)
            
            # アノテーション（心拍の診断情報）を読み込み
            annotation = wfdb.rdann(record_name, 'atr', pn_dir=f'{DB_NAME}/1.0.0', local_dir=DATA_DIR)

            print(f"Record {record_name} loaded successfully.")
            print(f"  - Sampling frequency: {record.fs} Hz")
            print(f"  - Signal length: {record.sig_len} samples")
            print(f"  - Number of annotations: {len(annotation.symbol)}")

            # 波形とアノテーションをプロット (最初の10秒間)
            fig, ax = plt.subplots(figsize=(15, 5))
            wfdb.plot_wfdb(record=record, annotation=annotation,
                           title=f'Record {record_name} from MIT-BIH DB (First 10 seconds)',
                           time_units='seconds', ax=[ax], plot_sym=True)
            ax.set_xlim(0, 10) # X軸を0秒から10秒に制限

            # プロットを画像ファイルとして保存
            output_filename = f'./data/record_{record_name}_plot.png'
            fig.savefig(output_filename)
            print(f"  - Plot saved to '{output_filename}'")
            plt.close(fig) # メモリ解放

        except Exception as e:
            print(f"Error processing record {record_name}: {e}")

if __name__ == '__main__':
    main()