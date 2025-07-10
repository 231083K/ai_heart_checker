import wfdb, os
DB_NAME = 'svdb'
DATA_DIR = f'./data/{DB_NAME}'
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
wfdb.dl_database(DB_NAME, DATA_DIR)
print(f"Downloaded SVDB to {DATA_DIR}")