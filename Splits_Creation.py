# splits_creation.py
import gcsfs
import numpy as np
import os
from tqdm import tqdm

fs = gcsfs.GCSFileSystem(token='google_default')
preproc_bucket = "clinimcl-data/OASIS3/preprocessed_new"
splits_bucket = "clinimcl-data/OASIS3/splits_new"

np.random.seed(42)

all_files = [f for f in fs.ls(preproc_bucket) if f.endswith('.pt')]
total_scans = len(all_files)
assert total_scans >= 4116, f"Expected at least 4116 scans, found {total_scans}"
np.random.shuffle(all_files)
train_files = all_files[:2800]
test_files = all_files[2800:3640] # 2800 + 840
val_files = all_files[3640:4116]  # Remaining 476

def save_split(files, split_name, parts):
    per_folder = len(files) // parts
    for i in tqdm(range(parts), desc=f"Uploading {split_name} split explicitly"):
        start_idx = i * per_folder
        end_idx = start_idx + per_folder if i < parts - 1 else len(files)
        files_chunk = files[start_idx:end_idx]
        folder = f"{splits_bucket}/{split_name}/part_{i+1:02d}"
        for file_path in files_chunk:
            file_name = os.path.basename(file_path)
            new_file_path = f"{folder}/{file_name}"
            with fs.open(file_path, 'rb') as src, fs.open(new_file_path, 'wb') as dst:
                dst.write(src.read())

save_split(train_files, "train_new", 14)        # 200 files per folder
save_split(test_files, "test_new", 14)          # 60 files per folder
save_split(val_files, "validation_new", 14)     # 34 files per folder (last folder may vary slightly)
print("Splits creation and uploading explicitly complete.")