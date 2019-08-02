"""
Copys data which is relevant for us to a new folder
"""

import shutil
import os

source_dir = "E:/ScanNet/scans/"
dest_dir = "E:/ScanNet/relevant/"
for subdir, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith("_vh_clean_2.ply") or file.endswith("_vh_clean_2.labels.ply"):
            print(f"copy file: {subdir + '/' + file}")
            print(f"to: {dest_dir + file}")
            shutil.copy(subdir + '/' + file, dest_dir + file)
            print("done")
