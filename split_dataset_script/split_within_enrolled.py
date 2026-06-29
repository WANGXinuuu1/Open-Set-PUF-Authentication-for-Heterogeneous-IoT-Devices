import os
import shutil
import random
from pathlib import Path

# --- 1. Path configuration ---
# Base directory
BASE_DIR = Path('')
# Source data directory (contains all class subfolders)
SOURCE_DIR = BASE_DIR / "data_in"
# Destination base directory (will create 'closed' folder structure)
DEST_BASE_DIR = BASE_DIR / "closed"

# --- 2. Split count configuration ---
TRAIN_COUNT = 61
TEST_COUNT = 20
VAL_COUNT = 20
TOTAL_COUNT = TRAIN_COUNT + TEST_COUNT + VAL_COUNT

# Split names
SPLITS = {
    "train": TRAIN_COUNT,
    "test": TEST_COUNT,
    "val": VAL_COUNT
}

def copy_files(file_list: list[Path], dest_dir: Path):
    """Copy all files in file_list to dest_dir, preserving file metadata."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src_file in file_list:
        try:
            shutil.copy2(src_file, dest_dir / src_file.name)
        except Exception as e:
            print(f"Warning: Failed to copy {src_file.name}: {e}")

def split_and_copy_dataset():
    """Main function: iterate over class folders, split files, and copy to train/val/test."""
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Destination root: {DEST_BASE_DIR}")

    if not SOURCE_DIR.exists():
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
        return

    # Create top-level train/test/val directories
    for split_name in SPLITS.keys():
        (DEST_BASE_DIR / split_name).mkdir(parents=True, exist_ok=True)

    print(f"Created top-level directories: {list(SPLITS.keys())}")

    # Iterate over all class subdirectories under SOURCE_DIR
    for class_dir in SOURCE_DIR.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            print(f"\n--- Processing class: {class_name} ---")

            file_list = [f for f in class_dir.iterdir() if f.is_file()]
            file_count = len(file_list)
            print(f"Total files found: {file_count}")

            if file_count < TOTAL_COUNT:
                print(f"Warning: Insufficient files ({file_count} < {TOTAL_COUNT}). Skipping this class.")
                continue

            random.shuffle(file_list)

            current_index = 0
            train_files = file_list[current_index : current_index + TRAIN_COUNT]
            current_index += TRAIN_COUNT
            test_files = file_list[current_index : current_index + TEST_COUNT]
            current_index += TEST_COUNT
            val_files = file_list[current_index : current_index + VAL_COUNT]

            if len(train_files) != TRAIN_COUNT or len(test_files) != TEST_COUNT or len(val_files) != VAL_COUNT:
                print("Error: Split counts do not match expected values. Skipping.")
                continue

            dest_train = DEST_BASE_DIR / "train" / class_name
            print(f"  Copying {len(train_files)} files to {dest_train} ...")
            copy_files(train_files, dest_train)

            dest_test = DEST_BASE_DIR / "test" / class_name
            print(f"  Copying {len(test_files)} files to {dest_test} ...")
            copy_files(test_files, dest_test)

            dest_val = DEST_BASE_DIR / "val" / class_name
            print(f"  Copying {len(val_files)} files to {dest_val} ...")
            copy_files(val_files, dest_val)

            print(f"Class {class_name} done.")

    print("\n--- All classes processed ---")

if __name__ == "__main__":
    split_and_copy_dataset()
