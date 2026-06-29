import os
import random
import shutil


source_directory = ''
output_directory = ''

# Destination folder names
dest_data_in = os.path.join(output_directory, 'data_in')
dest_val_out = os.path.join(output_directory, 'val_out')
dest_test_out = os.path.join(output_directory, 'test_out')

num_data_in = 40
num_val_out = 52


def split_and_move_folders():
    """
    Randomly assign device subfolders to enrolled (data_in), validation outlier (val_out),
    and test outlier (test_out) sets, then move them to the corresponding destinations.
    """
    print(f"Processing source directory: {source_directory}")

    if not os.path.isdir(source_directory):
        print(f"Error: Source folder '{source_directory}' does not exist. Please check the path.")
        return

    try:
        all_subfolders = [d for d in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, d))]
        total_folders = len(all_subfolders)
        print(f"Found {total_folders} subfolders.")

        if total_folders != 144:
            print(f"Warning: Expected 144 folders, found {total_folders}. Proceeding with actual count.")

        if total_folders < num_data_in + num_val_out:
            print(f"Error: Not enough folders ({total_folders}) to fill data_in ({num_data_in}) and val_out ({num_val_out}).")
            return

    except OSError as e:
        print(f"Error: Could not read source directory contents. Reason: {e}")
        return

    # Shuffle folder list randomly
    random.shuffle(all_subfolders)
    print("Subfolder list shuffled randomly.")

    # Create destination directories if they do not exist
    print("Creating destination directories...")
    os.makedirs(dest_data_in, exist_ok=True)
    os.makedirs(dest_val_out, exist_ok=True)
    os.makedirs(dest_test_out, exist_ok=True)
    print(f" - {dest_data_in}")
    print(f" - {dest_val_out}")
    print(f" - {dest_test_out}")

    # Assign folders to each split
    folders_for_data = all_subfolders[:num_data_in]
    folders_for_val = all_subfolders[num_data_in : num_data_in + num_val_out]
    folders_for_test = all_subfolders[num_data_in + num_val_out:]

    print("\nAssignment plan:")
    print(f" - {len(folders_for_data)} folders -> data_in (enrolled)")
    print(f" - {len(folders_for_val)} folders -> val_out (validation outliers)")
    print(f" - {len(folders_for_test)} folders -> test_out (test outliers)")

    def move_folders(folder_list, destination_path):
        print(f"\n--- Moving folders to {destination_path} ---")
        for folder_name in folder_list:
            source_path = os.path.join(source_directory, folder_name)
            destination = os.path.join(destination_path, folder_name)
            try:
                shutil.move(source_path, destination)
                print(f"  Moved: {folder_name}")
            except Exception as e:
                print(f"  Failed to move: {folder_name}. Error: {e}")
        print(f"--- Done moving to {destination_path} ---")

    move_folders(folders_for_data, dest_data_in)
    move_folders(folders_for_val, dest_val_out)
    move_folders(folders_for_test, dest_test_out)

    print("\nAll operations complete. Folders have been randomly assigned and moved.")

# --- 2. Run script ---
if __name__ == "__main__":
    split_and_move_folders()
