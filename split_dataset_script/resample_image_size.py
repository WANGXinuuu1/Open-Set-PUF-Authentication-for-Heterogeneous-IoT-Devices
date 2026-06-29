import os
from PIL import Image

# --- Configuration ---
# Root directory of the original images
INPUT_ROOT = ''
# Root directory for saving the new images
OUTPUT_ROOT = ''

# Target image resolution
NEW_WIDTH = 50
NEW_HEIGHT = 50
NEW_SIZE = (NEW_WIDTH, NEW_HEIGHT)
# Number of pixels to extract (50x50 = 2500)
PIXEL_COUNT_TO_EXTRACT = NEW_WIDTH * NEW_HEIGHT

# Expected resolution of original images (used for validation, optional)
EXPECTED_ORIGINAL_SIZE = (200, 220)
# ---------------------

def process_single_image(input_path, output_path):
    """
    Process a single image file: extract the first PIXEL_COUNT_TO_EXTRACT pixels
    and reshape into a NEW_WIDTH x NEW_HEIGHT grayscale image.
    """
    filename = os.path.basename(input_path)
    try:
        img = Image.open(input_path).convert('L')
        if img.size != EXPECTED_ORIGINAL_SIZE:
            print(f"[WARNING] Skipping {filename}: size mismatch ({img.size} != {EXPECTED_ORIGINAL_SIZE})")
            return
        pixel_data = list(img.getdata())
        new_pixel_data = pixel_data[:PIXEL_COUNT_TO_EXTRACT]

        if len(new_pixel_data) != PIXEL_COUNT_TO_EXTRACT:
            print(f"[ERROR] Skipping {filename}: insufficient pixels ({len(new_pixel_data)} found)")
            raise RuntimeError(f"Processing failed: {filename} has insufficient pixels")

        new_img = Image.new('L', NEW_SIZE)
        new_img.putdata(new_pixel_data)
        new_img.save(output_path)
        print(f"[OK] Saved: {os.path.relpath(output_path, OUTPUT_ROOT)}")

    except Exception as e:
        print(f"[FAILED] Error processing {filename}: {e}")

def batch_process_all_subdirs():
    """
    Recursively traverse all subdirectories under INPUT_ROOT and process image files.
    The full folder structure is preserved in the output directory.
    """
    print(f"--- Starting batch processing ---")
    print(f"Input root:  {INPUT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print("-" * 30)

    for root, dirs, files in os.walk(INPUT_ROOT):
        relative_path = os.path.relpath(root, INPUT_ROOT)

        if relative_path == ".":
            current_output_dir = OUTPUT_ROOT
        else:
            current_output_dir = os.path.join(OUTPUT_ROOT, relative_path)

        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]

        if image_files:
            try:
                os.makedirs(current_output_dir, exist_ok=True)
                if relative_path != ".":
                    print(f"\n[DIR] Created/confirmed: {relative_path}")
            except Exception as e:
                print(f"[ERROR] Could not create output directory {relative_path}: {e}")
                continue

            for filename in image_files:
                input_path = os.path.join(root, filename)
                output_path = os.path.join(current_output_dir, filename)
                process_single_image(input_path, output_path)

    print("-" * 30)
    print("All images processed.")

if __name__ == "__main__":
    batch_process_all_subdirs()
