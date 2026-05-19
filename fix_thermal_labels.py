import os
import shutil
import glob
from tqdm import tqdm

def fix_thermal_labels():
    base_dir = "LLVIP"

    for cache_file in glob.glob(os.path.join(base_dir, 'infrared', '*', '*.cache')):
        print(f"Removing cache: {cache_file}")
        os.remove(cache_file)

    for split in ['train', 'test']:
        label_src_dir = os.path.join(base_dir, 'labels', split)
        thermal_img_dir = os.path.join(base_dir, 'infrared', split)

        print(f"Copying labels for {split} to infrared folder...")

        label_files = glob.glob(os.path.join(label_src_dir, "*.txt"))

        if not label_files:
            print(f"Warning: No labels found in {label_src_dir}!")
            continue

        for src in tqdm(label_files):
            filename = os.path.basename(src)
            dst = os.path.join(thermal_img_dir, filename)

            # Copying
            shutil.copy(src, dst)

    print("Done! Labels are now next to thermal photos.")

if __name__ == "__main__":
    fix_thermal_labels()
