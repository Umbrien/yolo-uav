import glob
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

BASE_DIR = "LLVIP"
RGB_DIR = os.path.join(BASE_DIR, "images")
THERMAL_DIR = os.path.join(BASE_DIR, "infrared")
OUTPUT_DIR = os.path.join(BASE_DIR, "fusion_early")


def create_fusion_dataset():
    for split in ["train", "test"]:
        out_path = os.path.join(OUTPUT_DIR, "images", split)
        os.makedirs(out_path, exist_ok=True)

        labels_out = os.path.join(OUTPUT_DIR, "labels", split)
        os.makedirs(labels_out, exist_ok=True)

        src_labels = os.path.join(BASE_DIR, "labels", split)
        print(f"Копіювання лейблів для {split}...")
        for label_file in glob.glob(os.path.join(src_labels, "*.txt")):
            shutil.copy(label_file, labels_out)

        print(f"Генерація Fusion зображень для {split}...")
        rgb_images = glob.glob(os.path.join(RGB_DIR, split, "*.jpg"))

        for rgb_path in tqdm(rgb_images):
            filename = os.path.basename(rgb_path)

            thermal_path = os.path.join(THERMAL_DIR, split, filename)

            if not os.path.exists(thermal_path):
                continue

            img_rgb = cv2.imread(rgb_path)
            img_thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)

            if img_rgb is None or img_thermal is None:
                continue

            img_rgb_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

            fused_img = cv2.merge([img_thermal, img_rgb_gray, img_thermal])

            out_file = os.path.join(out_path, filename)
            cv2.imwrite(out_file, fused_img)


if __name__ == "__main__":
    create_fusion_dataset()
    print("✅ Датасет для Методу B готовий!")
