import cv2
import os
import glob
import random
from ultralytics import YOLO

RGB_MODEL_PATH = 'diploma_experiments/method_a_rgb/weights/best.pt'
THERMAL_MODEL_PATH = 'diploma_experiments/method_thermal_only/weights/best.pt'
SOURCE_IMAGES = 'LLVIP/images/test'
OUTPUT_DIR = 'diploma_experiments/method_c_visuals'

def visualize_ensemble():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Завантаження моделей...")
    model_rgb = YOLO(RGB_MODEL_PATH)
    model_thermal = YOLO(THERMAL_MODEL_PATH)

    all_images = glob.glob(os.path.join(SOURCE_IMAGES, "*.jpg"))
    selected_images = random.sample(all_images, 10)

    print(f"Обробка {len(selected_images)} зображень...")

    for img_path in selected_images:
        filename = os.path.basename(img_path)

        res_rgb = model_rgb(img_path, conf=0.25, verbose=False)[0]

        thermal_path = img_path.replace('images', 'infrared')
        res_thermal = model_thermal(thermal_path, conf=0.25, verbose=False)[0]

        img = cv2.imread(img_path)

        for box in res_rgb.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "RGB", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for box in res_thermal.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img, (x1+2, y1+2), (x2-2, y2-2), (0, 0, 255), 2)
            cv2.putText(img, "Thermal", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        save_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(save_path, img)
        print(f"Збережено: {save_path}")

    print(f"\n✅ Готово! Результати в папці: {OUTPUT_DIR}")

if __name__ == '__main__':
    visualize_ensemble()
