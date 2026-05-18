from ultralytics import YOLO
import os
import torch
import gc

def run_method_a():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🚀 Пристрій: {device}")

    model = YOLO('yolov13n.pt')
    yaml_path = os.path.abspath('llvip_rgb.yaml')

    try:
        print("▶️ Починаємо тренування (без проміжної валідації)...")
        results = model.train(
            data=yaml_path,
            epochs=5,
            imgsz=512,
            batch=8,
            project='diploma_experiments',
            name='method_a_rgb',
            device=device,
            workers=0,
            exist_ok=True,
            amp=False,
            val=False,
            save=True,
            plots=False
        )

        print("✅ Тренування завершено! Виконуємо фінальну перевірку...")

        best_model = YOLO(os.path.join(results.save_dir, 'weights', 'last.pt'))

        metrics = best_model.val(
            split='test',
            imgsz=512,
            batch=8,
            conf=0.25,
            max_det=100,
            device=device
        )

        print(f"🏆 Результат mAP50: {metrics.box.map50}")

    except Exception as e:
        print(f"❌ Помилка: {e}")

if __name__ == '__main__':
    run_method_a()
