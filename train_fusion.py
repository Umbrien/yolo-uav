from ultralytics import YOLO
import os
import torch
import gc

def run_fusion_training():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🚀 Експеримент 3 (Fusion/Method B) на: {device}")

    model = YOLO('yolov13n.pt')
    yaml_path = os.path.abspath('llvip_fusion.yaml')

    try:
        print("▶️ Починаємо тренування FUSION моделі...")
        results = model.train(
            data=yaml_path,
            epochs=5,
            imgsz=512,
            batch=8,
            project='diploma_experiments',
            name='method_b_fusion',
            device=device,
            workers=0,
            exist_ok=True,
            amp=False,
            val=False,
            plots=False
        )

        print("✅ Тренування Fusion завершено! Фінальна перевірка...")

        best_model = YOLO(os.path.join(results.save_dir, 'weights', 'last.pt'))

        metrics = best_model.val(
            split='test',
            imgsz=512,
            batch=8,
            conf=0.25,
            max_det=100,
            device=device
        )

        print(f"\n☯️ Результати Fusion (Method B):")
        print(f"mAP50: {metrics.box.map50:.4f}")

    except Exception as e:
        print(f"❌ Помилка: {e}")

if __name__ == '__main__':
    run_fusion_training()
