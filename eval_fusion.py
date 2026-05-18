from ultralytics import YOLO
import torch
import gc
import os

def evaluate_fusion():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🚀 Використовуємо: {device}")

    model_path = 'diploma_experiments/method_b_fusion/weights/best.pt'

    if not os.path.exists(model_path):
        print(f"❌ Помилка: Не знайдено файл {model_path}")
        return

    print(f"📂 Завантаження ваг Fusion: {model_path}")
    model = YOLO(model_path)

    print("▶️ Починаємо розрахунок точності (Fusion mAP)...")

    metrics = model.val(
        data='llvip_fusion.yaml',
        split='test',
        imgsz=512,
        batch=8,
        conf=0.25,
        max_det=100,
        device=device,
        plots=True
    )

    print(f"\n☯️ Результати Fusion (Method B):")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

if __name__ == '__main__':
    evaluate_fusion()
