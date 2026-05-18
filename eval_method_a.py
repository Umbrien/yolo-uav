from ultralytics import YOLO
import torch
import gc

def evaluate_safely():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🚀 Використовуємо: {device}")

    model_path = 'diploma_experiments/method_a_rgb/weights/best.pt'

    print(f"📂 Завантаження ваг: {model_path}")
    model = YOLO(model_path)

    print("▶️ Починаємо розрахунок точності (mAP)...")

    metrics = model.val(
        data='llvip_rgb.yaml',
        split='test',
        imgsz=512,
        batch=8,
        conf=0.25,
        max_det=100,
        device=device,
        plots=True
    )

    print(f"\n✅ Результати Методу А (RGB):")
    print(f"mAP50 (Точність): {metrics.box.map50:.4f}")
    print(f"mAP50-95 (Сувора точність): {metrics.box.map:.4f}")

if __name__ == '__main__':
    evaluate_safely()
