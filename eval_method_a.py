from ultralytics import YOLO
import torch
import gc

def evaluate_safely():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    model_path = 'diploma_experiments/method_a_rgb/weights/best.pt'

    print(f"Loading weights: {model_path}")
    model = YOLO(model_path)

    print("Starting accuracy calculation (mAP)...")

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

    print(f"\nMethod A Results (RGB):")
    print(f"mAP50 (Accuracy): {metrics.box.map50:.4f}")
    print(f"mAP50-95 (Strict Accuracy): {metrics.box.map:.4f}")

if __name__ == '__main__':
    evaluate_safely()
