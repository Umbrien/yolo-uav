from ultralytics import YOLO
import os
import torch
import gc

def run_thermal_experiment():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Experiment 2 (Thermal Only) on device: {device}")

    model = YOLO('yolov13n.pt')
    yaml_path = os.path.abspath('llvip_thermal.yaml')

    try:
        print("Starting training on THERMAL data...")
        results = model.train(
            data=yaml_path,
            epochs=5,
            imgsz=512,
            batch=8,
            project='diploma_experiments',
            name='method_thermal_only',
            device=device,
            workers=0,
            exist_ok=True,
            amp=False,
            val=False,
            plots=False
        )

        print("Training completed! Validation...")

        best_model = YOLO(os.path.join(results.save_dir, 'weights', 'last.pt'))

        metrics = best_model.val(
            split='test',
            imgsz=512,
            batch=8,
            conf=0.25,
            max_det=100,
            device=device
        )

        print(f"\nThermal Results:")
        print(f"mAP50: {metrics.box.map50:.4f}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    run_thermal_experiment()
