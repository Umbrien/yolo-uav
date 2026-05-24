import glob
import logging
import os
import time

import torch
from ultralytics import YOLO


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "inference_run.log")

    logger = logging.getLogger("inference_job")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="w")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger, log_file


def main():
    output_dir = "imgs-to-test/results"
    logger, log_file = setup_logging(output_dir)

    logger.info("Starting inference job...")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    logger.info(f"Using compute device: {device}")

    rgb_model_path = "diploma_experiments/method_a_rgb/weights/best.pt"
    thermal_model_path = "diploma_experiments/method_thermal_only/weights/best.pt"

    if not os.path.exists(rgb_model_path):
        logger.warning(f"RGB model weights not found at {rgb_model_path}")
        rgb_model = None
    else:
        logger.info(f"Loading RGB model from {rgb_model_path}...")
        try:
            rgb_model = YOLO(rgb_model_path)
            logger.info("RGB model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load RGB model: {e}")
            rgb_model = None

    if not os.path.exists(thermal_model_path):
        logger.warning(f"Thermal model weights not found at {thermal_model_path}")
        thermal_model = None
    else:
        logger.info(f"Loading Thermal model from {thermal_model_path}...")
        try:
            thermal_model = YOLO(thermal_model_path)
            logger.info("Thermal model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Thermal model: {e}")
            thermal_model = None

    input_dir = "imgs-to-test"
    if not os.path.exists(input_dir):
        logger.error(f"Input directory '{input_dir}' does not exist.")
        return

    image_paths = sorted(
        glob.glob(os.path.join(input_dir, "*.jpg"))
        + glob.glob(os.path.join(input_dir, "*.png"))
    )
    if not image_paths:
        logger.warning(f"No test images found in directory '{input_dir}'.")
        return

    logger.info(
        f"Found {len(image_paths)} image(s) to process: {[os.path.basename(p) for p in image_paths]}"
    )

    start_time = time.time()
    processed_count = 0

    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        logger.info(f"[{idx}/{len(image_paths)}] Processing image: {filename}")

        if rgb_model is not None:
            logger.info(f"  Running RGB model on {filename}...")
            run_start = time.time()
            try:
                res = rgb_model(img_path, conf=0.25, device=device, verbose=False)[0]
                elapsed = time.time() - run_start
                num_detections = len(res.boxes)
                logger.info(
                    f"    RGB model finished in {elapsed:.3f}s. Detected {num_detections} person(s)."
                )

                out_path = os.path.join(output_dir, f"{name}_rgb_detected{ext}")
                res.save(filename=out_path)
                logger.info(f"    Saved result to {out_path}")
            except Exception as e:
                logger.error(f"    Error running RGB model on {filename}: {e}")

        if thermal_model is not None:
            logger.info(f"  Running Thermal model on {filename}...")
            run_start = time.time()
            try:
                res = thermal_model(img_path, conf=0.25, device=device, verbose=False)[
                    0
                ]
                elapsed = time.time() - run_start
                num_detections = len(res.boxes)
                logger.info(
                    f"    Thermal model finished in {elapsed:.3f}s. Detected {num_detections} person(s)."
                )

                out_path = os.path.join(output_dir, f"{name}_thermal_detected{ext}")
                res.save(filename=out_path)
                logger.info(f"    Saved result to {out_path}")
            except Exception as e:
                logger.error(f"    Error running Thermal model on {filename}: {e}")

        processed_count += 1

    total_elapsed = time.time() - start_time
    logger.info(
        f"Job completed! Processed {processed_count} image(s) in {total_elapsed:.2f}s."
    )
    logger.info(f"Logs saved to {log_file}")
    logger.info(f"Output images saved to '{output_dir}/' directory.")


if __name__ == "__main__":
    main()
