from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "article_assets"
SAMPLE_ID = "240280"


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def save_channel_example() -> None:
    rgb_path = ROOT / "LLVIP" / "images" / "test" / f"{SAMPLE_ID}.jpg"
    thermal_path = ROOT / "LLVIP" / "infrared" / "test" / f"{SAMPLE_ID}.jpg"

    rgb = read_rgb(rgb_path)
    thermal = read_gray(thermal_path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    tgt = cv2.merge([thermal, gray, thermal])

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.0), constrained_layout=True)
    panels = [
        ("RGB frame", rgb, None),
        ("Thermal channel T", thermal, "inferno"),
        ("Gray channel", gray, "gray"),
        ("T-G-T tensor", tgt, None),
    ]

    for ax, (title, image, cmap) in zip(axes, panels):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.suptitle("Example of Thermal-Gray-Thermal formation on LLVIP pair", fontsize=14)
    fig.savefig(OUT_DIR / "fig_tgt_example.png", dpi=220)
    plt.close(fig)


def save_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.4, 3.0, 2.0, 1.0, "RGB\nframe"),
        (0.4, 1.0, 2.0, 1.0, "IR/Thermal\nframe"),
        (3.1, 3.0, 2.0, 1.0, "Gray =\n0.299R+0.587G+0.114B"),
        (5.9, 2.0, 2.0, 1.0, "T-G-T\n[T, Gray, T]"),
        (8.5, 2.0, 1.8, 1.0, "YOLOv13n"),
        (10.8, 2.0, 1.0, 1.0, "bbox\nperson"),
    ]

    for x, y, w, h, label in boxes:
        rect = plt.Rectangle((x, y), w, h, linewidth=1.8, edgecolor="#243b53", facecolor="#f4f7fb")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)

    arrows = [
        ((2.4, 3.5), (3.1, 3.5)),
        ((2.4, 1.5), (5.9, 2.35)),
        ((5.1, 3.5), (5.9, 2.65)),
        ((7.9, 2.5), (8.5, 2.5)),
        ((10.3, 2.5), (10.8, 2.5)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#243b53"})

    ax.text(6.9, 0.6, "Early Fusion: detector architecture remains unchanged, only the input tensor changes.", ha="center", fontsize=11)
    fig.savefig(OUT_DIR / "fig_pipeline_tgt.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_channel_example()
    save_pipeline()
    print(f"Saved figures to {OUT_DIR}")
