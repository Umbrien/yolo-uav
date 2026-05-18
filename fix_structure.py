import os
import shutil

base_path = "LLVIP"
old_img_dir = os.path.join(base_path, "visible")
new_img_dir = os.path.join(base_path, "images")

if os.path.exists(old_img_dir) and not os.path.exists(new_img_dir):
    print(f"Перейменування {old_img_dir} -> {new_img_dir}")
    os.rename(old_img_dir, new_img_dir)
else:
    print("Папка 'images' вже існує або 'visible' не знайдена.")

if os.path.exists(os.path.join(base_path, "labels")):
    print("Папка 'labels' на місці ✅")
else:
    print("❌ Папка 'labels' відсутня! Запустіть prepare_data.py знову.")

yaml_content = f"""
path: {os.path.abspath(base_path)} # Повний шлях
train: images/train
val: images/test
test: images/test

names:
  0: person
"""

with open("llvip_rgb.yaml", "w") as f:
    f.write(yaml_content)

print("YAML файл оновлено ✅. Тепер запускайте тренування.")
