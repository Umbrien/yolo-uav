import os
import shutil

base_path = "LLVIP"
old_img_dir = os.path.join(base_path, "visible")
new_img_dir = os.path.join(base_path, "images")

if os.path.exists(old_img_dir) and not os.path.exists(new_img_dir):
    print(f"Renaming {old_img_dir} -> {new_img_dir}")
    os.rename(old_img_dir, new_img_dir)
else:
    print("Folder 'images' already exists or 'visible' not found.")

if os.path.exists(os.path.join(base_path, "labels")):
    print("Folder 'labels' is in place")
else:
    print("Error: Folder 'labels' is missing! Run prepare_data.py again.")

yaml_content = f"""
path: {os.path.abspath(base_path)} # Full path
train: images/train
val: images/test
test: images/test

names:
  0: person
"""

with open("llvip_rgb.yaml", "w") as f:
    f.write(yaml_content)

print("YAML file updated. Now run the training.")
