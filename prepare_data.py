import os
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm

BASE_DIR = "LLVIP"
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "Annotations")
IMAGES_ROOT = os.path.join(BASE_DIR, "visible")

CLASSES = ['person']

def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def convert_annotation(image_id, output_dir):
    in_file = open(os.path.join(ANNOTATIONS_DIR, f'{image_id}.xml'))
    out_file = open(os.path.join(output_dir, f'{image_id}.txt'), 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in CLASSES or int(difficult) == 1:
            continue
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert_box((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    in_file.close()
    out_file.close()

def process_dataset():
    for split in ['train', 'test']:
        img_dir = os.path.join(IMAGES_ROOT, split)
        label_dir = os.path.join(BASE_DIR, 'labels', split)

        os.makedirs(label_dir, exist_ok=True)

        print(f"Обробка {split} даних...")
        image_files = glob.glob(os.path.join(img_dir, '*.jpg'))

        for img_path in tqdm(image_files):
            image_id = os.path.splitext(os.path.basename(img_path))[0]

            xml_path = os.path.join(ANNOTATIONS_DIR, f'{image_id}.xml')
            if os.path.exists(xml_path):
                convert_annotation(image_id, label_dir)

if __name__ == "__main__":
    print("Починаємо конвертацію анотацій LLVIP у формат YOLO...")
    process_dataset()
    print("Готово! Папка 'labels' створена.")
