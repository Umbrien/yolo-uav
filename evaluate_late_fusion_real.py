import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics, box_iou
from ultralytics.utils import ops
from ensemble_boxes import weighted_boxes_fusion

RGB_MODEL_PATH = 'diploma_experiments/method_a_rgb/weights/best.pt'
THERMAL_MODEL_PATH = 'diploma_experiments/method_thermal_only/weights/best.pt'
TEST_IMAGES_RGB = 'LLVIP/images/test'
TEST_IMAGES_TH = 'LLVIP/infrared/test'
TEST_LABELS = 'LLVIP/labels/test'
IMG_SIZE = 640

def load_gt(label_path):
    if not os.path.exists(label_path):
        return np.array([])
    with open(label_path, 'r') as f:
        lines = f.readlines()
    gt = []
    for line in lines:
        cls, x, y, w, h = map(float, line.split())
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        gt.append([cls, x1, y1, x2, y2])
    return np.array(gt)

def match_predictions(pred_classes, true_classes, iou, iouv):
    """Match predictions to ground truth objects using IoU."""
    # Dx10 matrix, where D - detections, 10 - IoU thresholds
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
    # LxD matrix where L - labels (rows), D - detections (columns)
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class
    iou = iou.cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.nonzero(iou >= threshold)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

def run_evaluation():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if torch.cuda.is_available(): device = 'cuda'
    print(f"Using device: {device}")

    model_rgb = YOLO(RGB_MODEL_PATH)
    model_th = YOLO(THERMAL_MODEL_PATH)

    img_files = sorted(glob.glob(os.path.join(TEST_IMAGES_RGB, "*.jpg")))

    weights = [1, 1]
    iou_thr = 0.5
    skip_box_thr = 0.0001

    pbar = tqdm(img_files, desc="Evaluating Late Fusion")

    all_fused_preds = []
    all_gt = []

    for img_path in pbar:
        img_name = os.path.basename(img_path)
        label_path = os.path.join(TEST_LABELS, img_name.replace('.jpg', '.txt'))
        th_img_path = os.path.join(TEST_IMAGES_TH, img_name)

        res_rgb = model_rgb(img_path, verbose=False, imgsz=IMG_SIZE, device=device)[0]
        res_th = model_th(th_img_path, verbose=False, imgsz=IMG_SIZE, device=device)[0]

        boxes_rgb = res_rgb.boxes.xyxyn.cpu().numpy().tolist()
        scores_rgb = res_rgb.boxes.conf.cpu().numpy().tolist()
        labels_rgb = res_rgb.boxes.cls.cpu().numpy().tolist()

        boxes_th = res_th.boxes.xyxyn.cpu().numpy().tolist()
        scores_th = res_th.boxes.conf.cpu().numpy().tolist()
        labels_th = res_th.boxes.cls.cpu().numpy().tolist()

        boxes_list = [boxes_rgb, boxes_th]
        scores_list = [scores_rgb, scores_th]
        labels_list = [labels_rgb, labels_th]

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )

        gt = load_gt(label_path)

        t_gt_boxes = torch.zeros((0, 4))
        t_gt_cls = torch.zeros((0,))
        if len(gt) > 0:
            t_gt_boxes = torch.from_numpy(gt[:, 1:]).float().to(device)
            t_gt_cls = torch.from_numpy(gt[:, 0]).long().to(device)

        if len(fused_boxes) == 0:
            t_pred = torch.zeros((0, 6)).to(device)
        else:
            t_pred = torch.zeros((len(fused_boxes), 6)).to(device)
            t_pred[:, :4] = torch.from_numpy(fused_boxes).float().to(device)
            t_pred[:, 4] = torch.from_numpy(fused_scores).float().to(device)
            t_pred[:, 5] = torch.from_numpy(fused_labels).long().to(device)

        all_fused_preds.append(t_pred)
        all_gt.append({'boxes': t_gt_boxes, 'cls': t_gt_cls})

    print("\nCalculating final metrics...")
    metrics = DetMetrics(names={0: 'person'})
    iouv = torch.linspace(0.5, 0.95, 10).to(device)

    tp_list = []
    conf_list = []
    pred_cls_list = []
    target_cls_list = []

    for i in range(len(all_fused_preds)):
        pred = all_fused_preds[i]
        gt = all_gt[i]

        target_cls = gt['cls']
        target_boxes = gt['boxes']

        if len(target_cls) == 0:
            if len(pred) > 0:
                tp = torch.zeros((len(pred), 10), dtype=torch.bool).to(device)
                tp_list.append(tp)
                conf_list.append(pred[:, 4])
                pred_cls_list.append(pred[:, 5])
            continue

        if len(pred) == 0:
            target_cls_list.append(target_cls)
            continue

        iou = box_iou(target_boxes, pred[:, :4])
        tp = match_predictions(pred[:, 5], target_cls, iou, iouv)

        tp_list.append(tp)
        conf_list.append(pred[:, 4])
        pred_cls_list.append(pred[:, 5])
        target_cls_list.append(target_cls)

    if len(tp_list) > 0 or len(target_cls_list) > 0:
        tp = torch.cat(tp_list, 0) if tp_list else torch.zeros((0, 10), dtype=torch.bool).to(device)
        conf = torch.cat(conf_list, 0) if conf_list else torch.zeros((0,)).to(device)
        pred_cls = torch.cat(pred_cls_list, 0) if pred_cls_list else torch.zeros((0,)).to(device)
        target_cls = torch.cat(target_cls_list, 0) if target_cls_list else torch.zeros((0,)).to(device)

        metrics.update_stats({
            "tp": tp.cpu().numpy(),
            "conf": conf.cpu().numpy(),
            "pred_cls": pred_cls.cpu().numpy(),
            "target_cls": target_cls.cpu().numpy(),
            "target_img": np.unique(target_cls.cpu().numpy())
        })

        metrics.process()

        print(f"\nLATE FUSION RESULTS:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
    else:
        print("No detections or ground truth found.")

if __name__ == '__main__':
    run_evaluation()
