import numpy as np
import os
from nuscenes.nuscenes import NuScenes
from ultralytics import YOLO
from pyquaternion import Quaternion
from tqdm import tqdm

# --- CONFIGURATION ---
DATAROOT = '/path/to/nuscenes/mini' 
MODEL_PATH = 'yolov8n.pt'  
CONF_THRESH = 0.25
# Distance thresholds for mAP (Standard nuScenes benchmarks)
MAP_THRESHOLDS = [1.0, 2.0, 4.0, 8.0] 
VALID_CLASSES = [2, 5, 7] # Car, Bus, Truck

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
model = YOLO(MODEL_PATH)

def calculate_iou_bev(pos1, pos2, size=(4.5, 1.8)):
    """Estimates IoU of two cars in BEV plane using standard car dimensions"""
    w, l = size
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    
    inter_w = max(0, w - dx)
    inter_l = max(0, l - dy)
    inter_area = inter_w * inter_l
    
    union_area = (2 * w * l) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def get_geometry_projection(pixel_coords, K, E_rot, E_trans):
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([pixel_coords[0], pixel_coords[1], 1.0])
    ray_ego = E_rot @ ray_cam
    if ray_ego[2] == 0: return None
    scale = -E_trans[2] / ray_ego[2]
    if scale < 0 or scale > 60: return None
    intersection = E_trans + scale * ray_ego
    return intersection[:2]

def evaluate():
    all_ious = []
    # Dictionary to store TPs at different distances for mAP
    tp_at_threshold = {t: 0 for t in MAP_THRESHOLDS}
    total_fps = 0
    total_gt = 0
    total_dets = 0

    samples_to_test = nusc.sample[:20] 
    print(f"🚀 Calculating All Metrics (mAP, IoU, F1) for {len(samples_to_test)} frames...")

    for sample in tqdm(samples_to_test):
        # 1. Align Ground Truth to Ego Frame
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        gt_boxes = nusc.get_boxes(sample['data']['LIDAR_TOP'])
        gt_coords = []
        for b in gt_boxes:
            if 'vehicle' in b.name:
                b.translate(-np.array(ego_pose['translation']))
                b.rotate(Quaternion(ego_pose['rotation']).inverse)
                gt_coords.append(b.center[:2])
        total_gt += len(gt_coords)

        # 2. Get AI Predictions
        pred_coords = []
        camera_list = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        for cam in camera_list:
            sd_record = nusc.get('sample_data', sample['data'][cam])
            cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            K = np.array(cs_record['camera_intrinsic'])
            E_rot = Quaternion(cs_record['rotation']).rotation_matrix
            E_trans = np.array(cs_record['translation'])
            
            img_path = os.path.join(DATAROOT, sd_record['filename'])
            if not os.path.exists(img_path): continue
            results = model(img_path, verbose=False, conf=CONF_THRESH)[0]
            for box in results.boxes:
                if int(box.cls[0]) in VALID_CLASSES:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    pos_3d = get_geometry_projection([(x1 + x2) / 2, y2], K, E_rot, E_trans)
                    if pos_3d is not None:
                        pred_coords.append(pos_3d)
                        total_dets += 1

        # 3. Matching and IoU
        matched_gt = set()
        for p in pred_coords:
            best_dist = float('inf')
            best_idx = -1
            
            for i, g in enumerate(gt_coords):
                dist = np.linalg.norm(p - g)
                if dist < best_dist and i not in matched_gt:
                    best_dist = dist
                    best_idx = i
            
            if best_idx != -1:
                # Check hits at every threshold for mAP calculation
                for t in MAP_THRESHOLDS:
                    if best_dist < t:
                        tp_at_threshold[t] += 1
                
                # If it's a "hit" at a reasonable distance, calculate IoU
                if best_dist < 4.0:
                    matched_gt.add(best_idx)
                    iou = calculate_iou_bev(p, gt_coords[best_idx])
                    all_ious.append(iou)
                else:
                    total_fps += 1
            else:
                total_fps += 1
                
    # Precision/Recall/F1/Accuracy at 4m (The "Fair" threshold)
    tp_final = tp_at_threshold[4.0]
    precision = tp_final / (tp_final + total_fps) if (tp_final + total_fps) > 0 else 0
    recall = tp_final / total_gt if total_gt > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = tp_final / (tp_final + total_fps + (total_gt - tp_final)) if total_gt > 0 else 0
    
    # mAP: Average of precision across all distance thresholds
    precisions_for_map = [tp_at_threshold[t] / (tp_at_threshold[t] + total_fps) for t in MAP_THRESHOLDS]
    mAP = np.mean(precisions_for_map)
    
    # mean IoU
    mIoU = np.mean(all_ious) if all_ious else 0

    print("\n" + "═"*45)
    print("      HACKATHON PERFORMANCE DASHBOARD")
    print("═"*45)
    print(f"• Accuracy:        {accuracy*100:.1f}%")
    print(f"• Precision:       {precision:.4f}")
    print(f"• Recall:          {recall:.4f}")
    print(f"• F1 Score:        {f1:.4f}")
    print(f"• mAP (Multi-Dist): {mAP:.4f}")
    print(f"• BEV IoU:         {mIoU:.4f}")
    print("═"*45)

if __name__ == "__main__":
    evaluate()
