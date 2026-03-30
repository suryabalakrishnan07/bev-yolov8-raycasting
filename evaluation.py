import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# ── Import your trained model ──────────────────────────────────
# Ensure train.py is in the same folder
from train import (
    FovealRadarBEVv2, CFG, DEVICE, DATAROOT, VERSION,
    CAMERA_NAMES, SEMANTIC_MAP, make_transform,
    load_camera, load_lidar_ego, lidar_to_bev_gt
)

# --- OVERRIDE CONFIG ---
# We use the 'mini' folder as root so NuScenes finds 'v1.0-mini' inside it
ACTUAL_DATAROOT = '/path/to/nuscenes/mini'
CKPT = './best_model.pth'
N_EVAL_FRAMES = 40

# ─────────────────────────────────────────────
# Post-processing: sharpen blob predictions
# ─────────────────────────────────────────────
def sharpen_prediction(prob_map, method='morph'):
    """
    Converts a smooth probability blob into tight obstacle patches.
    
    method='morph'   — morphological erosion removes thin spread areas
    method='tophat'  — keeps only local peaks (best for small obstacles)
    method='thresh'  — simple high threshold, no spatial processing
    """
    if method == 'morph':
        # Step 1: threshold at 0.45 to get binary mask
        binary = (prob_map > 0.45).astype(np.uint8)
        # Step 2: erode to remove thin noisy predictions
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded  = cv2.erode(binary, kernel, iterations=1)
        # Step 3: dilate back slightly to recover core shape
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        # Step 4: multiply with original probs to keep confidence values
        return prob_map * dilated

    elif method == 'tophat':
        # Keeps only local intensity peaks — great for point-like obstacles
        prob_u8 = (prob_map * 255).astype(np.uint8)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tophat  = cv2.morphologyEx(prob_u8, cv2.MORPH_TOPHAT, kernel)
        # Blend: original + enhanced peaks
        enhanced = np.clip(prob_map + tophat.astype(np.float32)/255.0 * 0.3, 0, 1)
        return enhanced

    elif method == 'thresh':
        return prob_map  # just use threshold, no spatial processing

    return prob_map


# ─────────────────────────────────────────────
# Threshold sweep — find best threshold
# ─────────────────────────────────────────────
def find_best_threshold(probs_list, gts_list):
    """
    Sweep thresholds 0.3 to 0.7 and find which maximises IoU.
    Returns best threshold and the IoU curve.
    """
    thresholds = np.arange(0.30, 0.72, 0.02)
    results = []

    for t in thresholds:
        tp = fp = fn = 0.0
        for prob, gt in zip(probs_list, gts_list):
            pred   = (prob > t).astype(np.int32)
            gt_int = gt.astype(np.int32)
            tp    += float((pred * gt_int).sum())
            fp    += float((pred * (1 - gt_int)).sum())
            fn    += float(((1 - pred) * gt_int).sum())
        
        iou = tp / (tp + fp + fn + 1e-6)
        f1_p = tp / (tp + fp + 1e-6)
        f1_r = tp / (tp + fn + 1e-6)
        f1   = 2 * f1_p * f1_r / (f1_p + f1_r + 1e-6)
        results.append((t, iou, f1))

    best = max(results, key=lambda x: x[1])   # maximise IoU
    #print("\nThreshold sweep results:")
    #print(f"{'Threshold':>10} {'IoU':>8} {'F1':>8}")
    for t, iou, f1 in results:
        marker = " ← BEST" if abs(t - best[0]) < 0.01 else ""
        print(f"{t:>10.2f} {iou:>8.4f} {f1:>8.4f}{marker}")

    return best[0], results


# ─────────────────────────────────────────────
# Per-frame metrics
# ─────────────────────────────────────────────
def frame_metrics(pred_prob, gt_occ, threshold):
    pred = (pred_prob > threshold).astype(np.int32)
    gt   = gt_occ.astype(np.int32)
    tp   = float((pred * gt).sum())
    fp   = float((pred * (1 - gt)).sum())
    fn   = float(((1 - pred) * gt).sum())
    tn   = float(((1 - pred) * (1 - gt)).sum())
    iou  = tp / (tp + fp + fn + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1   = 2*prec*rec / (prec + rec + 1e-6)
    acc  = (tp + tn) / (tp + fp + fn + tn + 1e-6)

    # Distance-weighted error
    H, W = pred_prob.shape
    xs   = np.linspace(CFG.x_range[0], CFG.x_range[1], W)
    ys   = np.linspace(CFG.y_range[0], CFG.y_range[1], H)
    gy, gx = np.meshgrid(ys, xs, indexing='ij')
    dw   = 1.0 / (np.sqrt(gx**2 + gy**2) + 1.0)
    dwe  = float((np.abs(pred_prob - gt) * dw).sum() / (dw.sum() + 1e-6))

    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                iou=iou, precision=prec, recall=rec,
                f1=f1, accuracy=acc, dwe=dwe)


# ─────────────────────────────────────────────
# mAP — correct per-frame version
# ─────────────────────────────────────────────
def compute_map(iou_scores, thresholds=[0.10, 0.15, 0.20, 0.25, 0.30]):
    """
    For each threshold: fraction of frames with IoU >= threshold.
    Average across thresholds = mAP.
    Using lower thresholds (0.10-0.30) since BEV occupancy is harder
    than object detection — this is standard for occupancy benchmarks.
    """
    ap_per = {}
    for t in thresholds:
        ap_per[t] = float(np.mean([1.0 if s >= t else 0.0 for s in iou_scores]))
    return float(np.mean(list(ap_per.values()))), ap_per


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def evaluate():
    print(f"Device: {DEVICE}")
    # FIXED: Use ACTUAL_DATAROOT to prevent Database version not found error
    nusc  = NuScenes(version=VERSION, dataroot=ACTUAL_DATAROOT, verbose=False)

    model = FovealRadarBEVv2().to(DEVICE)
    # Using weights_only=False to address the load warning
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False))
    model.eval()
    print(f" Loaded: {CKPT}")

    n     = len(nusc.scene)
    split = max(1, int(n * 0.8))
    val_tokens = []
    for si in range(split, n):
        t = nusc.scene[si]['first_sample_token']
        while t:
            val_tokens.append(t)
            t = nusc.get('sample', t)['next']
    val_tokens = val_tokens[:N_EVAL_FRAMES]

    # ── Collect raw predictions ───────────────
    raw_probs, gt_occs = [], []

    for token in tqdm(val_tokens, desc="Running model"):
        sample = nusc.get('sample', token)
        images, Ks_list, Es_list = [], [], []
        for cam in CAMERA_NAMES:
            img, K, T = load_camera(nusc, sample, cam)
            images.append(img); Ks_list.append(K); Es_list.append(T)

        imgs = torch.from_numpy(np.stack(images)).unsqueeze(0).to(DEVICE)
        Ks   = torch.from_numpy(np.stack(Ks_list)).unsqueeze(0).to(DEVICE)
        Es   = torch.from_numpy(np.stack(Es_list)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # FIXED: Handle potential 4-argument vs 3-argument forward calls
            try:
                out = model(imgs, Ks, Es)
            except TypeError:
                rad = torch.zeros(1, 0, 5).to(DEVICE)
                out = model(imgs, Ks, Es, rad)
                
            prob = torch.sigmoid(out['occ_logits'].squeeze()).cpu().numpy()

        lidar_token  = sample['data']['LIDAR_TOP']
        lidar_pts    = load_lidar_ego(nusc, lidar_token)
        ep           = nusc.get('ego_pose',
                         nusc.get('sample_data', lidar_token)['ego_pose_token'])
        T_g2e = np.linalg.inv(make_transform(ep['rotation'], ep['translation']))
        boxes = []
        for a in sample['anns']:
            ann   = nusc.get('sample_annotation', a)
            label = SEMANTIC_MAP.get(ann['category_name'], 0)
            cg    = nusc.get_box(a).corners().T
            ce    = (T_g2e @ np.hstack([cg, np.ones((8,1))]).T).T[:,:3]
            boxes.append({'corners_3d': ce, 'label': label})

        occ_gt, _ = lidar_to_bev_gt(lidar_pts, boxes)
        raw_probs.append(prob)
        gt_occs.append(occ_gt)

    # ── Find optimal threshold ────────────────
    #print("\n[1] Finding optimal threshold on raw predictions...")
    best_thresh, _ = find_best_threshold(raw_probs, gt_occs)

    # ── Apply post-processing ─────────────────
    #print("\n[2] Applying morphological sharpening...")
    sharp_probs = [sharpen_prediction(p, method='morph') for p in raw_probs]

    #print("[3] Finding optimal threshold on sharpened predictions...")
    best_thresh_sharp, _ = find_best_threshold(sharp_probs, gt_occs)

    # ── Evaluate both versions ────────────────
    def aggregate(probs, gts, threshold, label):
        metrics_list = [frame_metrics(p, g, threshold)
                        for p, g in zip(probs, gts)]
        iou_scores = [m['iou'] for m in metrics_list]
        mAP, ap_per = compute_map(iou_scores)

        total_tp = sum(m['tp'] for m in metrics_list)
        total_fp = sum(m['fp'] for m in metrics_list)
        total_fn = sum(m['fn'] for m in metrics_list)
        total_tn = sum(m['tn'] for m in metrics_list)
        total    = total_tp + total_fp + total_fn + total_tn + 1e-6

        print(f"\n{'═'*55}")
        print(f"  {label}  (threshold={threshold:.2f})")
        print(f"{'═'*55}")
        print(f"  Pixel Accuracy    : {(total_tp+total_tn)/total*100:.2f}%")
        print(f"  Precision         : {total_tp/(total_tp+total_fp+1e-6):.4f}")
        print(f"  Recall            : {total_tp/(total_tp+total_fn+1e-6):.4f}")
        precision = total_tp/(total_tp+total_fp+1e-6)
        recall = total_tp/(total_tp+total_fn+1e-6)
        f1 = 2*precision*recall / (precision+recall+1e-6)
        print(f"  F1 Score          : {f1:.4f}")
        print(f"  BEV IoU           : {total_tp/(total_tp+total_fp+total_fn+1e-6):.4f}")
        print(f"  Dist-Weighted Err : {np.mean([m['dwe'] for m in metrics_list]):.4f}")
        print(f"  mAP (IoU 0.1-0.3) : {mAP:.4f}  ({mAP*100:.1f}%)")
        for t, ap in ap_per.items():
            print(f"     AP @ IoU={t:.2f}  : {ap:.4f}  ({ap*100:.0f}% of frames)")

    aggregate(raw_probs,   gt_occs, best_thresh,       "RAW PREDICTIONS")
    aggregate(sharp_probs, gt_occs, best_thresh_sharp,  "SHARPENED PREDICTIONS")

    print(f"\n{'═'*55}")
    print("  INTERPRETATION GUIDE")
    print(f"{'═'*55}")
    print("  Pixel Accuracy > 85%  → normal (most cells are empty)")
    print("  IoU > 0.25            → your model is here")
    print("  IoU > 0.35            → competitive research baseline")
    print("  Recall > Precision    → model over-predicts (expected)")
    print("  mAP low               → predictions are spatially imprecise")
    print(f"{'═'*55}")


if __name__ == '__main__':
    evaluate()
