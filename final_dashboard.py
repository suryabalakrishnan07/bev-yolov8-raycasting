import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon, Circle
from ultralytics import YOLO
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

# Importing your specific setup from train.py
from train import CFG, NuScenesBEVDataset, DEVICE, DATAROOT, VERSION

# --- CLASS CONFIG ---
CLASS_MAP = {
    0: {'name': 'Ped', 'color': 'yellow', 'size': (0.8, 0.8)},
    2: {'name': 'Car', 'color': 'lime', 'size': (1.9, 4.2)},
    5: {'name': 'Bus', 'color': 'cyan', 'size': (2.8, 12.0)},
    7: {'name': 'Truck', 'color': 'orange', 'size': (2.6, 9.0)}
}

def project_to_ground_direct(u, v, K, E):
    """Direct Geometric Projection using Dataset Matrices."""
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([u, v, 1.0])
    R, t = E[:3, :3], E[:3, 3]
    R_z, t_z = R[2, :], t[2]
    
    denom = np.dot(R_z, ray_cam)
    if abs(denom) < 0.001: return None
    
    scale = -t_z / denom
    if scale < 1.0 or scale > 75.0: return None # Distance Filter
    
    p_ego = scale * (R @ ray_cam) + t
    return p_ego[0], p_ego[1]

def generate_final_dashboard(scene_no=1, frame_idx=20):
    print(f"Device: {DEVICE} | 🚀 Starting Final Fusion Engine...")
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    ds = NuScenesBEVDataset(nusc, [scene_no], augment=False)
    yolo_model = YOLO('yolov8n.pt').to(DEVICE)

    sample_data = ds[frame_idx]
    imgs, Ks, Es = sample_data['images'], sample_data['intrinsics'], sample_data['extrinsics']
    
    fig = plt.figure(figsize=(26, 12), facecolor='black')
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 3.5])
    plt.style.use('dark_background')

    all_dets = []
    cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    for i, name in enumerate(cam_names):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
        img_u8 = (img_np * 255).astype(np.uint8)
        
        results = yolo_model(img_u8, verbose=False, conf=0.35)[0]
        ax.imshow(img_np)
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in CLASS_MAP:
                cfg = CLASS_MAP[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=cfg['color'], linewidth=2))
                
                pos = project_to_ground_direct((x1+x2)/2, y2, Ks[i].numpy(), Es[i].numpy())
                if pos:
                    all_dets.append({'pos': pos, 'cfg': cfg})

        ax.set_title(f"{name}", color='cyan', fontsize=10)
        ax.axis('off')

    # --- BEV RADAR (THE FIX IS HERE) ---
    ax_bev = fig.add_subplot(gs[:, 3])
    
    # Corrected Record Fetching
    scene_rec = nusc.scene[scene_no] # This is already the dictionary!
    log_rec = nusc.get('log', scene_rec['log_token']) # Pass the token string, not the dict
    nmap = NuScenesMap(dataroot=DATAROOT, map_name=log_rec['location'])
    
    sample_token = ds.samples[frame_idx]
    sample_rec = nusc.get('sample', sample_token)
    sd_rec = nusc.get('sample_data', sample_rec['data']['CAM_FRONT'])
    ego_pose = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    
    patch_angle = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0] * 180 / np.pi 
    patch_box = (ego_pose['translation'][0], ego_pose['translation'][1], 102.4, 102.4)
    map_mask = nmap.get_map_mask(patch_box, patch_angle, ['drivable_area', 'walkway'], (200, 200))
    
    ax_bev.imshow(map_mask[0], extent=CFG.x_range + CFG.y_range, cmap='bone', alpha=0.3)
    ax_bev.imshow(np.ma.masked_where(map_mask[1] == 0, map_mask[1]), extent=CFG.x_range + CFG.y_range, cmap='winter', alpha=0.15)

    # BEV-Clustering
    merged = []
    for d in all_dets:
        if not any(np.linalg.norm(np.array(d['pos']) - np.array(m['pos'])) < 3.0 for m in merged):
            merged.append(d)

    for obj in merged:
        x, y = obj['pos']; w, l = obj['cfg']['size']
        corners = np.array([[x-w/2, y-l/2], [x+w/2, y-l/2], [x+w/2, y+l/2], [x-w/2, y+l/2]])
        ax_bev.add_patch(Polygon(corners, closed=True, color=obj['cfg']['color'], alpha=0.8, edgecolor='white'))
        dist = int(np.sqrt(x**2 + y**2))
        ax_bev.text(x, y + l/2 + 0.5, f"{obj['cfg']['name']} {dist}m", color='white', fontsize=10, ha='center', fontweight='bold')

    for r in [10, 20, 30, 40, 50]:
        ax_bev.add_patch(Circle((0, 0), r, color='cyan', fill=False, linestyle='--', alpha=0.3))

    ax_bev.plot(0, 0, 'rx', markersize=15, markeredgewidth=3)
    ax_bev.set_xlim(CFG.x_range); ax_bev.set_ylim(CFG.y_range); ax_bev.axis('off')
    ax_bev.set_title(f"360° GEOMETRIC RADAR | {log_rec['location'].upper()}", fontsize=24, color='white', pad=30)

    plt.tight_layout()
    plt.savefig("fusion_dashboard_final.png", facecolor='black', dpi=150)
    print(f"✨ Success! Dashboard saved as 'fusion_dashboard_final.png'.")
    plt.show()

if __name__ == "__main__":
    generate_final_dashboard(scene_no=1)