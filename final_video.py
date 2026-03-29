import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon, Circle
from ultralytics import YOLO
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

# Importing your specific setup from train.py
from train import CFG, NuScenesBEVDataset, DEVICE, DATAROOT, VERSION

# --- CONFIG ---
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
    if scale < 1.0 or scale > 65.0: return None 
    p_ego = scale * (R @ ray_cam) + t
    return p_ego[0], p_ego[1]

def generate_scene_video(scene_no=1, output_name="boston_seaport_perception.mp4"):
    print(f"Device: {DEVICE} | Initializing Video Engine for Scene {scene_no}...")
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    
    # 1. Setup Scene & Map
    scene_rec = nusc.scene[scene_no]
    log_rec = nusc.get('log', scene_rec['log_token'])
    nmap = NuScenesMap(dataroot=DATAROOT, map_name=log_rec['location'])
    
    # Get all sample tokens for the scene
    sample_tokens = []
    curr_token = scene_rec['first_sample_token']
    while curr_token != "":
        sample_tokens.append(curr_token)
        curr_token = nusc.get('sample', curr_token)['next']
    
    print(f"Found {len(sample_tokens)} frames to process.")
    
    # 2. Init Models & Dataset
    ds = NuScenesBEVDataset(nusc, [scene_no], augment=False)
    yolo_model = YOLO('yolov8n.pt').to(DEVICE)
    
    frames_dir = "temp_video_frames"
    os.makedirs(frames_dir, exist_ok=True)
    frame_files = []

    # 3. Frame Processing Loop
    for idx, sample_token in enumerate(tqdm(sample_tokens, desc="Processing Scene")):
        sample_rec = nusc.get('sample', sample_token)
        data = ds[idx]
        imgs, Ks, Es = data['images'], data['intrinsics'], data['extrinsics']

        # Setup Figure
        fig = plt.figure(figsize=(26, 12), facecolor='black')
        gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 3.5])
        plt.style.use('dark_background')

        all_dets = []
        cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

        # Multi-Camera Inference
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
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=CLASS_MAP[cls_id]['color'], linewidth=2))
                    pos = project_to_ground_direct((x1+x2)/2, y2, Ks[i].numpy(), Es[i].numpy())
                    if pos:
                        all_dets.append({'pos': pos, 'cfg': CLASS_MAP[cls_id]})
            ax.axis('off')

        # BEV Radar Rendering
        ax_bev = fig.add_subplot(gs[:, 3])
        sd_rec = nusc.get('sample_data', sample_rec['data']['CAM_FRONT'])
        ego_pose = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        
        patch_angle = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0] * 180 / np.pi 
        patch_box = (ego_pose['translation'][0], ego_pose['translation'][1], 102.4, 102.4)
        map_mask = nmap.get_map_mask(patch_box, patch_angle, ['drivable_area', 'walkway'], (200, 200))
        
        ax_bev.imshow(map_mask[0], extent=CFG.x_range + CFG.y_range, cmap='bone', alpha=0.3)
        ax_bev.imshow(np.ma.masked_where(map_mask[1] == 0, map_mask[1]), extent=CFG.x_range + CFG.y_range, cmap='winter', alpha=0.15)

        # BEV Clustering (NMS)
        merged = []
        for d in all_dets:
            if not any(np.linalg.norm(np.array(d['pos']) - np.array(m['pos'])) < 3.5 for m in merged):
                merged.append(d)

        for obj in merged:
            x, y = obj['pos']; w, l = obj['cfg']['size']
            corners = np.array([[x-w/2, y-l/2], [x+w/2, y-l/2], [x+w/2, y+l/2], [x-w/2, y+l/2]])
            ax_bev.add_patch(Polygon(corners, closed=True, color=obj['cfg']['color'], alpha=0.8, edgecolor='white'))
            dist = int(np.sqrt(x**2 + y**2))
            ax_bev.text(x, y + l/2 + 0.5, f"{obj['cfg']['name']} {dist}m", color='white', fontsize=10, ha='center', fontweight='bold')

        # Radar Rings
        for r in [10, 20, 30, 40, 50]:
            ax_bev.add_patch(Circle((0, 0), r, color='cyan', fill=False, linestyle='--', alpha=0.3))

        ax_bev.plot(0, 0, 'rx', markersize=15, markeredgewidth=3)
        ax_bev.set_xlim(CFG.x_range); ax_bev.set_ylim(CFG.y_range); ax_bev.axis('off')
        ax_bev.set_title(f"SCENE {scene_no} | FRAME {idx:02d} | {log_rec['location'].upper()}", fontsize=22, color='white', pad=30)

        # Save Frame
        f_path = os.path.join(frames_dir, f"frame_{idx:04d}.png")
        plt.savefig(f_path, bbox_inches='tight', facecolor='black', dpi=100)
        frame_files.append(f_path)
        plt.close(fig) 

    # 4. Final Video Assembly
    print(" Compiling frames into MP4...")
    sample_img = cv2.imread(frame_files[0])
    h, w, _ = sample_img.shape
    # 4 FPS is standard for nuScenes Keyframes
    v_out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), 4, (w, h))
    
    for f in frame_files:
        v_out.write(cv2.imread(f))
    v_out.release()
    print(f" SUCCESS! Final video saved as: {output_name}")

if __name__ == "__main__":
    generate_scene_video(scene_no=1)
