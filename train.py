# ============================================================
# Bird's-Eye-View 2D Occupancy — FovealRadarBEVv2 (PRO CONFIG)
# Optimized for High Fidelity + 8GB VRAM + Distance-Weighted Loss
# Team: Why not?
# ============================================================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.ops import FeaturePyramidNetwork
from PIL import Image
from pyquaternion import Quaternion
import cv2

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# ── Paths & Device ──────────────────────────────────────────
DATAROOT = '/home/abhi_pop/MAHE/bev_project/nuscenes'
VERSION  = 'v1.0-mini'
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT     = './best_model.pth'
print(f"Device: {DEVICE}")

# ============================================================
# CELL 3 — Config (High Precision)
# ============================================================
class CFG:
    img_h, img_w = 256, 704      # High resolution for detailed vision
    d_min, d_max = 1.0, 33.0
    depth_bins = 64              # ULTRA-PRECISION (0.5m steps)
    d_step = 0.5                 # Eliminates the "Starfish" effect
    img_feat_dim = 128           # High model capacity
    bev_feat_dim = 128           
    batch_size = 2               # Balanced for 8GB VRAM
    bev_h, bev_w = 256, 256
    x_range, y_range = (-51.2, 51.2), (-51.2, 51.2)
    resolution = 0.4
    n_sem_classes = 5
    sparse_topk = 0.25

IMG_SIZE = (CFG.img_h, CFG.img_w)
CAMERA_NAMES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']
SEMANTIC_MAP = {'vehicle.car': 1, 'vehicle.bus.rigid': 1, 'vehicle.bus.bendy': 1, 'vehicle.truck': 1, 'vehicle.motorcycle': 2, 'vehicle.bicycle': 2, 'human.pedestrian.adult': 3, 'human.pedestrian.child': 3, 'human.pedestrian.wheelchair': 3, 'human.pedestrian.stroller': 3, 'movable_object.barrier': 4, 'movable_object.trafficcone': 4, 'static_object.bicycle_rack': 4}

# ============================================================
# CELL 4 — Dataset Helpers
# ============================================================
def make_transform(rotation, translation):
    T = np.eye(4, dtype=np.float64); T[:3, :3] = Quaternion(rotation).rotation_matrix; T[:3, 3] = translation; return T

def load_camera(nusc, sample, cam_name, img_size=IMG_SIZE):
    sd = nusc.get('sample_data', sample['data'][cam_name]); img_path = os.path.join(nusc.dataroot, sd['filename'])
    if not os.path.exists(img_path): return (np.zeros((3, img_size[0], img_size[1]), np.float32), np.eye(3, dtype=np.float32), np.eye(4, dtype=np.float32))
    img = Image.open(img_path).convert('RGB').resize((img_size[1], img_size[0]), Image.BILINEAR)
    img = np.array(img, np.float32) / 255.0; img = img.transpose(2, 0, 1)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token']); K = np.array(cs['camera_intrinsic'], dtype=np.float32)
    K[0] *= img_size[1] / sd['width']; K[1] *= img_size[0] / sd['height']
    return img, K, make_transform(cs['rotation'], cs['translation']).astype(np.float32)

def load_lidar_ego(nusc, token):
    sd = nusc.get('sample_data', token); path = os.path.join(nusc.dataroot, sd['filename'])
    if not os.path.exists(path): return np.zeros((0, 4), dtype=np.float32)
    raw = np.fromfile(path, dtype=np.float32); pts = raw.reshape(-1, 5 if raw.size % 5 == 0 else 4)[:, :4]
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token']); T = make_transform(cs['rotation'], cs['translation'])
    return np.hstack([(T @ np.hstack([pts[:, :3], np.ones((len(pts), 1))]).T).T[:, :3], pts[:, 3:4]])

def lidar_to_bev_gt(pts, boxes):
    xmin, xmax, ymin, ymax, res = *CFG.x_range, *CFG.y_range, CFG.resolution
    occ, sem = np.zeros((CFG.bev_h, CFG.bev_w), dtype=np.uint8), np.zeros((CFG.bev_h, CFG.bev_w), dtype=np.uint8)
    if len(pts) > 0:
        m = (pts[:,0]>=xmin)&(pts[:,0]<xmax)&(pts[:,1]>=ymin)&(pts[:,1]<ymax)&(pts[:,2]>-4.7)&(pts[:,2]<3.0); p = pts[m]
        if len(p): occ[np.clip(((p[:, 1] - ymin) / res).astype(np.int32), 0, CFG.bev_h - 1), np.clip(((p[:, 0] - xmin) / res).astype(np.int32), 0, CFG.bev_w - 1)] = 1
    for b in boxes:
        if b['label'] == 0: continue
        px, py = (b['corners_3d'][:4, 0]-xmin)/res, (b['corners_3d'][:4, 1]-ymin)/res
        poly = np.stack([px, py], 1).astype(np.int32)
        cv2.fillPoly(sem, [poly], int(b['label'])); cv2.fillPoly(occ, [poly], 1)
    return occ, sem

def lidar_to_cam_gt(pts, E, K, h, w):
    if len(pts) == 0: return np.zeros((h, w), dtype=np.int64)
    p_cam = (np.linalg.inv(E) @ np.hstack([pts[:,:3], np.ones((len(pts),1))]).T).T[:,:3]
    v = p_cam[:,2] > 0.1; p_cam = p_cam[v]
    if len(p_cam) == 0: return np.zeros((h, w), dtype=np.int64)
    uvw = (K @ p_cam.T).T; z = uvw[:,2].clip(1e-6, None)
    u, v = np.clip(uvw[:,0]/z, 0, w-1).astype(np.int32), np.clip(uvw[:,1]/z, 0, h-1).astype(np.int32)
    lmap = np.zeros((h, w), dtype=np.int64); lmap[v, u] = 1; return lmap

# ============================================================
# CELL 5 — Dataset
# ============================================================
class NuScenesBEVDataset(Dataset):
    def __init__(self, nusc, scene_indices, augment=False):
        self.nusc, self.augment, self.samples = nusc, augment, []
        for si in scene_indices:
            t = nusc.scene[si]['first_sample_token']
            while t: self.samples.append(t); t = nusc.get('sample', t)['next']
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.nusc.get('sample', self.samples[idx]); imgs, Ks, Es = [], [], []
        for c in CAMERA_NAMES:
            img, K, E = load_camera(self.nusc, s, c); imgs.append(img); Ks.append(K); Es.append(E)
        lp = load_lidar_ego(self.nusc, s['data']['LIDAR_TOP'])
        ep = self.nusc.get('ego_pose', self.nusc.get('sample_data', s['data']['LIDAR_TOP'])['ego_pose_token'])
        T_g2e = np.linalg.inv(make_transform(ep['rotation'], ep['translation']))
        boxes = []
        for a in s['anns']:
            ann = self.nusc.get('sample_annotation', a); cg = self.nusc.get_box(a).corners().T
            ce = (T_g2e @ np.hstack([cg, np.ones((8, 1))]).T).T[:, :3]
            boxes.append({'corners_3d': ce, 'label': SEMANTIC_MAP.get(ann['category_name'], 0)})
        og, sg = lidar_to_bev_gt(lp, boxes); cg = np.stack([lidar_to_cam_gt(lp, Es[i], Ks[i], CFG.img_h//8, CFG.img_w//8) for i in range(6)])
        if self.augment:
            if np.random.rand() > 0.5:
                imgs = [i[:,:,::-1].copy() for i in imgs]; og, sg, cg = og[:,::-1].copy(), sg[:,::-1].copy(), cg[:,:,::-1].copy()
            if np.random.rand() > 0.5:
                factor = np.random.uniform(0.7, 1.3); imgs = [np.clip(i * factor, 0, 1) for i in imgs]
            if np.random.rand() > 0.7: imgs[np.random.randint(0, 6)] = np.zeros_like(imgs[0])
        return {'images': torch.from_numpy(np.stack(imgs)).float(), 'intrinsics': torch.from_numpy(np.stack(Ks)), 'extrinsics': torch.from_numpy(np.stack(Es)), 'radar_pts': torch.zeros(0, 5), 'occ_gt': torch.from_numpy(og).long(), 'sem_gt': torch.from_numpy(sg).long(), 'cam_gt': torch.from_numpy(cg)}

def build_dataloader(nusc, batch_size=CFG.batch_size, num_workers=0):
    n = len(nusc.scene); split = max(1, int(n * 0.8))
    def collate(b): return {k: torch.stack([x[k] for x in b]) for k in b[0].keys()}
    tl = DataLoader(NuScenesBEVDataset(nusc, list(range(split)), augment=True), batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate, pin_memory=True, drop_last=True)
    vl = DataLoader(NuScenesBEVDataset(nusc, list(range(split, n)), augment=False), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate, pin_memory=True)
    return tl, vl

# ============================================================
# CELL 6 — Model
# ============================================================
class ImageBackbone(nn.Module):
    def __init__(self, out_ch=CFG.img_feat_dim):
        super().__init__(); swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.patch_embed = swin.features[0]; self.stages = nn.ModuleList([swin.features[i] for i in range(1, 8)])
        self.fpn = FeaturePyramidNetwork([96, 192, 384, 768], out_ch)
        for p in list(self.patch_embed.parameters())+list(self.stages[0].parameters()): p.requires_grad_(False)
        self.cam_seg_head = nn.Conv2d(out_ch, 1, 1)
    def forward(self, x):
        x = self.patch_embed(x); feats = []
        for i, s in enumerate(self.stages):
            x = cp.checkpoint(s, x, use_reentrant=False) if i > 0 else s(x)
            if i in [0, 2, 4, 6]: feats.append(x.permute(0, 3, 1, 2).contiguous())
        f = self.fpn({str(i): feats[i] for i in range(4)})['1']; return f, self.cam_seg_head(f)

class LSSViewTransformer(nn.Module):
    def __init__(self):
        super().__init__(); self.register_buffer('d_vals', torch.arange(CFG.d_min, CFG.d_min + CFG.depth_bins * CFG.d_step, CFG.d_step))
    def _splat(self, img_feat, dp, Ks, Es, d_vals):
        B, N, C, Hf, Wf = img_feat.shape; D = d_vals.shape[0]; device = img_feat.device
        xs = torch.linspace(0, CFG.img_w - 1, Wf, device=device); ys = torch.linspace(0, CFG.img_h - 1, Hf, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij'); pix = torch.stack([gx, gy, torch.ones_like(gx)], 0).reshape(3, -1)
        voxel = (img_feat.unsqueeze(3) * dp.unsqueeze(2)).view(B, N, C, D, Hf * Wf)
        bev, count = torch.zeros(B, C, CFG.bev_h, CFG.bev_w, device=device), torch.zeros(B, 1, CFG.bev_h, CFG.bev_w, device=device)
        for b in range(B):
            for n in range(N):
                Kinv, T = torch.inverse(Ks[b, n].float()), Es[b, n].float()
                pts_ego = torch.einsum('ij,djp->dip', T, torch.cat([Kinv @ pix * d_vals.view(D, 1, 1), torch.ones(D, 1, Hf*Wf, device=device)], 1))[:, :3, :]
                bx = ((pts_ego[:, 0] - CFG.x_range[0]) / (CFG.x_range[1] - CFG.x_range[0]) * CFG.bev_w).long()
                by = ((pts_ego[:, 1] - CFG.y_range[0]) / (CFG.y_range[1] - CFG.y_range[0]) * CFG.bev_h).long()
                v = (bx>=0)&(bx<CFG.bev_w)&(by>=0)&(by<CFG.bev_h); f_dp = voxel[b, n].permute(1, 0, 2)
                for di in range(D):
                    if not v[di].any(): continue
                    idx = by[di][v[di]] * CFG.bev_w + bx[di][v[di]]
                    bev[b].view(C, -1).index_add_(1, idx, f_dp[di][:, v[di]])
                    count[b].view(1, -1).index_add_(1, idx, torch.ones(1, v[di].sum(), device=device))
        return bev / (count + 1e-6)
    def forward(self, img_feat, dp, Ks, Es):
        with autocast('cuda', enabled=False):
            step = max(1, CFG.depth_bins // 8); s_bev = self._splat(img_feat.float(), dp[:,:,::step].float(), Ks.float(), Es.float(), self.d_vals[::step])
            mask = (s_bev.abs().mean(1) >= torch.topk(s_bev.abs().mean(1).view(img_feat.shape[0], -1), int(CFG.bev_h*CFG.bev_w*CFG.sparse_topk), 1).values[:, -1].view(-1, 1, 1)).float()
            return self._splat(img_feat.float(), dp.float(), Ks.float(), Es.float(), self.d_vals) * mask.unsqueeze(1), mask

class FovealTransformer(nn.Module):
    def __init__(self, alpha=0.6):
        super().__init__()
        ys, xs = torch.meshgrid(torch.linspace(-1,1,256), torch.linspace(-1,1,256), indexing='ij')
        r = torch.sqrt(xs**2+ys**2).clamp(max=1.0); ang = torch.atan2(ys, xs)
        grid = torch.stack([r**alpha*torch.cos(ang), r**alpha*torch.sin(ang)], -1).unsqueeze(0)
        self.register_buffer('grid', grid)
    def forward(self, x):
        return F.grid_sample(x, self.grid.expand(x.shape[0], -1, -1, -1), align_corners=True)

class FovealRadarBEVv2(nn.Module):
    def __init__(self):
        super().__init__(); self.backbone = ImageBackbone(); self.depth_head = nn.Sequential(nn.Conv2d(CFG.img_feat_dim, CFG.img_feat_dim, 3, padding=1), nn.BatchNorm2d(CFG.img_feat_dim), nn.ReLU(True), nn.Conv2d(CFG.img_feat_dim, CFG.depth_bins, 1))
        self.view_tfm = LSSViewTransformer(); self.foveal = FovealTransformer(); self.bev_proj = nn.Conv2d(CFG.img_feat_dim, CFG.bev_feat_dim, 1)
        self.decoder = nn.Sequential(nn.Conv2d(CFG.bev_feat_dim, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.occ_head = nn.Conv2d(128, 1, 1); self.sem_head = nn.Conv2d(128, 5, 1); self.unc_head = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(True), nn.Conv2d(64, 1, 1), nn.Tanh())
    def forward(self, imgs, Ks, Es):
        B, N, _, H, W = imgs.shape; f, cp = self.backbone(imgs.view(B*N, 3, H, W))
        dp = self.depth_head(f).softmax(1); bev, _ = self.view_tfm(f.view(B, N, -1, H//8, W//8), dp.view(B, N, -1, H//8, W//8), Ks, Es)
        dec = self.decoder(self.foveal(self.bev_proj(bev)))
        return {'occ_logits': self.occ_head(dec), 'sem_logits': self.sem_head(dec), 'log_sigma': self.unc_head(dec), 'cam_pred': cp}
    def predict_tta(self, imgs, Ks, Es, rad):
        o1 = self.forward(imgs, Ks, Es); o2 = self.forward(torch.flip(imgs, [-1]), Ks, Es); o3 = self.forward(imgs*0.85, Ks, Es)
        o1['occ_logits'] = (o1['occ_logits'] + torch.flip(o2['occ_logits'], [-1]) + o3['occ_logits']) / 3.0; return o1

# ============================================================
# CELL 7 — Upgraded Losses
# ============================================================
def get_dist_w(H, W, device, gamma=1.5):
    xs = torch.linspace(*CFG.x_range, W, device=device)
    ys = torch.linspace(*CFG.y_range, H, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    dist = torch.sqrt(gx**2 + gy**2)
    dist_w = 1.0 / (dist + 1.0)**gamma
    return dist_w / dist_w.mean()

def occ_loss(logits, gt, device):
    w = get_dist_w(logits.shape[-2], logits.shape[-1], device, gamma=1.5)
    pw = torch.tensor([5.0], device=device) 
    gt_smooth = gt.float().unsqueeze(1) * 0.9 + 0.05
    l = F.binary_cross_entropy_with_logits(logits, gt_smooth, pos_weight=pw, reduction='none')
    return (l * w).mean()

def compute_losses(out, batch, dev):
    og, sg, cg = batch['occ_gt'].to(dev), batch['sem_gt'].to(dev), batch['cam_gt'].to(dev)
    lo  = occ_loss(out['occ_logits'], og, dev)
    ls  = F.cross_entropy(out['sem_logits'], sg, weight=torch.tensor([0.1, 2., 3., 4., 2.], device=dev))
    lu  = (0.5 * torch.exp(-2 * out['log_sigma']) * (torch.sigmoid(out['occ_logits']) - og.float().unsqueeze(1))**2 + out['log_sigma']).mean()
    lc  = F.binary_cross_entropy_with_logits(out['cam_pred'], cg.view(-1, 1, CFG.img_h//8, CFG.img_w//8).float())
    return lo + 0.5*ls + 0.05*lu + 0.02*lc, {'occ': lo.item()}

# ============================================================
# CELL 8 — Training Loop
# ============================================================
def run_epoch(model, loader, opt, sched, scaler, device, is_train=True, ep=0):
    model.train() if is_train else model.eval()
    for i, b in enumerate(loader):
        imgs, Ks, Es = b['images'].to(device), b['intrinsics'].to(device), b['extrinsics'].to(device)
        opt.zero_grad()
        with autocast('cuda'):
            out = model(imgs, Ks, Es); loss, parts = compute_losses(out, b, device)
        if is_train: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if i % 10 == 0:
            iou = (((torch.sigmoid(out['occ_logits'].squeeze(1))>0.4).long() * b['occ_gt'].to(device)).sum() / (((torch.sigmoid(out['occ_logits'].squeeze(1))>0.4).long() + b['occ_gt'].to(device) > 0).sum() + 1e-6)).item()
            tag = "TRAIN" if is_train else "VAL"
            print(f"[{tag}] ep={ep} batch={i}/{len(loader)} loss={loss.item():.4f} IoU={iou:.3f}")

# ============================================================
# CELL 9 — Main (Resume & Start Epoch Logic)
# ============================================================
def main():
    from nuscenes.nuscenes import NuScenes
    print(f"Loading {VERSION} (RESUMING WITH WEIGHTED LOSS)...")
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    tl, vl = build_dataloader(nusc) 
    model = FovealRadarBEVv2().to(DEVICE)

    # --- RESUME LOGIC ---
    if os.path.exists(CKPT):
        print(f"Restoring progress from {CKPT}...")
        checkpoint = torch.load(CKPT, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint)
    
    # Lower Learning Rate slightly for fine-tuning with aggressive loss
    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) 
    sched = CosineAnnealingLR(opt, T_max=22) 
    scaler = GradScaler('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    # Resuming from the next epoch after your stop (Epoch 7 complete, so start at 8)
    for ep in range(8, 30):
        run_epoch(model, tl, opt, sched, scaler, DEVICE, True, ep)
        sched.step(); torch.save(model.state_dict(), CKPT)
        print(f"Epoch {ep} saved with Distance-Weighted Loss.")

if __name__ == '__main__':
    main()