import sys, os
sys.path.insert(0, os.path.expanduser("~/yjh/diffusion_policy"))

import torch
import numpy as np
import pandas as pd
from diffusion_policy.model.differentiable_ik_model import DifferentiableIKHelper

URDF    = "../diffusion_policy/urdf_data/play_g2_usb_cam/urdf/play_g2_usb_cam.urdf"
PARQUET = "/home/user/Data/data/airbot/2026_0322_airbot_ALL_SplitNoRGB/data/chunk-000/episode_000000.parquet"

df      = pd.read_parquet(PARQUET)
q_curr  = np.stack(df["observation.state"].to_numpy()).astype(np.float32)[:, :6]
q_tgt   = np.stack(df["action"].to_numpy()).astype(np.float32)[:, :6]  # ← 直接就是绝对角度
eef_tgt = np.stack(df["action_quat"].to_numpy()).astype(np.float32)[:, :3]

print("delta_q 实际值（应该很小）:", (q_tgt - q_curr)[:3].round(4))

N  = min(100, len(q_tgt))
qt = torch.from_numpy(q_tgt[:N])
gt = eef_tgt[:N]

print("=" * 55)
for ee_link in ["link6", "eef_connect_base_link", "end_link"]:
    try:
        fk = DifferentiableIKHelper(urdf_path=URDF, end_effector_link_name=ee_link, device="cpu")
        pos, _ = fk.get_current_pose(qt)
        offset = gt - pos.numpy()
        err_mm = np.linalg.norm(offset, axis=-1) * 1000
        print(f"\n[{ee_link}]")
        print(f"  误差      mean={err_mm.mean():.1f}mm  std={err_mm.std():.1f}mm")
        print(f"  偏移均值  (mm): {(offset.mean(axis=0) * 1000).round(1)}")
        print(f"  偏移std   (mm): {(offset.std(axis=0) * 1000).round(1)}")
    except Exception as e:
        print(f"[{ee_link}] 失败: {e}")