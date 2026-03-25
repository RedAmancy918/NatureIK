#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
评估脚本：eval_ik.py
功能：
1. 读取 chunk-000 数据
2. 拆分双臂数据为单臂 Observation
3. 使用带 IK 引导 (Test-Time Guidance) 的 Diffusion Policy 进行推理
4. 合并结果并评估误差
"""

import sys
# sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
# sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import os
import glob
import json
import pathlib
from typing import List

import click
import hydra
import torch
import dill
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from diffusion_policy.workspace.base_workspace import BaseWorkspace

# --- [关键修改] 引入您刚才建立的两个模块 ---
# 请确保这两个文件在对应的目录下
from diffusion_policy.model.differentiable_ik_model import DifferentiableIKHelper
from diffusion_policy.gym_util.guidance_utils import guided_inference

OmegaConf.register_new_resolver("eval", eval, replace=True)


# -------------------------
# 1. 基础工具函数
# -------------------------
def _stack_col(df: pd.DataFrame, key: str) -> np.ndarray:
    """Stack column data to numpy array."""
    return np.stack(df[key].to_numpy(), axis=0).astype(np.float32)

def _finalize_pred(pred_sum: np.ndarray, pred_cnt: np.ndarray) -> np.ndarray:
    """Average predictions from sliding windows."""
    pred_full = np.full_like(pred_sum, np.nan, dtype=np.float32)
    mask = pred_cnt > 0
    pred_full[mask] = (pred_sum[mask] / pred_cnt[mask]).astype(np.float32)
    return pred_full

# -------------------------
# 2. 姿态转换工具 (Torch 版 - 用于计算 Target)
# -------------------------
def rpy_to_quat_torch(rpy_tensor):
    """
    将 RPY 转为四元数 (xyzw)
    Input: [B, 3] (roll, pitch, yaw)
    Output: [B, 4] (x, y, z, w)
    """
    roll, pitch, yaw = rpy_tensor[:, 0], rpy_tensor[:, 1], rpy_tensor[:, 2]
    cr = torch.cos(roll * 0.5); sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5); sy = torch.sin(yaw * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return torch.stack([qx, qy, qz, qw], dim=-1)

# -------------------------
# 3. 姿态转换工具 (Numpy 版 - 用于数据加载)
# -------------------------
def _rpy_to_quat_xyz(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cr = np.cos(roll * 0.5); sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5); sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5); sy = np.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.stack([qx, qy, qz, qw], axis=-1).astype(np.float32)

def _canonicalize_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    w = quat_xyzw[:, 3:4]
    sign = np.where(w < 0.0, -1.0, 1.0).astype(np.float32)
    return quat_xyzw * sign

def _gpos14_to_obs7_xyzquat(gpos14: np.ndarray, arm: str, canonicalize: bool = True) -> np.ndarray:
    """
    拆分 gpos14 为单臂 observation [xyz, quat] (7维)
    """
    if arm == "right":
        xyz = gpos14[:, 0:3]
        rpy = gpos14[:, 3:6]
    elif arm == "left":
        xyz = gpos14[:, 7:10]
        rpy = gpos14[:, 10:13]
    else:
        raise ValueError(f"arm must be right/left, got {arm}")

    quat = _rpy_to_quat_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
    if canonicalize:
        quat = _canonicalize_quat(quat)
    return np.concatenate([xyz.astype(np.float32), quat.astype(np.float32)], axis=-1).astype(np.float32)

def _gt14_to_gt12_drop_gripper(gt14: np.ndarray) -> np.ndarray:
    """Drop gripper dimensions (6 and 13) for evaluation."""
    r6 = gt14[:, 0:6]
    l6 = gt14[:, 7:13]
    return np.concatenate([r6, l6], axis=-1).astype(np.float32)

# -------------------------
# 4. 绘图工具
# -------------------------
def _save_two_panel_joint_plot(out_dir: pathlib.Path, episode_name: str, gt: np.ndarray, pred: np.ndarray, title_suffix: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    T, D = gt.shape
    t = np.arange(T)
    ncols = 2 if D > 6 else 1
    nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(14, 2.2 * nrows))
    axes = np.array(axes).reshape(-1)
    for j in range(D):
        if j >= len(axes): break
        ax = axes[j]
        ax.plot(t, gt[:, j], label="gt", alpha=0.7)
        ax.plot(t, pred[:, j], label="pred", linestyle='--', alpha=0.7)
        ax.set_title(f"Joint {j}")
        if j == 0: ax.legend(loc="upper right")
    fig.suptitle(f"{episode_name} | GT vs Pred {title_suffix}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{episode_name}_joints_overlay.png", dpi=100)
    plt.close(fig)

# -------------------------
# 5. 数据加载工具
# -------------------------
def _load_chunk_dir(data_dir: str, needed_cols: List[str]) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if len(files) == 0: raise FileNotFoundError(f"No parquet files under: {data_dir}")
    dfs = []
    for fp in files:
        df = pd.read_parquet(fp)
        dfs.append(df[needed_cols])
    out = pd.concat(dfs, axis=0, ignore_index=True)
    sort_keys = [k for k in ["episode_index", "index", "frame_index"] if k in out.columns]
    if sort_keys: out = out.sort_values(sort_keys).reset_index(drop=True)
    return out

def _split_episode_ids(all_episode_ids: np.ndarray, split: str, val_ratio: float, seed: int = 44) -> np.ndarray:
    uniq = np.unique(all_episode_ids).astype(np.int64)
    if val_ratio <= 0.0: return uniq
    rng = np.random.RandomState(seed)
    ids = uniq.copy()
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_ratio)))
    return np.sort(ids[:n_val]) if split.lower() == "val" else np.sort(ids[n_val:])

# -------------------------
# Main Function
# -------------------------
@click.command()
@click.option("--checkpoint", "-c", required=True, type=str, help="Model checkpoint path")
@click.option("--output-dir", "-o", required=True, type=str, help="Output directory")
@click.option("--data-dir", "-d", required=True, type=str, help="Data directory (chunk-000)")
@click.option("--device", default="cuda:0", type=str, help="Inference device")
@click.option("--split", default="val", type=str)
@click.option("--val-ratio", default=0.1, type=float)
@click.option("--max-episodes", default=5, type=int)
@click.option("--gt-key", default="observation.state", type=str)
@click.option("--gpos-key", default="observation.state_gpos", type=str)
# --- [NEW] IK Specific Options ---
@click.option("--urdf-path", required=True, type=str, help="Path to robot URDF file")
@click.option("--ee-link", default="link6", type=str, help="End-effector link name (e.g., link6)")
@click.option("--guidance-scale", default=20.0, type=float, help="Strength of IK guidance")
def main(checkpoint, output_dir, data_dir, device, split, val_ratio, max_episodes,
         gt_key, gpos_key, urdf_path, ee_link, guidance_scale):

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading Checkpoint: {checkpoint}")
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    
    # 1. Init Workspace & Policy
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=str(output_path))
    workspace.load_payload(payload)
    policy = workspace.model
    if cfg.training.use_ema: 
        policy = workspace.ema_model
    
    device_t = torch.device(device)
    policy.to(device_t).eval()

    # 2. [NEW] 获取 Action 统计数据 (用于 IK 反归一化)
    # 假设 LinearNormalizer 存储方式为 policy.normalizer['action']
    norm_params = policy.normalizer['action'].params_dict
    action_stats = {
        'min': norm_params['min'].to(device_t) if 'min' in norm_params else None,
        'max': norm_params['max'].to(device_t) if 'max' in norm_params else None,
        'mean': norm_params['mean'].to(device_t) if 'mean' in norm_params else None,
        'scale': norm_params['scale'].to(device_t) if 'scale' in norm_params else None
    }

    # 3. [NEW] 初始化 IK Helper
    print(f"Initializing IK Helper: URDF={urdf_path}, Link={ee_link}")
    ik_helper = DifferentiableIKHelper(
        urdf_path=urdf_path, 
        end_effector_link_name=ee_link,
        dof=6,              # 单臂自由度 (根据您的数据，不含夹爪)
        device=device,
        tool_offset_z=0.0,  # [确认] 不含夹爪，Offset 为 0
        apply_z180=True     # [确认] 开启 180 度修正 (解决背对背安装坐标系问题)
    )

    # 4. Load Data
    needed_cols = ["episode_index", "index", "frame_index", gpos_key, gt_key]
    df_all = _load_chunk_dir(data_dir, needed_cols=needed_cols)
    ep_ids = _split_episode_ids(df_all["episode_index"].to_numpy().astype(np.int64), split, val_ratio)
    if max_episodes and max_episodes > 0: 
        ep_ids = ep_ids[:max_episodes]

    results = []
    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)
    
    # 获取预测 Horizon (例如 16)
    # 这里的 horizon 指的是 policy 输出的总步数 (包含 obs 历史)
    pred_horizon = cfg.policy.horizon 

    print(f"Starting Eval: {len(ep_ids)} episodes, Guidance Scale={guidance_scale}, EE Link={ee_link}")

    for episode_idx, ep_id in enumerate(tqdm(ep_ids, desc="Evaluating")):
        ep_df = df_all[df_all["episode_index"] == ep_id].copy()
        
        # Sort
        if "index" in ep_df.columns: ep_df = ep_df.sort_values(["index"])
        else: ep_df = ep_df.sort_values(["frame_index"])
        ep_df = ep_df.reset_index(drop=True)
        
        T = len(ep_df)
        if T < n_obs_steps: 
            print(f"Skipping short episode {ep_id}")
            continue

        # Data Preparation
        gpos14 = _stack_col(ep_df, gpos_key)
        obs_r = _gpos14_to_obs7_xyzquat(gpos14, arm="right")
        obs_l = _gpos14_to_obs7_xyzquat(gpos14, arm="left")
        gt12 = _gt14_to_gt12_drop_gripper(_stack_col(ep_df, gt_key))

        # Prediction Buffers
        D = 12
        pred_sum = np.zeros((T, D), dtype=np.float32)
        pred_cnt = np.zeros((T, D), dtype=np.int32)
        episode_results = {"episode_index": int(ep_id), "predictions": []}

        # Sliding Window Inference
        for start_idx in range(0, T - n_obs_steps + 1, n_action_steps):
            t_end = start_idx + n_obs_steps - 1
            
            # --- [NEW] 确定 IK 目标 (Target) ---
            # 目标设置为预测窗口的最后一帧 (End of Horizon)
            # pred_horizon 通常包含 obs_steps (例如 obs=2, horizon=16, 未来14帧)
            # 我们想约束的是最远的那个时刻
            tgt_idx = min(start_idx + pred_horizon - 1, T - 1)
            
            # 获取该帧的 GT gpos (14维) [1, 14]
            tgt_gpos = torch.from_numpy(gpos14[tgt_idx]).to(device_t).unsqueeze(0)

            # ===============================
            # 1. 右臂推理 (Right Arm Guided)
            # ===============================
            # 提取右臂目标: Pos(0:3), RPY(3:6) -> Quat
            target_r_pos = tgt_gpos[:, 0:3]
            target_r_quat = rpy_to_quat_torch(tgt_gpos[:, 3:6])
            
            # 构造目标字典
            target_dict_r = {
                'pos': target_r_pos,
                'quat': target_r_quat # 如果 helper 支持旋转约束
            }

            # 准备 Obs
            obs_win_r = obs_r[start_idx:start_idx + n_obs_steps]
            obs_dict_r = {"obs": torch.from_numpy(obs_win_r).unsqueeze(0).to(device_t)}

            # 执行带引导的推理
            # 注意：不使用 torch.no_grad()，因为 guided_inference 内部需要求导
            pred_r_tensor = guided_inference(
                policy=policy,
                obs_dict=obs_dict_r,
                ik_helper=ik_helper,
                target_pose_dict=target_dict_r,
                action_stats=action_stats,
                guidance_scale=guidance_scale,
                device=device_t
            )
            pred_r = pred_r_tensor.detach().cpu().numpy()[0].astype(np.float32)

            # ===============================
            # 2. 左臂推理 (Left Arm Guided)
            # ===============================
            # 提取左臂目标: Pos(7:10), RPY(10:13)
            target_l_pos = tgt_gpos[:, 7:10]
            target_l_quat = rpy_to_quat_torch(tgt_gpos[:, 10:13])
            
            target_dict_l = {
                'pos': target_l_pos,
                'quat': target_l_quat
            }

            obs_win_l = obs_l[start_idx:start_idx + n_obs_steps]
            obs_dict_l = {"obs": torch.from_numpy(obs_win_l).unsqueeze(0).to(device_t)}

            pred_l_tensor = guided_inference(
                policy=policy,
                obs_dict=obs_dict_l,
                ik_helper=ik_helper,
                target_pose_dict=target_dict_l,
                action_stats=action_stats,
                guidance_scale=guidance_scale,
                device=device_t
            )
            pred_l = pred_l_tensor.detach().cpu().numpy()[0].astype(np.float32)

            # ===============================
            # 3. 合并与累积
            # ===============================
            # pred_r/l shape: [horizon, 6]
            # 这里的 pred_r[:, :6] 是为了防御性编程，确保只取关节角
            pred_block = np.concatenate([pred_r[:, :6], pred_l[:, :6]], axis=-1)
            
            # Time alignment accumulation
            # Diffusion Policy 输出通常包含 obs_steps 的历史重构
            # 如果 horizon=16, obs=2，通常前2帧是历史，后14帧是未来
            # 我们直接对齐到 t_end - (obs_steps - 1) 开始
            # 简单起见，这里假设 outputs 0 对应 start_idx
            
            for k in range(pred_block.shape[0]):
                t = start_idx + k # 假设输出从 start_idx 开始覆盖
                if t >= T: break
                pred_sum[t] += pred_block[k]
                pred_cnt[t] += 1

        # Finalize
        pred12 = _finalize_pred(pred_sum, pred_cnt)
        episode_results["gt12"] = gt12.tolist()
        episode_results["pred12"] = pred12.tolist()
        results.append(episode_results)
        
        _save_two_panel_joint_plot(
            output_path, 
            f"episode_{ep_id:06d}", 
            gt12, 
            pred12, 
            title_suffix=f"(Scale={guidance_scale})"
        )

    # Save JSON
    output_file = output_path / "eval_results.json"
    with open(output_file, "w") as f: 
        json.dump(results, f, indent=2)
    
    # Calc Stats
    all_err = []
    for ep in results:
        gt_arr = np.array(ep["gt12"])
        pd_arr = np.array(ep["pred12"])
        valid = np.isfinite(pd_arr).all(axis=1)
        if valid.sum() > 0: 
            all_err.append(np.abs(pd_arr[valid] - gt_arr[valid]))
    
    if all_err:
        all_err = np.concatenate(all_err, axis=0)
        mean_err = all_err.mean()
        print(f"\nEvaluation Complete.")
        print(f"Mean Error (Joints): {mean_err:.6f}")
        
        stats = {
            "mean_error": float(mean_err),
            "max_error": float(all_err.max()),
            "std_error": float(all_err.std())
        }
        with open(output_path / "eval_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()