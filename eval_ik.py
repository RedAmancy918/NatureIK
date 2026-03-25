#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import os
import glob
import json
import pathlib
from typing import List

from diffusion_policy.dataset.robot_feature_utils import build_robot_feature_map
from diffusion_policy.dataset.robot_specs import ROBOT_SPECS

import click
import hydra
import torch
import dill
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

# -------------------------
# basic utils
# -------------------------
def _stack_col(df: pd.DataFrame, key: str) -> np.ndarray:
    return np.stack(df[key].to_numpy(), axis=0).astype(np.float32)

def _finalize_pred(pred_sum: np.ndarray, pred_cnt: np.ndarray) -> np.ndarray:
    pred_full = np.full_like(pred_sum, np.nan, dtype=np.float32)
    mask = pred_cnt > 0
    pred_full[mask] = (pred_sum[mask] / pred_cnt[mask]).astype(np.float32)
    return pred_full

def _align_quaternions(pose_array: np.ndarray, w_idx: int = 6, quat_start: int = 3) -> np.ndarray:
    """对齐四元数，必须和 Dataset 里的预处理保持绝对一致"""
    neg_mask = pose_array[:, w_idx] < 0
    pose_array[neg_mask, quat_start:quat_start+4] = -pose_array[neg_mask, quat_start:quat_start+4]
    return pose_array

# -------------------------
# plotting
# -------------------------
def _save_two_panel_joint_plot(out_dir: pathlib.Path, episode_name: str, gt: np.ndarray, pred: np.ndarray, title_suffix: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    T, D = gt.shape
    t = np.arange(T)

    ncols = 2 if D > 4 else 1
    nrows = int(np.ceil(D / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(14, 2.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for j in range(D):
        ax = axes[j]
        ax.plot(t, gt[:, j], label="GT Absolute Joint")
        ax.plot(t, pred[:, j], label="Pred Absolute Joint")
        ax.set_title(f"Joint {j}")
        ax.legend(loc="upper right")

    for k in range(D, nrows * ncols):
        axes[k].axis("off")

    fig.suptitle(f"{episode_name} | GT vs Pred (Overlay) {title_suffix}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{episode_name}_joints_overlay.png", dpi=150)
    plt.close(fig)

# -------------------------
# chunk loader + episode split
# -------------------------
def _load_chunk_dir(data_dir: str, needed_cols: List[str]) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files under: {data_dir}")

    dfs = []
    for fp in files:
        df = pd.read_parquet(fp)
        miss = [c for c in needed_cols if c not in df.columns]
        if miss:
            raise KeyError(f"{fp} missing columns: {miss}. Has: {list(df.columns)}")
        dfs.append(df[needed_cols])

    out = pd.concat(dfs, axis=0, ignore_index=True)
    sort_keys = [k for k in ["episode_index", "index", "frame_index"] if k in out.columns]
    if sort_keys:
        out = out.sort_values(sort_keys).reset_index(drop=True)
    return out

def _split_episode_ids(all_episode_ids: np.ndarray, split: str, val_ratio: float, seed: int = 44) -> np.ndarray:
    uniq = np.unique(all_episode_ids).astype(np.int64)
    if val_ratio <= 0.0: return uniq
    rng = np.random.RandomState(seed)
    ids = uniq.copy()
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_ratio)))
    val_ids = np.sort(ids[:n_val])
    trn_ids = np.sort(ids[n_val:])
    return val_ids if split.lower() == "val" else trn_ids

# -------------------------
# main
# -------------------------
@click.command()
@click.option("--checkpoint", "-c", required=True, type=str, help="模型checkpoint路径")
@click.option("--output-dir", "-o", required=True, type=str, help="输出目录")
@click.option("--data-dir", "-d", required=True, type=str, help="chunk-000 数据目录")
@click.option("--device", default="cuda:0", type=str, help="设备")
@click.option("--split", default="val", type=str, help="train/val（按 episode 切分）")
@click.option("--val-ratio", default=0.1, type=float, help="val episode 比例")
@click.option("--max-episodes", default=5, type=int, help="最大评估episode数量")
@click.option("--curr-q-key", default="observation.state", type=str, help="当前关节列名")
@click.option("--curr-eef-key", default="observation.state_quat", type=str, help="当前末端位姿列名")
@click.option("--tgt-q-key", default="action", type=str, help="目标关节列名 (GT)")
@click.option("--tgt-eef-key", default="action_quat", type=str, help="目标末端位姿列名")
@click.option("--save-plots/--no-save-plots", default=True)
@click.option("--target-robot", default=None, type=str, help="测试跨形态迁移时，强行指定新的机器人名称")
def main(checkpoint, output_dir, data_dir, device, split, val_ratio, max_episodes,
         curr_q_key, curr_eef_key, tgt_q_key, tgt_eef_key, save_plots, target_robot):

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    # 严格校验维度
    obs_dim = int(cfg.policy.obs_dim)
    action_dim = int(cfg.policy.action_dim)
    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)

    #if obs_dim != 20: raise ValueError(f"Expected obs_dim=20. Got {obs_dim}")
    #if action_dim != 6: raise ValueError(f"Expected action_dim=6. Got {action_dim}")

    # === 新增：处理机器人结构特征 ===
    # === 处理机器人结构特征（支持跨形态迁移） ===
    use_robot_feature = cfg.task.dataset.get('use_robot_feature', False)
    
    # 如果你在命令行传了新的机器人名字，就用新的；否则用训练时的老名字
    train_robot_name = cfg.task.dataset.get('robot_name', 'airbot_single_arm')
    robot_name = target_robot if target_robot else train_robot_name
    
    robot_feature = None
    
    if use_robot_feature:
        if target_robot:
            print(f"⚠️ 跨形态迁移激活！训练机器: {train_robot_name} -> 测试机器: {robot_name}")
        else:
            print(f"正在构建 {robot_name} 的结构编码...")
            
        max_joints = cfg.task.dataset.get('max_joints', 16)
        feature_map = build_robot_feature_map(ROBOT_SPECS, max_joints=max_joints)
        
        if robot_name not in feature_map:
            raise KeyError(
                f"报错了！你在命令行指定的 '{robot_name}' 不存在。\n"
                f"请先去 'diffusion_policy/dataset/robot_specs.py' 的 ROBOT_SPECS 字典里把它注册进去！"
            )
            
        robot_feature = feature_map[robot_name].astype(np.float32)
        print(f"成功加载特征，维度: {robot_feature.shape}")
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=str(output_path))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    device_t = torch.device(device)
    policy.to(device_t).eval()

    # 只需要这四个关键列
    needed_cols = ["episode_index", curr_q_key, curr_eef_key, tgt_q_key, tgt_eef_key]
    sample_df = pd.read_parquet(glob.glob(os.path.join(data_dir, "*.parquet"))[0])
    if "index" in sample_df.columns: needed_cols.append("index")
    if "frame_index" in sample_df.columns: needed_cols.append("frame_index")

    df_all = _load_chunk_dir(data_dir, needed_cols=needed_cols)

    ep_ids = _split_episode_ids(df_all["episode_index"].to_numpy(), split, val_ratio, 44)
    if max_episodes: ep_ids = ep_ids[:max_episodes]

    results = []
    with torch.no_grad():
        for episode_idx, ep_id in enumerate(tqdm(ep_ids, desc="评估中")):
            ep_df = df_all[df_all["episode_index"] == ep_id].copy()
            if "index" in ep_df.columns: ep_df = ep_df.sort_values(["index"])
            elif "frame_index" in ep_df.columns: ep_df = ep_df.sort_values(["frame_index"])
            ep_df = ep_df.reset_index(drop=True)

            T = len(ep_df)
            if T < n_obs_steps: continue

            # 1. 提取当前状态并切片切除夹爪 ([: ,:6] 和 [:, :7])
            q_curr = _stack_col(ep_df, curr_q_key)[:, :6]
            eef_curr = _align_quaternions(_stack_col(ep_df, curr_eef_key)[:, :7])

            # 2. 提取目标状态
            q_tgt = _stack_col(ep_df, tgt_q_key)[:, :6] # 作为计算误差的 GT
            eef_tgt = _align_quaternions(_stack_col(ep_df, tgt_eef_key)[:, :7])

            # 3. 组装 20 维观测序列
            obs_seq = np.concatenate([q_curr, eef_curr, eef_tgt], axis=-1)
            
            # 如果开启了结构编码，将特征广播到时间维度并拼接
            if use_robot_feature and robot_feature is not None:
                # robot_feature 形状是 (D_feat,)
                # 我们需要把它扩展成 (T, D_feat) 然后拼到 obs_seq 后面
                feat_seq = np.tile(robot_feature, (T, 1))
                obs_seq = np.concatenate([obs_seq, feat_seq], axis=-1)

            pred_sum = np.zeros((T, 6), dtype=np.float32)
            pred_cnt = np.zeros((T, 6), dtype=np.int32)

            # 滑动窗口预测
            for start_idx in range(0, T - n_obs_steps + 1, n_action_steps):
                t_start_pred = start_idx + n_obs_steps
                if t_start_pred >= T: break

                obs_win = obs_seq[start_idx:start_idx + n_obs_steps]
                res = policy.predict_action({"obs": torch.from_numpy(obs_win).unsqueeze(0).to(device_t)})
                delta_pred = res["action" if cfg.pred_action_steps_only else "action_pred"].cpu().numpy()[0]

                # 物理积分：把 delta_q 加到预测起点的 q_curr 上
                for k in range(delta_pred.shape[0]):
                    t = t_start_pred + k
                    if t >= T: break
                    
                    abs_q = q_curr[t] + delta_pred[k]
                    pred_sum[t] += abs_q
                    pred_cnt[t] += 1

            pred6 = _finalize_pred(pred_sum, pred_cnt)
            results.append({"episode": f"episode_{ep_id:06d}", "gt6": q_tgt.tolist(), "pred6": pred6.tolist()})

            if save_plots:
                _save_two_panel_joint_plot(output_path, f"episode_{ep_id:06d}", q_tgt, pred6, "(Abs Joint Overlay)")

    with open(output_path / "eval_results.json", "w") as f: json.dump(results, f)

    # 计算误差
    all_err = []
    for ep in results:
        gt_arr, pd_arr = np.array(ep["gt6"]), np.array(ep["pred6"])
        valid = np.isfinite(pd_arr).all(axis=1)
        if valid.sum() > 0: all_err.append(np.abs(pd_arr[valid] - gt_arr[valid]))

    if all_err:
        all_err = np.concatenate(all_err, axis=0)
        stats = {
            "mean_error": float(all_err.mean()), "std_error": float(all_err.std()), "max_error": float(all_err.max()),
            "mean_error_per_joint": all_err.mean(axis=0).tolist()
        }
        with open(output_path / "eval_stats.json", "w") as f: json.dump(stats, f, indent=2)
        print(f"评估完成 | 平均绝对关节误差: {stats['mean_error']:.6f} rad")
    else:
        print("警告: 预测结果全部为 NaN。")

if __name__ == "__main__":
    main()