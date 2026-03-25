#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eval IK ckpt on RoboTwin HDF5 dataset (dual-arm -> single-arm model run twice).

Dataset format (per episode*.hdf5):
- endpose/left_endpose  : (T,7) float32  [x,y,z,qx,qy,qz,qw]  (xyzw)
- endpose/right_endpose : (T,7) float32
- joint_action/left_arm : (T,6) float64/float32
- joint_action/right_arm: (T,6) float64/float32

We:
- obs_r = right_endpose (T,7)  (optional canonicalize qw>=0)
- obs_l = left_endpose  (T,7)
- gt12 = [right_arm(6), left_arm(6)] -> (T,12)

Time alignment same as your original eval:
- for start_idx in range(0, T-n_obs_steps+1, n_action_steps):
    t_end = start_idx + n_obs_steps - 1
    pred_block aligns to t = t_end + k
Scatter-average overlapping predictions.
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import os
import glob
import json
import pathlib
from typing import List, Tuple

import click
import hydra
import torch
import dill
import numpy as np
import h5py
from omegaconf import OmegaConf
from tqdm import tqdm

from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

# -------------------------
# utils
# -------------------------
def _finalize_pred(pred_sum: np.ndarray, pred_cnt: np.ndarray) -> np.ndarray:
    pred_full = np.full_like(pred_sum, np.nan, dtype=np.float32)
    mask = pred_cnt > 0
    pred_full[mask] = (pred_sum[mask] / pred_cnt[mask]).astype(np.float32)
    return pred_full

def _canonicalize_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    """
    quat_xyzw: (...,4)
    enforce qw>=0
    """
    quat_xyzw = quat_xyzw.astype(np.float32)
    w = quat_xyzw[..., 3:4]
    sign = np.where(w < 0.0, -1.0, 1.0).astype(np.float32)
    return quat_xyzw * sign

def _load_robotwin_episode(h5_path: str,
                           canonicalize_quat: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      obs_r (T,7), obs_l (T,7), gt_r (T,6), gt_l (T,6)
    """
    with h5py.File(h5_path, "r") as f:
        obs_l = np.asarray(f["endpose/left_endpose"][:], dtype=np.float32)
        obs_r = np.asarray(f["endpose/right_endpose"][:], dtype=np.float32)
        gt_l = np.asarray(f["joint_action/left_arm"][:], dtype=np.float32)
        gt_r = np.asarray(f["joint_action/right_arm"][:], dtype=np.float32)

    assert obs_l.ndim == 2 and obs_l.shape[1] == 7
    assert obs_r.ndim == 2 and obs_r.shape[1] == 7
    assert gt_l.ndim == 2 and gt_l.shape[1] == 6
    assert gt_r.ndim == 2 and gt_r.shape[1] == 6
    T = obs_l.shape[0]
    assert obs_r.shape[0] == T and gt_l.shape[0] == T and gt_r.shape[0] == T, "Length mismatch"

    if canonicalize_quat:
        obs_l[:, 3:7] = _canonicalize_quat_xyzw(obs_l[:, 3:7])
        obs_r[:, 3:7] = _canonicalize_quat_xyzw(obs_r[:, 3:7])

    return obs_r, obs_l, gt_r, gt_l

def _save_two_panel_joint_plot(out_dir: pathlib.Path,
                               episode_name: str,
                               gt: np.ndarray,
                               pred: np.ndarray,
                               title_suffix: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    T, D = gt.shape
    t = np.arange(T)

    ncols = 2 if D > 6 else 1
    nrows = int(np.ceil(D / ncols))

    fig, axes = plt.subplots(nrows=nrows * 2, ncols=ncols, sharex=True,
                             figsize=(14, 2.2 * nrows * 2))
    axes = np.array(axes).reshape(-1)

    for j in range(D):
        ax_gt = axes[j]
        ax_gt.plot(t, gt[:, j])
        ax_gt.set_title(f"Joint {j} (GT)")
        ax_gt.set_ylabel("angle")

        ax_pd = axes[j + nrows * ncols]
        ax_pd.plot(t, pred[:, j])
        ax_pd.set_title(f"Joint {j} (Pred)")
        ax_pd.set_ylabel("angle")

    for k in range(D, nrows * ncols):
        axes[k].axis("off")
        axes[k + nrows * ncols].axis("off")

    fig.suptitle(f"{episode_name} | GT vs Pred (Two Panel) {title_suffix}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{episode_name}_joints_two_panel.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(14, 2.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for j in range(D):
        ax = axes[j]
        ax.plot(t, gt[:, j], label="gt")
        ax.plot(t, pred[:, j], label="pred")
        ax.set_title(f"Joint {j}")
        ax.legend(loc="upper right")

    for k in range(D, nrows * ncols):
        axes[k].axis("off")

    fig.suptitle(f"{episode_name} | GT vs Pred (Overlay) {title_suffix}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{episode_name}_joints_overlay.png", dpi=150)
    plt.close(fig)

def _split_episode_ids(num_episodes: int, split: str, val_ratio: float, seed: int = 44) -> np.ndarray:
    ids = np.arange(num_episodes, dtype=np.int64)
    if val_ratio <= 0.0:
        return ids
    rng = np.random.RandomState(seed)
    perm = ids.copy()
    rng.shuffle(perm)
    n_val = max(1, int(round(len(ids) * val_ratio)))
    val_ids = np.sort(perm[:n_val])
    trn_ids = np.sort(perm[n_val:])
    return val_ids if split.lower() == "val" else trn_ids

# -------------------------
# main
# -------------------------
@click.command()
@click.option("--checkpoint", "-c", required=True, type=str, help="模型checkpoint路径")
@click.option("--output-dir", "-o", required=True, type=str, help="输出目录")
@click.option("--root", "-r", required=True, type=str, help="Robotwin dataset root (eef_data_collection_demo)")
@click.option("--device", default="cuda:0", type=str, help="设备")
@click.option("--split", default="val", type=str, help="train/val（按 episode id 切分）")
@click.option("--val-ratio", default=0.1, type=float, help="val episode 比例")
@click.option("--max-episodes", default=5, type=int, help="最大评估episode数量（0表示全部）")
@click.option("--canonicalize-quat/--no-canonicalize-quat", default=True, help="是否做 qw>=0 canonicalize")
@click.option("--save-plots/--no-save-plots", default=True, help="是否保存GT vs Pred图表（png）")
def main(checkpoint, output_dir, root, device, split, val_ratio, max_episodes,
         canonicalize_quat, save_plots):

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"加载checkpoint: {checkpoint}")
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    obs_dim = int(cfg.policy.obs_dim)
    action_dim = int(cfg.policy.action_dim)
    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)

    print(f"[CKPT] obs_dim={obs_dim}, action_dim={action_dim}, n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}")
    if obs_dim != 7:
        raise ValueError(f"This eval expects obs_dim=7 (xyz+quat). Got obs_dim={obs_dim}")
    if action_dim != 6:
        raise ValueError(f"This eval expects action_dim=6 (joints only). Got action_dim={action_dim}")

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=str(output_path))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device_t = torch.device(device)
    policy.to(device_t).eval()
    print(f"模型已加载到设备: {device_t}")

    data_dir = os.path.join(root, "data")
    h5_paths = sorted(glob.glob(os.path.join(data_dir, "episode*.hdf5")))
    if not h5_paths:
        raise FileNotFoundError(f"No episode*.hdf5 under: {data_dir}")

    # split by episode id (0..N-1 in filename order)
    all_ids = np.arange(len(h5_paths), dtype=np.int64)
    split_ids = _split_episode_ids(len(h5_paths), split=split, val_ratio=val_ratio, seed=44)
    if max_episodes and max_episodes > 0:
        split_ids = split_ids[:max_episodes]

    print(f"Robotwin root: {root}")
    print(f"Total episodes found: {len(h5_paths)}")
    print(f"Eval split: {split} (val_ratio={val_ratio}) -> evaluating {len(split_ids)} episodes")
    print(f"Episode indices (by sorted filenames): {split_ids.tolist()}")

    results = []
    print("\n开始评估...")

    with torch.no_grad():
        for local_i, ep_idx in enumerate(tqdm(split_ids, desc="评估中")):
            h5p = h5_paths[int(ep_idx)]
            episode_name = os.path.splitext(os.path.basename(h5p))[0]  # episode0
            obs_r, obs_l, gt_r6, gt_l6 = _load_robotwin_episode(h5p, canonicalize_quat=canonicalize_quat)

            T = obs_r.shape[0]
            if T < n_obs_steps:
                print(f"警告: {episode_name} 太短 T={T}，跳过")
                continue

            gt12 = np.concatenate([gt_r6, gt_l6], axis=-1).astype(np.float32)  # (T,12)

            D = 12
            pred_sum = np.zeros((T, D), dtype=np.float32)
            pred_cnt = np.zeros((T, D), dtype=np.int32)

            episode_results = {
                "episode": episode_name,
                "episode_idx": int(local_i),
                "episode_file_index": int(ep_idx),
                "h5_path": h5p,
                "predictions": [],
                "note": "obs from endpose/*_endpose (xyz+quat xyzw), GT from joint_action/*_arm (6+6).",
            }

            # same alignment as original script
            for start_idx in range(0, T - n_obs_steps + 1, n_action_steps):
                t_end = start_idx + n_obs_steps - 1

                obs_win_r = obs_r[start_idx:start_idx + n_obs_steps]
                pred_r = policy.predict_action({"obs": torch.from_numpy(obs_win_r).unsqueeze(0).to(device_t)})["action"]
                pred_r = pred_r.detach().cpu().numpy()[0].astype(np.float32)  # (n_action_steps,6) or (H,6)

                obs_win_l = obs_l[start_idx:start_idx + n_obs_steps]
                pred_l = policy.predict_action({"obs": torch.from_numpy(obs_win_l).unsqueeze(0).to(device_t)})["action"]
                pred_l = pred_l.detach().cpu().numpy()[0].astype(np.float32)

                if pred_r.shape[1] < 6 or pred_l.shape[1] < 6:
                    raise ValueError(f"Unexpected output dims: pred_r={pred_r.shape}, pred_l={pred_l.shape}")

                pred_block = np.concatenate([pred_r[:, :6], pred_l[:, :6]], axis=-1).astype(np.float32)

                for k in range(pred_block.shape[0]):
                    t = t_end + k
                    if t >= T:
                        break
                    pred_sum[t] += pred_block[k]
                    pred_cnt[t] += 1

                episode_results["predictions"].append({
                    "start_idx": int(start_idx),
                    "t_end": int(t_end),
                    "action12": pred_block.tolist(),
                })

            pred12 = _finalize_pred(pred_sum, pred_cnt)

            episode_results["gt12"] = gt12.tolist()
            episode_results["pred12"] = pred12.tolist()
            results.append(episode_results)

            if save_plots:
                _save_two_panel_joint_plot(
                    out_dir=output_path,
                    episode_name=episode_name,
                    gt=gt12,
                    pred=pred12,
                    title_suffix="(input=endpose/*_endpose -> obs7 xyz+quat, gt=joint_action/*_arm)"
                )

    output_file = output_path / "eval_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_file}")

    # stats
    print("\n计算误差统计（time-aligned, 12 dims）...")
    all_err = []
    valid_cnt = 0
    for ep in results:
        gt_arr = np.array(ep["gt12"], dtype=np.float32)
        pd_arr = np.array(ep["pred12"], dtype=np.float32)
        valid = np.isfinite(pd_arr).all(axis=1)
        valid_cnt += int(valid.sum())
        if valid.sum() == 0:
            continue
        err = np.abs(pd_arr[valid] - gt_arr[valid])
        all_err.append(err)

    if len(all_err) > 0:
        all_err = np.concatenate(all_err, axis=0)
        stats = {
            "num_episodes": len(results),
            "num_valid_frames": int(valid_cnt),
            "mean_error": float(all_err.mean()),
            "std_error": float(all_err.std()),
            "max_error": float(all_err.max()),
            "mean_error_per_joint": all_err.mean(axis=0).tolist(),
            "note": "Errors on aligned pred12 vs gt12. No grippers here (GT is 6+6 already)."
        }
        stats_file = output_path / "eval_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"平均误差: {stats['mean_error']:.6f}")
        print(f"标准差: {stats['std_error']:.6f}")
        print(f"最大误差: {stats['max_error']:.6f}")
        print(f"统计信息已保存到: {stats_file}")
    else:
        print("警告: 没有找到可用于计算误差的数据（可能pred全是NaN）")

    print("\n评估完成！")

if __name__ == "__main__":
    main()
