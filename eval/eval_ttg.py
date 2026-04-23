#!/usr/bin/env python3
"""TTG vs No-TTG 消融验证脚本

针对未参与训练的 unseen parquet 数据集，对比：
  - Baseline：MIP / NatureIK 直接推理（无 TTG）
  - TTG：推理后加 FK 梯度引导

输出：
  1. experiments/results/ttg_eval_{timestamp}.csv   —— 逐 episode 指标
  2. experiments/results/ttg_eval_{timestamp}.png   —— 三行轨迹对比图（可选，按 episode 抽样）
  3. 终端打印汇总表 + 分层成功率

用法示例：
  # MIP solver
  python eval/eval_ttg.py \\
      --solver mip \\
      --ckpt path/to/mip_bundle.pt \\
      --urdf diffusion_policy/urdf_data/panda.urdf \\
      --ee_link link_eef \\
      --data_dir path/to/unseen_dataset \\
      --scan_chunks

  # NatureIK (Diffusion ResNet) solver
  python eval/eval_ttg.py \\
      --solver diffusion \\
      --ckpt path/to/diffusion.ckpt \\
      --urdf diffusion_policy/urdf_data/panda.urdf \\
      --data_dir path/to/unseen_dataset
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ── 路径配置 ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from diffusion_policy.model.differentiable_ik_model import DifferentiableIKHelper, TTGLoss
from diffusion_policy.dataset.parquet_path_utils import collect_episode_parquet_paths

# ── 常量 ──────────────────────────────────────────────────────────────────────
SUCCESS_THRESH_MM = [5.0, 10.0, 20.0]   # 位置误差阈值（毫米）
RESULTS_DIR = _ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═════════════════════════════════════════════════════════════════════════════

def _align_quat(arr: np.ndarray) -> np.ndarray:
    """w < 0 时翻转四元数到上半球，与训练预处理保持一致。"""
    arr = arr.copy()
    if arr.ndim == 1:
        if arr[6] < 0:
            arr[3:7] = -arr[3:7]
    else:
        mask = arr[:, 6] < 0
        arr[mask, 3:7] = -arr[mask, 3:7]
    return arr


def load_episode(path: str) -> dict | None:
    """加载单个 parquet episode，返回 numpy 字典；缺少必要列时返回 None。"""
    df = pd.read_parquet(path)
    required = ["observation.state", "action", "observation.state_quat", "action_quat"]
    if not all(c in df.columns for c in required):
        return None

    def _col(key, n):
        arr = np.stack(df[key].to_numpy()).astype(np.float32)
        return arr[:, :n]

    q_curr = _col("observation.state", 6)
    q_tgt  = _col("action", 6)          # 绝对关节角目标（非 delta）
    return {
        "q_curr":   q_curr,
        "q_tgt":    q_tgt,
        "delta_q":  q_tgt - q_curr,     # 真实 delta_q，用于关节空间误差评估
        "eef_curr": _align_quat(_col("observation.state_quat", 7)),
        "eef_tgt":  _align_quat(_col("action_quat", 7)),
    }


def pos_error_mm(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    """逐帧位置误差（毫米）。"""
    return np.linalg.norm(pred_xyz - gt_xyz, axis=-1) * 1000.0


def rot_error_deg(pred_rot: np.ndarray, gt_quat: np.ndarray) -> np.ndarray:
    """用四元数点积近似旋转误差（度）。pred_rot: (T,3,3)  gt_quat: (T,7) xyz+quat"""
    # 从 FK 旋转矩阵提取四元数（简化：用迹算旋转角）
    trace = pred_rot[:, 0, 0] + pred_rot[:, 1, 1] + pred_rot[:, 2, 2]
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


# ═════════════════════════════════════════════════════════════════════════════
# Solver 包装器（直接调用类，不走 HTTP）
# ═════════════════════════════════════════════════════════════════════════════

class _MIPWrapper:
    """轻量 MIP 推理包装，支持 TTG 开关。"""

    def __init__(self, ckpt: str, urdf: str, ee_link: str, use_ttg: bool,
                 ttg_lr: float, ttg_steps: int, obs_steps: int,
                 ttg_lambda_pose: float = 1.0,
                 ttg_lambda_hist: float = 0.3,
                 ttg_lambda_smooth: float = 0.05):
        from ik_server import MIPIKSolver
        self.solver_no  = MIPIKSolver(ckpt=ckpt, use_ttg=False)
        self.solver_ttg = MIPIKSolver(
            ckpt=ckpt, urdf_path=urdf, ee_link=ee_link,
            use_ttg=use_ttg, ttg_lr=ttg_lr, ttg_steps=ttg_steps,
            ttg_lambda_pose=ttg_lambda_pose,
            ttg_lambda_hist=ttg_lambda_hist,
            ttg_lambda_smooth=ttg_lambda_smooth,
        )

    def reset(self):
        """清空历史 buffer（每条 episode 开始前调用）。"""
        from collections import deque
        obs_steps = self.solver_no.obs_steps
        self.solver_no._history_l  = deque(maxlen=obs_steps)
        self.solver_no._history_r  = deque(maxlen=obs_steps)
        self.solver_no._dq_prev_l  = None
        self.solver_ttg._history_l = deque(maxlen=obs_steps)
        self.solver_ttg._history_r = deque(maxlen=obs_steps)
        self.solver_ttg._dq_prev_l = None

    def infer_step(self, q_curr, ee_curr, ee_tgt):
        """返回 (dq_no_ttg, dq_ttg, t_no_ttg_ms, t_ttg_ms)。"""
        t0 = time.perf_counter()
        dq_no  = self.solver_no.solve_arm(q_curr, ee_curr, ee_tgt)
        t_no   = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dq_ttg = self.solver_ttg.solve_arm(q_curr, ee_curr, ee_tgt)
        t_ttg  = (time.perf_counter() - t0) * 1000

        return dq_no, dq_ttg, t_no, t_ttg


class _DiffusionWrapper:
    """轻量 NatureIK (Diffusion ResNet) 推理包装，支持 TTG 开关。"""

    def __init__(self, ckpt: str, urdf: str, ee_link: str, robot_name: str,
                 use_ttg: bool, guidance_scale: float,
                 ttg_lambda_pose: float = 1.0,
                 ttg_lambda_hist: float = 0.3,
                 ttg_lambda_smooth: float = 0.05):
        from ik_server import NatureIKSolver
        self.solver_no  = NatureIKSolver(ckpt=ckpt, robot_name=robot_name, use_ttg=False)
        self.solver_ttg = NatureIKSolver(
            ckpt=ckpt, robot_name=robot_name,
            urdf_path=urdf, ee_link=ee_link,
            use_ttg=use_ttg, guidance_scale=guidance_scale,
            ttg_lambda_pose=ttg_lambda_pose,
            ttg_lambda_hist=ttg_lambda_hist,
            ttg_lambda_smooth=ttg_lambda_smooth,
        )

    def reset(self):
        self.solver_no._dq_prev_l  = None
        self.solver_no._dq_prev_r  = None
        self.solver_ttg._dq_prev_l = None
        self.solver_ttg._dq_prev_r = None

    def infer_step(self, q_curr, ee_curr, ee_tgt):
        t0 = time.perf_counter()
        dq_no  = self.solver_no.solve_arm(q_curr, ee_curr, ee_tgt)
        t_no   = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dq_ttg = self.solver_ttg.solve_arm(q_curr, ee_curr, ee_tgt)
        t_ttg  = (time.perf_counter() - t0) * 1000

        return dq_no, dq_ttg, t_no, t_ttg


# ═════════════════════════════════════════════════════════════════════════════
# 单 episode 评测
# ═════════════════════════════════════════════════════════════════════════════

def eval_episode(ep: dict, wrapper, fk: DifferentiableIKHelper, device: str):
    """
    返回：
        result_no  : dict  逐帧结果（无 TTG）
        result_ttg : dict  逐帧结果（有 TTG）
    每个 dict 包含：
        eef_pred_xyz : (T, 3)
        eef_pred_rot : (T, 3, 3)
        pos_err_mm   : (T,)
        rot_err_deg  : (T,)
        latency_ms   : (T,)
    """
    T = len(ep["q_curr"])
    wrapper.reset()

    # 累积结果
    res_no  = {"eef_xyz": [], "eef_rot": [], "latency_ms": []}
    res_ttg = {"eef_xyz": [], "eef_rot": [], "latency_ms": []}

    for t in range(T):
        q_c  = ep["q_curr"][t]
        ec   = ep["eef_curr"][t]
        et   = ep["eef_tgt"][t]

        dq_no, dq_ttg, t_no, t_ttg = wrapper.infer_step(q_c, ec, et)

        # FK: 模型输出 delta_q，加上 q_curr 得到绝对关节角目标
        for dq, res, lat in [(dq_no, res_no, t_no), (dq_ttg, res_ttg, t_ttg)]:
            q_pred = torch.from_numpy((q_c + dq).astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                pos, rot = fk.get_current_pose(q_pred)
            res["eef_xyz"].append(pos.cpu().numpy()[0])    # (3,)
            res["eef_rot"].append(rot.cpu().numpy()[0])    # (3,3)
            res["latency_ms"].append(lat)

    gt_xyz = ep["eef_tgt"][:, :3]     # (T, 3)
    gt_quat = ep["eef_tgt"]            # (T, 7)

    def _finalize(res):
        xyz = np.stack(res["eef_xyz"])     # (T, 3)
        rot = np.stack(res["eef_rot"])     # (T, 3, 3)
        return {
            "eef_pred_xyz": xyz,
            "eef_pred_rot": rot,
            "pos_err_mm":   pos_error_mm(xyz, gt_xyz),
            "rot_err_deg":  rot_error_deg(rot, gt_quat),
            "latency_ms":   np.array(res["latency_ms"]),
        }

    return _finalize(res_no), _finalize(res_ttg)


# ═════════════════════════════════════════════════════════════════════════════
# 可视化：三行轨迹对比图
# ═════════════════════════════════════════════════════════════════════════════

def plot_episode(ep: dict, res_no: dict, res_ttg: dict, out_path: str, ep_name: str):
    """
    三行图：
      Row 0: GT eef_tgt xyz
      Row 1: 无 TTG 预测 eef xyz  + 误差曲线
      Row 2: 有 TTG 预测 eef xyz  + 误差曲线
    每行左侧为 xyz 轨迹，右侧为位置误差（mm）。
    """
    T     = len(ep["eef_tgt"])
    ts    = np.arange(T)
    gt    = ep["eef_tgt"][:, :3]
    labels = ["X (m)", "Y (m)", "Z (m)"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    # 全局 y 轴范围（三行 xyz 用同一尺度）
    all_xyz = np.concatenate([gt, res_no["eef_pred_xyz"], res_ttg["eef_pred_xyz"]], axis=0)
    y_min, y_max = all_xyz.min() - 0.02, all_xyz.max() + 0.02

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(f"TTG Evaluation — {ep_name}", fontsize=14, fontweight="bold")

    row_data = [
        ("GT EEF Target",        gt,                        None,                   None),
        ("No TTG (Baseline)",     res_no["eef_pred_xyz"],    res_no["pos_err_mm"],   res_no["latency_ms"]),
        ("With TTG",              res_ttg["eef_pred_xyz"],   res_ttg["pos_err_mm"],  res_ttg["latency_ms"]),
    ]

    for row, (title, xyz, err, lat) in enumerate(row_data):
        # xyz 轨迹（前3列）
        for col, (lbl, clr) in enumerate(zip(labels, colors)):
            ax = axes[row][col]
            ax.plot(ts, xyz[:, col], color=clr, lw=1.5, label="pred" if row > 0 else "gt")
            if row > 0:
                ax.plot(ts, gt[:, col], color=clr, lw=1.0, alpha=0.4, ls="--", label="gt")
            ax.set_ylim(y_min, y_max)
            ax.set_ylabel(lbl, fontsize=9)
            ax.set_title(f"{title} — {lbl}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if row == 2:
                ax.set_xlabel("Step")

        # 右侧：误差 / 延迟
        ax_r = axes[row][3]
        if err is not None:
            ax_r.plot(ts, err, color="#e67e22", lw=1.5, label="pos err (mm)")
            ax_r.axhline(5,  ls="--", color="gray", lw=0.8, label="5mm")
            ax_r.axhline(10, ls=":",  color="gray", lw=0.8, label="10mm")
            ax_r.set_ylabel("Position Error (mm)", fontsize=9)
            ax_r.set_title(f"{title} — Pos Error", fontsize=9)
            ax_r.legend(fontsize=7)
            ax_r.grid(True, alpha=0.3)
            if row == 2:
                ax_r.set_xlabel("Step")
        else:
            # GT 行右侧显示 xyz 数值范围信息
            ax_r.axis("off")
            ax_r.text(0.5, 0.5, f"GT xyz range\n"
                      f"X: [{gt[:,0].min():.3f}, {gt[:,0].max():.3f}]\n"
                      f"Y: [{gt[:,1].min():.3f}, {gt[:,1].max():.3f}]\n"
                      f"Z: [{gt[:,2].min():.3f}, {gt[:,2].max():.3f}]",
                      ha="center", va="center", fontsize=9,
                      transform=ax_r.transAxes)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# 分层成功率图（按 OOD 距离）
# ═════════════════════════════════════════════════════════════════════════════

def plot_stratified(df: pd.DataFrame, out_path: str):
    """
    按 eef_tgt 距训练集中心的距离分层（用 gt_pos_range 近似），
    画出各分层的成功率 bar chart（5mm 阈值）。
    """
    if "dist_bin" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TTG vs No-TTG — Stratified by OOD Distance", fontsize=13)

    for ax, thresh in zip(axes, [5.0, 10.0]):
        col_no  = f"succ_{int(thresh)}mm_no"
        col_ttg = f"succ_{int(thresh)}mm_ttg"
        if col_no not in df.columns:
            continue
        grp = df.groupby("dist_bin")[[col_no, col_ttg]].mean() * 100

        x     = np.arange(len(grp))
        width = 0.35
        ax.bar(x - width/2, grp[col_no],  width, label="No TTG",  color="#95a5a6", alpha=0.85)
        ax.bar(x + width/2, grp[col_ttg], width, label="With TTG", color="#2980b9", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(grp.index, rotation=15, fontsize=9)
        ax.set_ylabel("Success Rate (%)")
        ax.set_title(f"Success @ {thresh}mm")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TTG vs No-TTG evaluation on unseen data")
    parser.add_argument("--solver",       choices=["mip", "diffusion"], required=True)
    parser.add_argument("--ckpt",         required=True,  help="模型 checkpoint 路径")
    parser.add_argument("--urdf",         required=True,  help="URDF 路径（FK 用）")
    parser.add_argument("--ee_link",      default="end_link")
    parser.add_argument("--base_offset",  type=float, nargs=3, default=None,
                        help="机器人基座在世界坐标系的平移偏移 x y z（米），用于 FK 坐标系对齐。"
                             "airbot G2 实测约：0.097 -0.006 -0.091")
    parser.add_argument("--robot",        default="airbot_single_arm", help="NatureIK 用")
    parser.add_argument("--data_dir",     required=True,  help="unseen 数据集根目录")
    parser.add_argument("--scan_chunks",   action="store_true", help="是否扫描 chunk-* 子目录")
    parser.add_argument("--episode_glob", default="episode_*.parquet", help="parquet 文件名 glob（默认 episode_*.parquet）")
    parser.add_argument("--max_episodes", type=int, default=None, help="最多评测几条 episode")
    parser.add_argument("--plot_episodes",type=int, default=3,   help="生成三行图的 episode 数量（0=不画）")

    # TTG 参数
    parser.add_argument("--ttg_lr",            type=float, default=0.02)
    parser.add_argument("--ttg_steps",         type=int,   default=5)
    parser.add_argument("--guidance_scale",    type=float, default=10.0)
    parser.add_argument("--ttg_lambda_pose",   type=float, default=1.0)
    parser.add_argument("--ttg_lambda_hist",   type=float, default=0.3)
    parser.add_argument("--ttg_lambda_smooth", type=float, default=0.05)

    # 训练集中心（用于计算 OOD 距离分层）
    parser.add_argument("--train_center",      type=float, nargs=3, default=None,
                        help="训练集 eef_tgt xyz 均值，用于 OOD 分层（可选）")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 新建本次实验专属目录 ───────────────────────────────────────────────
    exp_name = f"{args.solver}_ttg_{ts}"
    exp_dir  = _ROOT / "experiments" / "results" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] 实验目录: {exp_dir}")

    # ── 加载 FK helper ─────────────────────────────────────────────────────
    print(f"[*] 加载 FK helper: {args.urdf}")
    fk = DifferentiableIKHelper(
        urdf_path=args.urdf,
        end_effector_link_name=args.ee_link,
        base_offset=args.base_offset,
        device=device,
    )

    # ── 加载 solver ────────────────────────────────────────────────────────
    print(f"[*] 初始化 solver: {args.solver}")
    if args.solver == "mip":
        wrapper = _MIPWrapper(
            ckpt=args.ckpt, urdf=args.urdf, ee_link=args.ee_link,
            use_ttg=True, ttg_lr=args.ttg_lr, ttg_steps=args.ttg_steps,
            obs_steps=4,
            ttg_lambda_pose=args.ttg_lambda_pose,
            ttg_lambda_hist=args.ttg_lambda_hist,
            ttg_lambda_smooth=args.ttg_lambda_smooth,
        )
    else:
        wrapper = _DiffusionWrapper(
            ckpt=args.ckpt, urdf=args.urdf, ee_link=args.ee_link,
            robot_name=args.robot, use_ttg=True,
            guidance_scale=args.guidance_scale,
            ttg_lambda_pose=args.ttg_lambda_pose,
            ttg_lambda_hist=args.ttg_lambda_hist,
            ttg_lambda_smooth=args.ttg_lambda_smooth,
        )

    # ── 收集 episode 路径 ──────────────────────────────────────────────────
    ep_paths = collect_episode_parquet_paths(
        data_dir=args.data_dir,
        episode_glob=args.episode_glob,
        scan_chunk_subdirs=args.scan_chunks,
    )
    if args.max_episodes:
        ep_paths = ep_paths[:args.max_episodes]
    print(f"[*] 找到 {len(ep_paths)} 条 episode")

    # ── 逐 episode 评测 ────────────────────────────────────────────────────
    rows = []
    plot_count = 0

    for idx, ep_path in enumerate(ep_paths):
        ep_name = Path(ep_path).stem
        ep = load_episode(ep_path)
        if ep is None:
            print(f"  [!] 跳过（缺少必要列）: {ep_path}")
            continue

        T = len(ep["q_curr"])
        print(f"  [{idx+1}/{len(ep_paths)}] {ep_name}  T={T}", end="", flush=True)

        res_no, res_ttg = eval_episode(ep, wrapper, fk, device)

        # 计算 OOD 距离（到训练集中心的平均距离）
        gt_xyz    = ep["eef_tgt"][:, :3]
        ood_dist  = float(np.linalg.norm(gt_xyz.mean(0) - (args.train_center or [0, 0, 0])))
        dist_bin  = f"<{int(ood_dist*100//10)*10+10}cm" if args.train_center else "N/A"

        # 汇总指标
        row = {
            "episode":          ep_name,
            "T":                T,
            "solver":           args.solver,
            "ood_dist_m":       round(ood_dist, 4),
            "dist_bin":         dist_bin,
            # 无 TTG
            "mean_pos_err_no_mm":   round(float(res_no["pos_err_mm"].mean()), 3),
            "final_pos_err_no_mm":  round(float(res_no["pos_err_mm"][-1]), 3),
            "mean_lat_no_ms":       round(float(res_no["latency_ms"].mean()), 2),
            # 有 TTG
            "mean_pos_err_ttg_mm":  round(float(res_ttg["pos_err_mm"].mean()), 3),
            "final_pos_err_ttg_mm": round(float(res_ttg["pos_err_mm"][-1]), 3),
            "mean_lat_ttg_ms":      round(float(res_ttg["latency_ms"].mean()), 2),
        }
        # 各阈值成功率（用最终步误差判断，符合 IK 任务定义：最终是否到达目标）
        for th in SUCCESS_THRESH_MM:
            key = int(th)
            row[f"succ_{key}mm_no"]       = int(res_no["pos_err_mm"][-1]   < th)
            row[f"succ_{key}mm_ttg"]      = int(res_ttg["pos_err_mm"][-1]  < th)
            # 额外记录均值阈值版本（用于 mean-error 分析）
            row[f"succ_{key}mm_mean_no"]  = int(res_no["pos_err_mm"].mean()  < th)
            row[f"succ_{key}mm_mean_ttg"] = int(res_ttg["pos_err_mm"].mean() < th)

        rows.append(row)
        improve = res_no["pos_err_mm"].mean() - res_ttg["pos_err_mm"].mean()
        print(f"  no_ttg={row['mean_pos_err_no_mm']:.1f}mm  "
              f"ttg={row['mean_pos_err_ttg_mm']:.1f}mm  "
              f"Δ={improve:+.1f}mm")

        # 三行可视化（前 N 条）
        if plot_count < args.plot_episodes:
            fig_path = str(exp_dir / f"traj_{ep_name}.png")
            plot_episode(ep, res_no, res_ttg, fig_path, ep_name)
            print(f"    → 图表已保存: {fig_path}")
            plot_count += 1

    if not rows:
        print("[!] 没有有效 episode，退出。")
        return

    df = pd.DataFrame(rows)

    # ── 保存逐 episode CSV ────────────────────────────────────────────────
    csv_path = exp_dir / "episodes.csv"
    df.to_csv(csv_path, index=False)

    # ── 保存本次实验配置（方便复现）──────────────────────────────────────
    import json
    cfg_path = exp_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── 汇总指标 ──────────────────────────────────────────────────────────
    summary = {
        "exp_name":              exp_name,
        "solver":                args.solver,
        "n_episodes":            len(df),
        "ee_link":               args.ee_link,
        "urdf":                  args.urdf,
        "ckpt":                  args.ckpt,
        "data_dir":              args.data_dir,
        # 位置误差
        "mean_pos_err_no_mm":    round(df["mean_pos_err_no_mm"].mean(),  2),
        "mean_pos_err_ttg_mm":   round(df["mean_pos_err_ttg_mm"].mean(), 2),
        "final_pos_err_no_mm":   round(df["final_pos_err_no_mm"].mean(), 2),
        "final_pos_err_ttg_mm":  round(df["final_pos_err_ttg_mm"].mean(),2),
        "improvement_mean_mm":   round((df["mean_pos_err_no_mm"]  - df["mean_pos_err_ttg_mm"]).mean(),  2),
        "improvement_final_mm":  round((df["final_pos_err_no_mm"] - df["final_pos_err_ttg_mm"]).mean(), 2),
        # 延迟
        "lat_no_ms":             round(df["mean_lat_no_ms"].mean(),  2),
        "lat_ttg_ms":            round(df["mean_lat_ttg_ms"].mean(), 2),
        "lat_overhead_ms":       round((df["mean_lat_ttg_ms"] - df["mean_lat_no_ms"]).mean(), 2),
        # 成功率
        **{f"sr_{int(th)}mm_no":  round(df[f"succ_{int(th)}mm_no"].mean()  * 100, 1) for th in SUCCESS_THRESH_MM},
        **{f"sr_{int(th)}mm_ttg": round(df[f"succ_{int(th)}mm_ttg"].mean() * 100, 1) for th in SUCCESS_THRESH_MM},
        "timestamp":             ts,
    }
    summary_path = exp_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 写入 registry.csv（追加，有表头自动跳过）──────────────────────────
    registry_path = _ROOT / "experiments" / "registry.csv"
    registry_fields = list(summary.keys())
    write_header = not registry_path.exists()
    with open(registry_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=registry_fields)
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

    # ── 分层成功率图 ───────────────────────────────────────────────────────
    strat_path = str(exp_dir / "stratified.png")
    plot_stratified(df, strat_path)

    print(f"\n[*] 实验目录: {exp_dir}")
    print(f"    ├── episodes.csv    (逐 episode 详细指标)")
    print(f"    ├── summary.json    (汇总指标，直接用于论文)")
    print(f"    ├── config.json     (本次运行参数，方便复现)")
    print(f"    ├── traj_*.png      (轨迹对比图)")
    print(f"    └── stratified.png  (分层成功率图)")
    print(f"[*] 已写入 registry: {registry_path}")

    # ── 终端汇总 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Episodes evaluated : {len(df)}")
    print(f"  Solver             : {args.solver}")
    print(f"  ── Position Error — final step (用于成功率判定) ──")
    print(f"  No TTG  : {df['final_pos_err_no_mm'].mean():.2f} ± "
          f"{df['final_pos_err_no_mm'].std():.2f} mm")
    print(f"  With TTG: {df['final_pos_err_ttg_mm'].mean():.2f} ± "
          f"{df['final_pos_err_ttg_mm'].std():.2f} mm")
    improvement_final = (df["final_pos_err_no_mm"] - df["final_pos_err_ttg_mm"]).mean()
    print(f"  Improvement (final): {improvement_final:+.2f} mm")
    print(f"  ── Position Error — mean over episode ──")
    print(f"  No TTG  : {df['mean_pos_err_no_mm'].mean():.2f} ± "
          f"{df['mean_pos_err_no_mm'].std():.2f} mm")
    print(f"  With TTG: {df['mean_pos_err_ttg_mm'].mean():.2f} ± "
          f"{df['mean_pos_err_ttg_mm'].std():.2f} mm")
    improvement = (df["mean_pos_err_no_mm"] - df["mean_pos_err_ttg_mm"]).mean()
    print(f"  Improvement (mean) : {improvement:+.2f} mm")
    print(f"  ── Latency ──")
    print(f"  No TTG  : {df['mean_lat_no_ms'].mean():.1f} ms/step")
    print(f"  With TTG: {df['mean_lat_ttg_ms'].mean():.1f} ms/step")
    print(f"  TTG overhead      : {(df['mean_lat_ttg_ms'] - df['mean_lat_no_ms']).mean():.1f} ms/step")
    print(f"  ── Success Rate (最终步误差 < 阈值) ──")
    for th in SUCCESS_THRESH_MM:
        k = int(th)
        sr_no  = df[f"succ_{k}mm_no"].mean()  * 100
        sr_ttg = df[f"succ_{k}mm_ttg"].mean() * 100
        print(f"  @{th:.0f}mm  No TTG: {sr_no:.1f}%   With TTG: {sr_ttg:.1f}%  "
              f"(Δ={sr_ttg-sr_no:+.1f}pp)")
    print("=" * 60)


if __name__ == "__main__":
    main()
