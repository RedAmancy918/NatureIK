"""
IKParquetDataset
================
专为 IK Diffusion Transformer 设计的数据集。

输入格式（每条 episode parquet 文件包含）：
  - observation.state_quat：末端位姿（N, 7）= xyz + quat
  - observation.state 或 action：关节角（N, 6）

与 ParquetDeltaJointDataset 的主要区别
--------------------------------------
- action = 绝对关节角（不是 delta_q）
- obs = 末端位姿 7 维（不是拼接的 20 维）
- horizon = 1（单步 IK，不做时序预测）
- 使用简化的 get_validation_dataset（按窗口随机划分）

四元数规范化
-----------
qw（索引 6）< 0 时，整行四元数取反，保持半球一致性。

支持多 chunk 目录（scan_chunk_subdirs=True）。
"""
from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.parquet_path_utils import collect_episode_parquet_paths


class IKParquetDataset(Dataset):
    """
    IK Diffusion Transformer 数据集。

    参数
    ----
    data_dir: str
        数据目录（单 chunk 直接含 parquet，或多 chunk 父目录）。
    scan_chunk_subdirs: bool
        True 时扫 data_dir/chunk-*/episode_*.parquet；
        False 时扫 data_dir/episode_*.parquet。
    chunk_dir_glob: str
        chunk 子目录名（默认 chunk-*）。
    horizon: int
        固定为 1（IK 是单步预测）。
    seed: int
        随机种子，用于 train/val 划分。
    val_ratio: float
        验证集比例（按 episode 划分）。
    max_episodes: int | None
        限制加载的最大 episode 数。
    ee_key: str
        末端位姿列名。
    joint_key: str
        关节角列名（优先级高于 fallback_joint_key）。
    fallback_joint_key: str
        当 joint_key 不存在时的备用列名。
    """

    def __init__(
        self,
        data_dir: str,
        scan_chunk_subdirs: bool = False,
        chunk_dir_glob: str = "chunk-*",
        horizon: int = 1,
        seed: int = 44,
        val_ratio: float = 0.1,
        max_episodes: Optional[int] = None,
        ee_key: str = "observation.state_quat",
        joint_key: str = "observation.state",
        fallback_joint_key: str = "action",
        split: str = "train",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.scan_chunk_subdirs = scan_chunk_subdirs
        self.chunk_dir_glob = chunk_dir_glob
        self.horizon = int(horizon)
        self.seed = seed
        self.val_ratio = float(val_ratio)
        self.max_episodes = max_episodes
        self.ee_key = ee_key
        self.joint_key = joint_key
        self.fallback_joint_key = fallback_joint_key
        self.split = split

        # 收集所有 parquet 路径
        files = collect_episode_parquet_paths(
            data_dir,
            "episode_*.parquet",
            scan_chunk_subdirs=scan_chunk_subdirs,
            chunk_dir_glob=chunk_dir_glob,
        )
        if not files:
            raise RuntimeError(f"未找到任何 parquet 文件: data_dir={data_dir}")

        if max_episodes is not None:
            files = files[:int(max_episodes)]

        print(f"[IKParquetDataset] 发现 {len(files)} 个 episode，split={split}...")

        # ------ 按 episode 划分 train/val ------
        rng = np.random.RandomState(seed)
        ep_indices = list(range(len(files)))
        rng.shuffle(ep_indices)
        n_val = max(1, int(round(len(ep_indices) * val_ratio))) if val_ratio > 0 else 0
        val_set = set(ep_indices[:n_val])

        if split == "val":
            keep = [i for i in ep_indices if i in val_set]
        else:
            keep = [i for i in ep_indices if i not in val_set]

        self.episode_files = [files[i] for i in keep]
        assert len(self.episode_files) > 0, f"Split={split} 产生了 0 个 episode，检查 val_ratio。"

        # ------ 加载数据 ------
        self._samples: List[Tuple[np.ndarray, np.ndarray]] = []  # (ee_7, joint_6)
        n_loaded = 0

        for fp in self.episode_files:
            df = pd.read_parquet(fp)

            # 末端位姿
            if self.ee_key not in df.columns:
                raise KeyError(f"文件 {fp} 缺少列 '{self.ee_key}'")
            try:
                ee_data = np.stack(df[self.ee_key].values).astype(np.float32)
            except ValueError:
                ee_data = np.array(df[self.ee_key].tolist(), dtype=np.float32)

            if ee_data.ndim == 1:
                ee_data = ee_data.reshape(1, -1)
            if ee_data.shape[1] != 7:
                raise ValueError(f"末端位姿维度错误，期望 7，实际 {ee_data.shape[1]}（{fp}）")

            # 四元数规范化（qw<0 → 取反）
            flip_mask = ee_data[:, 6] < 0
            ee_data[flip_mask, 3:7] *= -1

            # 关节角
            col = self.joint_key if self.joint_key in df.columns else self.fallback_joint_key
            if col not in df.columns:
                raise KeyError(f"文件 {fp} 缺少列 '{self.joint_key}' 和 '{self.fallback_joint_key}'")
            try:
                jt_data = np.stack(df[col].values).astype(np.float32)
            except ValueError:
                jt_data = np.array(df[col].tolist(), dtype=np.float32)

            if jt_data.ndim == 1:
                jt_data = jt_data.reshape(1, -1)
            if jt_data.shape[1] < 6:
                raise ValueError(f"关节角维度错误，期望 >=6，实际 {jt_data.shape[1]}（{fp}）")

            # 截取前 6 维臂关节
            jt_data = jt_data[:, :6]

            assert len(ee_data) == len(jt_data), (
                f"长度不一致: ee={len(ee_data)}, joint={len(jt_data)} in {fp}"
            )

            for i in range(len(ee_data)):
                self._samples.append((ee_data[i], jt_data[i]))

            n_loaded += 1

        print(f"[IKParquetDataset] 加载完成: {n_loaded} 个 episode，{len(self._samples)} 个样本。")

        # ------ 归一化 ------
        all_ee = np.stack([s[0] for s in self._samples])   # (N, 7)
        all_jt = np.stack([s[1] for s in self._samples])   # (N, 6)

        self._normalizer = LinearNormalizer()
        self._normalizer.fit({
            'obs/state_quat': torch.from_numpy(all_ee),
            'action': torch.from_numpy(all_jt),
        })

    # ------------------------------------------------------------------
    def get_normalizer(self) -> LinearNormalizer:
        return copy.deepcopy(self._normalizer)

    def get_validation_dataset(self) -> "IKParquetDataset":
        return IKParquetDataset(
            data_dir=self.data_dir,
            scan_chunk_subdirs=self.scan_chunk_subdirs,
            chunk_dir_glob=self.chunk_dir_glob,
            horizon=self.horizon,
            seed=self.seed,
            val_ratio=self.val_ratio,
            max_episodes=self.max_episodes,
            ee_key=self.ee_key,
            joint_key=self.joint_key,
            fallback_joint_key=self.fallback_joint_key,
            split="val",
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ee, jt = self._samples[idx]
        # obs 包在 {'state_quat': (1, 7)} 里，与 Policy.compute_loss 的 batch 格式对齐
        return {
            'obs': {
                'state_quat': torch.from_numpy(ee).float().unsqueeze(0),  # (1, 7)
            },
            'action': torch.from_numpy(jt).float().unsqueeze(0),          # (1, 6)
        }
