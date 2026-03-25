from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

from diffusion_policy.dataset.robot_feature_utils import build_robot_feature_map
from diffusion_policy.dataset.robot_specs import ROBOT_SPECS
from diffusion_policy.dataset.parquet_path_utils import collect_episode_parquet_paths


@dataclass
class _EpisodeCache:
    q_curr: np.ndarray       # (N, Dq)
    q_tgt: np.ndarray        # (N, Dq)
    eef_curr: np.ndarray     # (N, 7)
    eef_tgt: np.ndarray      # (N, 7)
    robot_name: Optional[str]


class ParquetDeltaJointDataset(BaseLowdimDataset):
    """
    单臂 parquet 序列版 delta-joint 数据集：
        input[t]  = [q_curr[t], eef_curr[t], eef_tgt[t]]
        target[t] = q_tgt[t] - q_curr[t]

    最终 __getitem__ 返回：
        obs    : (horizon, obs_dim)
        action : (horizon, action_dim)

    parquet 字段默认对应：
        observation.state       -> q_curr
        action                  -> q_tgt
        observation.state_quat  -> eef_curr
        action_quat             -> eef_tgt
    """

    def __init__(
        self,
        data_dir: str,
        horizon: int,
        episode_glob: str = "episode_*.parquet",
        scan_chunk_subdirs: bool = False,
        chunk_dir_glob: str = "chunk-*",
        stride: int = 1,
        cache_in_memory: bool = True,

        # split
        split: str = "train",
        val_ratio: float = 0.1,
        split_by_episode: bool = True,
        seed: int = 44,
        max_episodes: Optional[int] = None,

        # parquet keys
        q_curr_key: str = "observation.state",
        q_tgt_key: str = "action",
        eef_curr_key: str = "observation.state_quat",
        eef_tgt_key: str = "action_quat",

        # optional robot feature
        use_robot_feature: bool = False,
        robot_name: Optional[str] = None,
        max_joints: int = 16,

        # optional angle wrap for revolute joints
        wrap_delta_action: bool = False,
    ):
        super().__init__()

        assert data_dir is not None
        assert split in ["train", "val"]
        assert 0.0 <= val_ratio < 1.0
        assert horizon > 0
        assert stride > 0

        self.data_dir = data_dir
        self.horizon = int(horizon)
        self.episode_glob = episode_glob
        self.scan_chunk_subdirs = bool(scan_chunk_subdirs)
        self.chunk_dir_glob = chunk_dir_glob
        self.stride = int(stride)
        self.cache_in_memory = bool(cache_in_memory)

        self.split = split
        self.val_ratio = float(val_ratio)
        self.split_by_episode = bool(split_by_episode)
        self.seed = int(seed)
        self.max_episodes = max_episodes

        self.q_curr_key = q_curr_key
        self.q_tgt_key = q_tgt_key
        self.eef_curr_key = eef_curr_key
        self.eef_tgt_key = eef_tgt_key

        self.use_robot_feature = bool(use_robot_feature)
        self.robot_name = robot_name
        self.max_joints = int(max_joints)
        self.wrap_delta_action = bool(wrap_delta_action)

        self.robot_feature_map: Dict[str, np.ndarray] = {}
        if self.use_robot_feature:
            assert self.robot_name is not None, "use_robot_feature=True 时必须指定 robot_name"
            self.robot_feature_map = build_robot_feature_map(
                ROBOT_SPECS,
                max_joints=self.max_joints
            )
            if self.robot_name not in self.robot_feature_map:
                raise KeyError(
                    f"robot_name='{self.robot_name}' not found. "
                    f"Available: {list(self.robot_feature_map.keys())}"
                )

        episode_paths_all = collect_episode_parquet_paths(
            self.data_dir,
            self.episode_glob,
            scan_chunk_subdirs=self.scan_chunk_subdirs,
            chunk_dir_glob=self.chunk_dir_glob,
        )
        assert len(episode_paths_all) > 0, (
            f"No parquet found. data_dir={self.data_dir!r}, scan_chunk_subdirs={self.scan_chunk_subdirs}"
        )

        if self.max_episodes is not None:
            episode_paths_all = episode_paths_all[: int(self.max_episodes)]

        rng = np.random.RandomState(self.seed)

        if self.val_ratio <= 0.0:
            self.episode_paths = episode_paths_all
        else:
            if self.split_by_episode:
                ep_indices = list(range(len(episode_paths_all)))
                rng.shuffle(ep_indices)
                n_val = max(1, int(round(len(ep_indices) * self.val_ratio)))
                val_set = set(ep_indices[:n_val])

                if self.split == "val":
                    keep = [i for i in ep_indices if i in val_set]
                else:
                    keep = [i for i in ep_indices if i not in val_set]

                self.episode_paths = [episode_paths_all[i] for i in keep]
            else:
                self.episode_paths = episode_paths_all

        assert len(self.episode_paths) > 0, f"Split produced 0 episodes. split={self.split}"

        self._cache: Dict[int, _EpisodeCache] = {}
        self._index: List[Tuple[int, int]] = []
        all_windows: List[Tuple[int, int]] = []

        required_cols = [
            self.q_curr_key,
            self.q_tgt_key,
            self.eef_curr_key,
            self.eef_tgt_key,
        ]

        for ep_id, p in enumerate(self.episode_paths):
            df = pd.read_parquet(p, columns=required_cols)
            n = len(df)
            if n < self.horizon:
                continue

            max_start = n - self.horizon
            starts = list(range(0, max_start + 1, self.stride))

            if self.val_ratio > 0.0 and (not self.split_by_episode):
                for s in starts:
                    all_windows.append((ep_id, s))
            else:
                for s in starts:
                    self._index.append((ep_id, s))

            if self.cache_in_memory:
                q_curr = np.stack(df[self.q_curr_key].to_numpy(), axis=0).astype(np.float32)
                q_tgt = np.stack(df[self.q_tgt_key].to_numpy(), axis=0).astype(np.float32)
                eef_curr = np.stack(df[self.eef_curr_key].to_numpy(), axis=0).astype(np.float32)
                eef_tgt = np.stack(df[self.eef_tgt_key].to_numpy(), axis=0).astype(np.float32)

                self._cache[ep_id] = _EpisodeCache(
                    q_curr=q_curr,
                    q_tgt=q_tgt,
                    eef_curr=eef_curr,
                    eef_tgt=eef_tgt,
                    robot_name=self.robot_name if self.use_robot_feature else None,
                )

        if self.val_ratio > 0.0 and (not self.split_by_episode):
            assert len(all_windows) > 0, "No windows to split; check horizon/episode length."
            rng.shuffle(all_windows)
            n_val = max(1, int(round(len(all_windows) * self.val_ratio)))
            if self.split == "val":
                self._index = all_windows[:n_val]
            else:
                self._index = all_windows[n_val:]

        assert len(self._index) > 0, "No valid sequence windows found."

        sample0 = self[0]
        self.obs_dim = int(sample0["obs"].shape[-1])
        self.action_dim = int(sample0["action"].shape[-1])

        self.robot_feature_dim = None
        if self.use_robot_feature:
            self.robot_feature_dim = int(self.robot_feature_map[self.robot_name].shape[-1])

    @staticmethod
    def _angle_wrap(x: np.ndarray) -> np.ndarray:
        return (x + np.pi) % (2 * np.pi) - np.pi

    def _load_episode(self, ep_id: int) -> _EpisodeCache:
        if ep_id in self._cache:
            return self._cache[ep_id]

        p = self.episode_paths[ep_id]
        df = pd.read_parquet(
            p,
            columns=[self.q_curr_key, self.q_tgt_key, self.eef_curr_key, self.eef_tgt_key]
        )

        return _EpisodeCache(
            q_curr=np.stack(df[self.q_curr_key].to_numpy(), axis=0).astype(np.float32),
            q_tgt=np.stack(df[self.q_tgt_key].to_numpy(), axis=0).astype(np.float32),
            eef_curr=np.stack(df[self.eef_curr_key].to_numpy(), axis=0).astype(np.float32),
            eef_tgt=np.stack(df[self.eef_tgt_key].to_numpy(), axis=0).astype(np.float32),
            robot_name=self.robot_name if self.use_robot_feature else None,
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_id, start = self._index[idx]
        ep = self._load_episode(ep_id)
        T = self.horizon

        q_curr = ep.q_curr[start:start + T]       # (T, Dq)
        q_tgt = ep.q_tgt[start:start + T]         # (T, Dq)
        eef_curr = ep.eef_curr[start:start + T]   # (T, 7)
        eef_tgt = ep.eef_tgt[start:start + T]     # (T, 7)

        delta_q = q_tgt - q_curr
        if self.wrap_delta_action:
            delta_q = self._angle_wrap(delta_q)

        obs = np.concatenate([q_curr, eef_curr, eef_tgt], axis=-1).astype(np.float32)

        if self.use_robot_feature:
            robot_feature = self.robot_feature_map[ep.robot_name].astype(np.float32)   # (Dr,)
            robot_feature_seq = np.repeat(robot_feature[None, :], repeats=T, axis=0)   # (T, Dr)
            obs = np.concatenate([obs, robot_feature_seq], axis=-1).astype(np.float32)

        return {
            "obs": torch.from_numpy(obs).float(),         # (T, obs_dim)
            "action": torch.from_numpy(delta_q).float(),  # (T, action_dim)
        }

    def get_all_actions(self) -> torch.Tensor:
        acts = []
        for ep_id in range(len(self.episode_paths)):
            ep = self._load_episode(ep_id)
            delta = ep.q_tgt - ep.q_curr
            if self.wrap_delta_action:
                delta = self._angle_wrap(delta)
            acts.append(torch.from_numpy(delta))
        return torch.cat(acts, dim=0).float()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        max_windows = int(kwargs.get("max_windows", 4096))
        seed = int(kwargs.get("seed", self.seed))
        rng = np.random.RandomState(seed)

        n = len(self)
        if n == 0:
            raise RuntimeError("No windows in dataset; cannot fit normalizer.")

        if n > max_windows:
            sel = rng.choice(n, size=max_windows, replace=False)
        else:
            sel = np.arange(n)

        obs_list, act_list = [], []
        for i in sel:
            item = self[i]
            obs_list.append(item["obs"].numpy())      # (T, Do)
            act_list.append(item["action"].numpy())   # (T, Da)

        obs_arr = np.concatenate(obs_list, axis=0)    # (sumT, Do)
        act_arr = np.concatenate(act_list, axis=0)    # (sumT, Da)

        normalizer = LinearNormalizer()
        normalizer.fit({
            "obs": torch.from_numpy(obs_arr).float(),
            "action": torch.from_numpy(act_arr).float(),
        })
        return normalizer

    def get_validation_dataset(self) -> "ParquetDeltaJointDataset":
        return ParquetDeltaJointDataset(
            data_dir=self.data_dir,
            horizon=self.horizon,
            episode_glob=self.episode_glob,
            scan_chunk_subdirs=self.scan_chunk_subdirs,
            chunk_dir_glob=self.chunk_dir_glob,
            stride=self.stride,
            cache_in_memory=self.cache_in_memory,
            split="val",
            val_ratio=self.val_ratio,
            split_by_episode=self.split_by_episode,
            seed=self.seed,
            max_episodes=self.max_episodes,
            q_curr_key=self.q_curr_key,
            q_tgt_key=self.q_tgt_key,
            eef_curr_key=self.eef_curr_key,
            eef_tgt_key=self.eef_tgt_key,
            use_robot_feature=self.use_robot_feature,
            robot_name=self.robot_name,
            max_joints=self.max_joints,
            wrap_delta_action=self.wrap_delta_action,
        )