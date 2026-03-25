# diffusion_policy/dataset/parquet_lowdim_dataset.py
# 中文说明：
# 本文件实现了一个专门针对 LeRobot parquet 数据的低维数据集封装 `ParquetLowdimDataset`。
# 主要功能：
#   - 从目录中按 episode_*.parquet 加载数据；
#   - 将末端位姿 gpos14 (位置 + RPY + gripper) 转成 obs16 (位置 + 四元数 + gripper)，方便 Diffusion Policy 使用；
#   - 按时间窗口 (horizon, stride) 切分为训练样本，并支持 train/val 划分与 normalizer 拟合；
#   - 支持手动指定当前数据目录对应哪台机器人，并返回 robot_feature。

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

# NEW: robot feature support
from diffusion_policy.dataset.robot_feature_utils import build_robot_feature_map
from diffusion_policy.dataset.parquet_path_utils import collect_episode_parquet_paths
from diffusion_policy.dataset.robot_specs import ROBOT_SPECS


@dataclass
class _EpisodeCache:
    obs: np.ndarray              # (N, Do)
    action: np.ndarray           # (N, Da)
    robot_name: Optional[str]    # NEW


def _rpy_to_quat_xyz(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """
    Convert RPY to quaternion (x,y,z,w) assuming intrinsic xyz (roll->pitch->yaw),
    which matches the common robotics convention and scipy Rotation.from_euler('xyz', ...).
    Inputs are shape (T,).
    Returns shape (T,4).
    """
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quat = np.stack([qx, qy, qz, qw], axis=-1).astype(np.float32)
    return quat


def _canonicalize_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    """
    Canonicalize quaternion sign to reduce discontinuity due to q ~ -q.
    Rule: enforce w >= 0 (if w < 0, flip sign).
    quat_xyzw: (T,4)
    """
    w = quat_xyzw[:, 3:4]
    sign = np.where(w < 0.0, -1.0, 1.0).astype(np.float32)
    return quat_xyzw * sign


def _gpos14_rpy_to_obs16_quat(gpos14: np.ndarray, canonicalize: bool = True) -> np.ndarray:
    """
    gpos14: (T,14) = [rx,ry,rz,rroll,rpitch,ryaw,rgrip, lx,ly,lz,lroll,lpitch,lyaw,lgrip]
    out16: (T,16) = [rxyz(3), rquat(4), rgrip(1), lxyz(3), lquat(4), lgrip(1)]
    """
    assert gpos14.ndim == 2 and gpos14.shape[1] == 14, f"Expected (T,14), got {gpos14.shape}"

    # right
    rxyz = gpos14[:, 0:3].astype(np.float32)
    rroll = gpos14[:, 3]
    rpitch = gpos14[:, 4]
    ryaw = gpos14[:, 5]
    rgrip = gpos14[:, 6:7].astype(np.float32)
    rquat = _rpy_to_quat_xyz(rroll, rpitch, ryaw)
    if canonicalize:
        rquat = _canonicalize_quat(rquat)

    # left
    lxyz = gpos14[:, 7:10].astype(np.float32)
    lroll = gpos14[:, 10]
    lpitch = gpos14[:, 11]
    lyaw = gpos14[:, 12]
    lgrip = gpos14[:, 13:14].astype(np.float32)
    lquat = _rpy_to_quat_xyz(lroll, lpitch, lyaw)
    if canonicalize:
        lquat = _canonicalize_quat(lquat)

    out = np.concatenate([rxyz, rquat, rgrip, lxyz, lquat, lgrip], axis=-1).astype(np.float32)
    return out


class ParquetLowdimDataset(BaseLowdimDataset):
    """
    LeRobot-style parquet episodes -> windows for lowdim Diffusion Policy.

    中文补充说明：
      - 观测字段 `observation.state_gpos`：形状 (14,) = [x,y,z,roll,pitch,yaw,grip] * 2 臂；
      - 动作字段 `action`：形状 (14,) = 关节角 (7 + gripper) * 2 臂；
      - 如果 `obs_repr == "quat"`，会在线把 gpos14 (位置 + RPY) 转成 obs16 (位置 + 四元数 + gripper)，
        保证训练时的 obs 维度为 16，对应 config 里的 `obs_dim: 16`。
      - 如果 `use_robot_feature=True`，则需要手动指定 `robot_name`，例如：
            robot_name="panda"
        或
            robot_name="airbot_single_arm"
        并会在 sample 中返回固定的 robot_feature。
    """

    def __init__(
        self,
        data_dir: str,
        horizon: int,
        obs_key: str = "observation.state_gpos",
        action_key: str = "action",
        episode_glob: str = "episode_*.parquet",
        scan_chunk_subdirs: bool = False,
        chunk_dir_glob: str = "chunk-*",
        stride: int = 1,
        cache_in_memory: bool = True,

        # observation representation
        obs_repr: str = "quat",             # "rpy" or "quat"
        canonicalize_quat: bool = True,     # enforce qw>=0

        # split related
        split: str = "train",               # "train" or "val"
        val_ratio: float = 0.1,
        split_by_episode: bool = True,
        seed: int = 44,

        # safety
        min_episode_len: int = 2,
        max_episodes: Optional[int] = None,

        # NEW: robot feature related (A 方案：手动指定 robot_name)
        use_robot_feature: bool = False,
        robot_name: Optional[str] = None,   # 手动指定整个 data_dir 属于哪台机器人
        max_joints: int = 16,
    ):
        super().__init__()
        assert data_dir is not None, "data_dir is required"

        self.data_dir = data_dir
        self.horizon = int(horizon)
        self.obs_key = obs_key
        self.action_key = action_key
        self.episode_glob = episode_glob
        self.scan_chunk_subdirs = bool(scan_chunk_subdirs)
        self.chunk_dir_glob = chunk_dir_glob
        self.stride = int(stride)
        self.cache_in_memory = bool(cache_in_memory)

        self.obs_repr = str(obs_repr)
        assert self.obs_repr in ["rpy", "quat"], f"obs_repr must be rpy/quat, got {self.obs_repr}"
        self.canonicalize_quat = bool(canonicalize_quat)

        self.split = str(split)
        assert self.split in ["train", "val"], f"split must be train/val, got {self.split}"
        self.val_ratio = float(val_ratio)
        self.split_by_episode = bool(split_by_episode)
        self.seed = int(seed)

        self.min_episode_len = int(min_episode_len)
        self.max_episodes = max_episodes

        # NEW
        self.use_robot_feature = bool(use_robot_feature)
        self.robot_name = robot_name
        self.max_joints = int(max_joints)
        self.robot_feature_map: Dict[str, np.ndarray] = {}

        if self.use_robot_feature:
            assert self.robot_name is not None, (
                "use_robot_feature=True requires manual robot_name in A scheme, "
                "e.g. robot_name='panda' or robot_name='airbot_single_arm'."
            )
            self.robot_feature_map = build_robot_feature_map(
                ROBOT_SPECS,
                max_joints=self.max_joints
            )
            if self.robot_name not in self.robot_feature_map:
                raise KeyError(
                    f"robot_name='{self.robot_name}' not found in ROBOT_SPECS / robot_feature_map. "
                    f"Available: {list(self.robot_feature_map.keys())}"
                )

        assert self.horizon > 0
        assert self.stride > 0
        assert 0.0 <= self.val_ratio < 1.0, "val_ratio must be in [0, 1)"

        episode_paths_all = collect_episode_parquet_paths(
            self.data_dir,
            self.episode_glob,
            scan_chunk_subdirs=self.scan_chunk_subdirs,
            chunk_dir_glob=self.chunk_dir_glob,
        )
        assert len(episode_paths_all) > 0, (
            f"No episodes found: data_dir={self.data_dir!r}, episode_glob={self.episode_glob!r}, "
            f"scan_chunk_subdirs={self.scan_chunk_subdirs}"
        )
        if self.max_episodes is not None:
            episode_paths_all = episode_paths_all[: int(self.max_episodes)]

        rng = np.random.RandomState(self.seed)

        if self.val_ratio <= 0.0:
            episode_paths = episode_paths_all
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
                episode_paths = [episode_paths_all[i] for i in keep]
            else:
                episode_paths = episode_paths_all

        self.episode_paths: List[str] = episode_paths
        assert len(self.episode_paths) > 0, (
            f"Split produced 0 episodes. split={self.split}, val_ratio={self.val_ratio}, "
            f"split_by_episode={self.split_by_episode}"
        )

        self._index: List[Tuple[int, int]] = []
        self._cache: Dict[int, _EpisodeCache] = {}
        all_windows: List[Tuple[int, int]] = []

        for local_ep_id, p in enumerate(self.episode_paths):
            df = pd.read_parquet(p, columns=[self.obs_key, self.action_key])
            n = len(df)
            if n < max(self.horizon, self.min_episode_len):
                continue

            ep_robot_name = self.robot_name if self.use_robot_feature else None

            max_start = n - self.horizon
            starts = list(range(0, max_start + 1, self.stride))

            if self.val_ratio > 0.0 and (not self.split_by_episode):
                for s in starts:
                    all_windows.append((local_ep_id, s))
            else:
                for s in starts:
                    self._index.append((local_ep_id, s))

            if self.cache_in_memory:
                obs = np.stack(df[self.obs_key].to_numpy(), axis=0).astype(np.float32)
                act = np.stack(df[self.action_key].to_numpy(), axis=0).astype(np.float32)

                obs = self._maybe_convert_obs(obs)

                self._cache[local_ep_id] = _EpisodeCache(
                    obs=obs,
                    action=act,
                    robot_name=ep_robot_name,
                )

        if self.val_ratio > 0.0 and (not self.split_by_episode):
            assert len(all_windows) > 0, "No windows to split; check horizon/episode length."
            rng.shuffle(all_windows)
            n_val = max(1, int(round(len(all_windows) * self.val_ratio)))
            if self.split == "val":
                self._index = all_windows[:n_val]
            else:
                self._index = all_windows[n_val:]

        assert len(self._index) > 0, (
            "No valid windows built. "
            "Check horizon/stride, episode lengths, or split settings."
        )

        sample0 = self[0]
        self.obs_dim = int(sample0["obs"].shape[-1])
        self.action_dim = int(sample0["action"].shape[-1])

        self.robot_feature_dim = None
        if self.use_robot_feature:
            any_feat = self.robot_feature_map[self.robot_name]
            self.robot_feature_dim = int(any_feat.shape[-1])

    def _maybe_convert_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        If obs_repr == 'quat' and obs is state_gpos-style (T,14) RPY, convert to (T,16).
        Otherwise return as-is.

        obs_repr == 'quat' supports:
        - gpos14 RPY (T,14): convert to obs16 quat
        - state_quat (T,7): already quat, optionally canonicalize sign
        """
        if self.obs_repr != "quat":
            return obs

        if obs.ndim != 2:
            raise ValueError(f"Expected obs ndim=2, got {obs.ndim}, shape={obs.shape}")

        # Case 1: old dual-arm gpos14 (RPY) -> convert to obs16
        if obs.shape[1] == 14 and self.obs_key == "observation.state_gpos":
            return _gpos14_rpy_to_obs16_quat(obs, canonicalize=self.canonicalize_quat)

        # Case 2: new single-arm state_quat -> passthrough (optional canonicalize)
        if obs.shape[1] == 7 and self.obs_key == "observation.state_quat":
            out = obs.astype(np.float32)
            if self.canonicalize_quat:
                # quaternion is [qx,qy,qz,qw] at indices 3:7
                out[:, 3:7] = _canonicalize_quat(out[:, 3:7])
            return out

        raise ValueError(
            f"obs_repr=quat got unsupported obs shape {obs.shape} for obs_key={self.obs_key}. "
            "Expected (T,14) for observation.state_gpos or (T,7) for observation.state_quat."
        )

    def _load_episode(self, local_ep_id: int) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        if local_ep_id in self._cache:
            c = self._cache[local_ep_id]
            return c.obs, c.action, c.robot_name

        p = self.episode_paths[local_ep_id]
        df = pd.read_parquet(p, columns=[self.obs_key, self.action_key])

        obs = np.stack(df[self.obs_key].to_numpy(), axis=0).astype(np.float32)
        act = np.stack(df[self.action_key].to_numpy(), axis=0).astype(np.float32)
        obs = self._maybe_convert_obs(obs)

        robot_name = self.robot_name if self.use_robot_feature else None
        return obs, act, robot_name

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        local_ep_id, start = self._index[idx]
        obs_seq, act_seq, robot_name = self._load_episode(local_ep_id)
        T = self.horizon

        obs = obs_seq[start:start + T]
        act = act_seq[start:start + T]

        if self.use_robot_feature:
            assert robot_name is not None, "robot_name is required when use_robot_feature is True"
            robot_feature = self.robot_feature_map[robot_name].astype(np.float32)   # [276]
            robot_feature_seq = np.repeat(robot_feature[None, :], repeats=T, axis=0)  # [T, 276]
            obs = np.concatenate([obs, robot_feature_seq], axis=-1)  # [T, 283]
            
        return {
            "obs": torch.from_numpy(obs).float(),
            "action": torch.from_numpy(act).float(),
        }

    def get_all_actions(self) -> torch.Tensor:
        acts = []
        for local_ep_id in range(len(self.episode_paths)):
            _, act_seq, _ = self._load_episode(local_ep_id)
            acts.append(torch.from_numpy(act_seq))
        return torch.cat(acts, dim=0).float()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        max_windows = int(kwargs.get("max_windows", 4096))
        seed = int(kwargs.get("seed", self.seed))
        rng = np.random.RandomState(seed)

        n = len(self._index)
        if n == 0:
            raise RuntimeError("No windows in dataset; cannot fit normalizer.")

        if n > max_windows:
            sel = rng.choice(n, size=max_windows, replace=False)
        else:
            sel = np.arange(n)

        obs_list, act_list = [], []
        for i in sel:
            item = self[i]
            obs_list.append(item["obs"].numpy())
            act_list.append(item["action"].numpy())

        obs_arr = np.concatenate(obs_list, axis=0)
        act_arr = np.concatenate(act_list, axis=0)

        normalizer = LinearNormalizer()
        normalizer.fit({
            "obs": torch.from_numpy(obs_arr).float(),
            "action": torch.from_numpy(act_arr).float(),
        })
        return normalizer

    def get_validation_dataset(self) -> "ParquetLowdimDataset":
        return ParquetLowdimDataset(
            data_dir=self.data_dir,
            horizon=self.horizon,
            obs_key=self.obs_key,
            action_key=self.action_key,
            episode_glob=self.episode_glob,
            scan_chunk_subdirs=self.scan_chunk_subdirs,
            chunk_dir_glob=self.chunk_dir_glob,
            stride=self.stride,
            cache_in_memory=self.cache_in_memory,
            obs_repr=self.obs_repr,
            canonicalize_quat=self.canonicalize_quat,
            split="val",
            val_ratio=self.val_ratio,
            split_by_episode=self.split_by_episode,
            seed=self.seed,
            min_episode_len=self.min_episode_len,
            max_episodes=self.max_episodes,
            use_robot_feature=self.use_robot_feature,
            robot_name=self.robot_name,
            max_joints=self.max_joints,
        )