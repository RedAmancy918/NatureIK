"""IK dataset from NatureIK Parquet format.

Reads LeRobot-style multi-chunk Parquet data:
  data_dir/
    chunk-000/episode_000000.parquet
    chunk-001/episode_000001.parquet
    ...

Two modes:
  - "delta_joint": obs=20D (q_curr + eef_curr + eef_tgt), action=6D delta_q
  - "absolute":    obs=7D  (eef_pose),                    action=6D absolute joints
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from mip.dataset_utils import CompositeNormalizer, IdentityNormalizer, MinMaxNormalizer
from mip.datasets.base import BaseDataset


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_parquet_dir(data_dir: str, scan_chunk_subdirs: bool = True) -> pd.DataFrame:
    """Load all parquet files from data_dir (or its chunk-* subdirs)."""
    data_dir = Path(data_dir)
    parquet_files = []

    if scan_chunk_subdirs:
        chunk_dirs = sorted(data_dir.glob("chunk-*"))
        if chunk_dirs:
            for chunk_dir in chunk_dirs:
                parquet_files.extend(sorted(chunk_dir.glob("*.parquet")))
        else:
            parquet_files = sorted(data_dir.glob("*.parquet"))
    else:
        parquet_files = sorted(data_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    logger.info(f"Loading {len(parquet_files)} parquet files from {data_dir}")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    # Normalise episode column name
    if "episode_index" not in df.columns and "episode_id" in df.columns:
        df = df.rename(columns={"episode_id": "episode_index"})

    return df


def _align_quat(q: np.ndarray) -> np.ndarray:
    """Half-sphere align: flip quaternion if w < 0 to avoid double-covering.

    NatureIK quaternion layout: [x, y, z, w]  →  w is at index 3.
    """
    q = q.copy()
    mask = q[..., 3] < 0
    q[mask] = -q[mask]
    return q


def _parse_col(col: pd.Series) -> np.ndarray:
    """Convert a column of lists/arrays (or scalar floats) to (T, D) float32."""
    first = col.iloc[0]
    if isinstance(first, (list, np.ndarray)):
        return np.stack(col.values).astype(np.float32)
    return col.values[:, None].astype(np.float32)


def _build_episodes(
    df: pd.DataFrame,
    q_key: str,
    eef_key: str,
    action_q_key: str,
    action_eef_key: str,
    q_dim: int = 6,
    eef_dim: int = 7,
) -> list[dict]:
    """Split DataFrame into per-episode arrays and apply quaternion alignment.

    Slices q_curr / action_q to the first q_dim joints (e.g. 6 arm joints,
    excluding gripper) and eef to the first eef_dim dims (xyz + quat = 7),
    matching NatureIK's ParquetDeltaJointDataset convention.
    """
    episodes = []
    for _ep_id, group in df.groupby("episode_index", sort=True):
        if "frame_index" in group.columns:
            group = group.sort_values("frame_index")

        ep = {
            "q_curr":   _parse_col(group[q_key])[:, :q_dim],        # (T, q_dim)
            "eef_curr": _parse_col(group[eef_key])[:, :eef_dim],    # (T, eef_dim)
            "action_q": _parse_col(group[action_q_key])[:, :q_dim], # (T, q_dim)
        }
        if action_eef_key and action_eef_key in group.columns:
            ep["eef_tgt"] = _parse_col(group[action_eef_key])[:, :eef_dim]  # (T, eef_dim)

        # Align quaternion components: layout is [xyz | qxyzw], quat at [3:7]
        ep["eef_curr"][:, 3:7] = _align_quat(ep["eef_curr"][:, 3:7])
        if "eef_tgt" in ep:
            ep["eef_tgt"][:, 3:7] = _align_quat(ep["eef_tgt"][:, 3:7])

        episodes.append(ep)

    return episodes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IKParquetDataset(BaseDataset):
    """Dataset for IK training from NatureIK Parquet files.

    Args:
        data_dir: Path to dataset root.  Accepts either:
            - a directory containing chunk-* subdirs (LeRobot multi-chunk layout)
            - a directory containing *.parquet files directly
        mode: "delta_joint" — obs=20D [q_curr(6)|eef_curr(7)|eef_tgt(7)],
                               action=6D delta joint angles (q_tgt - q_curr)
              "absolute"    — obs=7D  [eef_pose xyz+quat],
                               action=6D absolute joint angles
        horizon: Sliding-window length.
            Recommended: delta_joint→4, absolute→1.
        obs_steps: Number of history frames fed to the encoder (≤ horizon).
            These are the first `obs_steps` frames of each window.
        val_ratio: Fraction of episodes withheld for validation (episode-level).
        split: "train" or "val".
        scan_chunk_subdirs: Whether to recurse into chunk-* subdirectories.
        q_key: Parquet column for current joint angles.
        eef_key: Parquet column for current EEF pose (xyz + quat [x,y,z,w]).
        action_q_key: Parquet column for target joint angles.
        action_eef_key: Parquet column for target EEF pose (used in delta_joint mode).
    """

    def __init__(
        self,
        data_dir: str,
        mode: str = "delta_joint",
        horizon: int = 4,
        obs_steps: int = 2,
        val_ratio: float = 0.1,
        split: str = "train",
        scan_chunk_subdirs: bool = True,
        q_key: str = "observation.state",
        eef_key: str = "observation.state_quat",
        action_q_key: str = "action",
        action_eef_key: str = "action_quat",
    ):
        super().__init__()
        assert mode in ("delta_joint", "absolute"), \
            f"mode must be 'delta_joint' or 'absolute', got '{mode}'"
        assert split in ("train", "val"), \
            f"split must be 'train' or 'val', got '{split}'"
        assert obs_steps >= 1 and horizon >= 1, \
            f"obs_steps and horizon must be ≥ 1"

        self.mode = mode
        self.horizon = horizon
        self.obs_steps = obs_steps
        # Total frames needed per window:
        #   obs uses [t, ..., t+obs_steps-1]
        #   act uses [t+obs_steps-1, ..., t+obs_steps+horizon-2]  (aligned to last obs frame)
        self.window_size = obs_steps + horizon - 1

        # ------------------------------------------------------------------
        # Load & split data
        # ------------------------------------------------------------------
        df = _load_parquet_dir(data_dir, scan_chunk_subdirs)
        all_episodes = _build_episodes(df, q_key, eef_key, action_q_key, action_eef_key)

        n_val = max(1, int(len(all_episodes) * val_ratio))
        if split == "val":
            episodes = all_episodes[-n_val:]
        else:
            episodes = all_episodes[:-n_val]

        logger.info(
            f"IKParquetDataset [{split}]: {len(episodes)} episodes | "
            f"mode={mode} | obs_steps={obs_steps} | horizon={horizon} | "
            f"window_size={self.window_size}"
        )

        # ------------------------------------------------------------------
        # Build flat (episode_idx, start_frame) index
        # ------------------------------------------------------------------
        self._episodes = episodes
        self._index: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(episodes):
            T = len(ep["q_curr"])
            for t in range(T - self.window_size + 1):
                self._index.append((ep_idx, t))

        # ------------------------------------------------------------------
        # Build normalizers (computed once from training data)
        # ------------------------------------------------------------------
        self._normalizer = self._build_normalizer()

    # ------------------------------------------------------------------
    # Normalizer construction
    # ------------------------------------------------------------------

    def _build_normalizer(self) -> dict:
        """Compute per-component normalizers over all episodes.

        Quaternion dims  → IdentityNormalizer  (already ~unit; MinMax distorts)
        Position / joint → MinMaxNormalizer    (maps to [-1, 1])
        """
        all_obs, all_act = [], []

        for ep in self._episodes:
            q_curr   = ep["q_curr"]    # (T, 6)
            eef_curr = ep["eef_curr"]  # (T, 7)
            action_q = ep["action_q"]  # (T, 6)

            if self.mode == "delta_joint":
                eef_tgt = ep.get("eef_tgt", eef_curr)  # (T, 7)
                obs = np.concatenate([q_curr, eef_curr, eef_tgt], axis=-1)  # (T, 20)
                act = action_q - q_curr                                       # (T, 6)
            else:  # absolute
                obs = eef_curr   # (T, 7)
                act = action_q   # (T, 6)

            all_obs.append(obs)
            all_act.append(act)

        all_obs = np.concatenate(all_obs, axis=0)  # (N, obs_dim)
        all_act = np.concatenate(all_act, axis=0)  # (N, act_dim)

        if self.mode == "delta_joint":
            # Layout: [q_curr 0:6 | eef_curr_pos 6:9 | eef_curr_quat 9:13 |
            #          eef_tgt_pos 13:16 | eef_tgt_quat 16:20]
            obs_norm = CompositeNormalizer(
                normalizers=[
                    MinMaxNormalizer(all_obs[:, 0:6]),    # q_curr joints
                    MinMaxNormalizer(all_obs[:, 6:9]),    # eef_curr xyz
                    IdentityNormalizer(),                  # eef_curr quat (9:13)
                    MinMaxNormalizer(all_obs[:, 13:16]),  # eef_tgt xyz
                    IdentityNormalizer(),                  # eef_tgt quat (16:20)
                ],
                dim_slices=[(0, 6), (6, 9), (9, 13), (13, 16), (16, 20)],
            )
        else:
            # Layout: [xyz 0:3 | quat 3:7]
            obs_norm = CompositeNormalizer(
                normalizers=[
                    MinMaxNormalizer(all_obs[:, 0:3]),  # xyz
                    IdentityNormalizer(),                 # quat (3:7)
                ],
                dim_slices=[(0, 3), (3, 7)],
            )

        return {
            "obs":    {"state": obs_norm},
            "action": MinMaxNormalizer(all_act),
        }

    # ------------------------------------------------------------------
    # Window extraction
    # ------------------------------------------------------------------

    def _get_window(self, ep: dict, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (obs, act) arrays aligned to the same 'current' frame.

        obs: frames [t,            ..., t+obs_steps-1]        shape (obs_steps, obs_dim)
        act: frames [t+obs_steps-1, ..., t+window_size-1]     shape (horizon,   act_dim)

        The last obs frame and first act frame share the same timestep,
        so the network sees history leading up to 'now' and predicts
        actions starting from 'now'.
        """
        act_start = t + self.obs_steps - 1  # index of the "current" frame

        # obs slice: obs_steps frames of history ending at act_start
        q_curr_obs   = ep["q_curr"][t : t + self.obs_steps]    # (To, 6)
        eef_curr_obs = ep["eef_curr"][t : t + self.obs_steps]  # (To, 7)

        # act slice: horizon frames starting from act_start
        q_curr_act   = ep["q_curr"][act_start : act_start + self.horizon]    # (Ta, 6)
        action_q_act = ep["action_q"][act_start : act_start + self.horizon]  # (Ta, 6)

        if self.mode == "delta_joint":
            eef_tgt_obs = ep.get("eef_tgt", ep["eef_curr"])[t : t + self.obs_steps]
            obs = np.concatenate([q_curr_obs, eef_curr_obs, eef_tgt_obs], axis=-1)  # (To, 20)
            act = action_q_act - q_curr_act                                           # (Ta,  6)
        else:
            obs = eef_curr_obs  # (To, 7)
            act = action_q_act  # (Ta, 6)

        return obs.astype(np.float32), act.astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def get_normalizer(self) -> dict:
        return self._normalizer

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, t = self._index[idx]
        obs, act = self._get_window(self._episodes[ep_idx], t)

        obs_norm = self._normalizer["obs"]["state"].normalize(obs)  # (H, obs_dim)
        act_norm = self._normalizer["action"].normalize(act)         # (H, act_dim)

        return {
            "obs":    {"state": torch.from_numpy(obs_norm)},  # (obs_steps, obs_dim)
            "action": torch.from_numpy(act_norm),              # (horizon,   act_dim)
        }
