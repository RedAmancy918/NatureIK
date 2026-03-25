"""
LeRobot 风格数据集路径：根目录下多个 chunk 子目录，每个内含 episode_*.parquet。

  <data_dir>/
    chunk-000/episode_000000.parquet ...
    chunk-001/...
"""
from __future__ import annotations

import glob
import os
from typing import List


def collect_episode_parquet_paths(
    data_dir: str,
    episode_glob: str,
    *,
    scan_chunk_subdirs: bool = False,
    chunk_dir_glob: str = "chunk-*",
) -> List[str]:
    """
    Args:
        data_dir: 单 chunk 目录（如 .../chunk-000），或 LeRobot 的 data 根目录（其下含 chunk-*）。
        episode_glob: 如 episode_*.parquet
        scan_chunk_subdirs: True 时在 data_dir/chunk-*/ 下搜索 episode。
        chunk_dir_glob: chunk 子目录名，默认 chunk-*。
    """
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    if not scan_chunk_subdirs:
        pattern = os.path.join(data_dir, episode_glob)
    else:
        pattern = os.path.join(data_dir, chunk_dir_glob, episode_glob)
    paths = sorted(glob.glob(pattern))
    return paths
