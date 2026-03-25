from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import os
import numpy as np
import xml.etree.ElementTree as ET


# -----------------------------
# basic configs
# -----------------------------

JOINT_TYPE_TO_ONEHOT: Dict[str, List[float]] = {
    "revolute":   [1.0, 0.0, 0.0, 0.0],
    "prismatic":  [0.0, 1.0, 0.0, 0.0],
    "continuous": [0.0, 0.0, 1.0, 0.0],
    "fixed":      [0.0, 0.0, 0.0, 1.0],
}

DEFAULT_EXCLUDE_JOINT_KEYWORDS = [
    "finger",
    "gripper",
    "jaw",
    "mimic",
    "cam",
    "camera",
]

# 每个 joint 的 handcrafted feature 维度
# 4: type onehot
# 3: axis
# 4: lower, upper, range, center
# 3: origin xyz
# 3: origin rpy
JOINT_FEAT_DIM = 17

# tool/eef 固定偏移特征维度
# 3: xyz
# 3: rpy
TOOL_OFFSET_DIM = 6

# 全局统计维度
GLOBAL_FEAT_DIM = 4


# -----------------------------
# dataclasses
# -----------------------------

@dataclass
class JointInfo:
    name: str
    joint_type: str
    axis: List[float]
    lower: float
    upper: float
    origin_xyz: List[float]
    origin_rpy: List[float]
    parent: str
    child: str


@dataclass
class RobotSpec:
    """
    每台机器人一份轻量配置。
    """
    robot_name: str
    urdf_path: str
    arm_joint_names: Optional[List[str]] = None
    include_fixed: bool = False
    exclude_joint_keywords: Optional[List[str]] = None

    # 如果你的 eef pose 定义在某个固定 tool frame 上，
    # 而不是最后一个可动 joint 的 child link 上，
    # 可以把这个固定偏移加进 feature。
    # 例如某些机器人数据里的 eef 在 gripper center。
    include_tool_offset: bool = False
    tool_origin_xyz: Optional[List[float]] = None
    tool_origin_rpy: Optional[List[float]] = None


# -----------------------------
# helpers
# -----------------------------

def _parse_float_list(
    text: Optional[str],
    expected_len: int,
    default: float = 0.0,
) -> List[float]:
    if text is None:
        return [default] * expected_len
    vals = [float(x) for x in text.strip().split()]
    if len(vals) != expected_len:
        raise ValueError(
            f"Expected {expected_len} floats, got {len(vals)} from text={text}"
        )
    return vals


def _normalize_axis(axis: List[float]) -> List[float]:
    a = np.asarray(axis, dtype=np.float32)
    norm = np.linalg.norm(a)
    if norm < 1e-8:
        return [0.0, 0.0, 0.0]
    a = a / norm
    return a.astype(np.float32).tolist()


def _safe_joint_limits(
    joint_type: str,
    limit_elem: Optional[ET.Element],
) -> tuple[float, float]:
    if joint_type == "fixed":
        return 0.0, 0.0

    if limit_elem is not None:
        lower = float(limit_elem.attrib.get("lower", -np.pi))
        upper = float(limit_elem.attrib.get("upper",  np.pi))
        return lower, upper

    if joint_type == "continuous":
        return -np.pi, np.pi

    return -np.pi, np.pi


def _should_exclude_joint(
    joint_name: str,
    exclude_joint_keywords: Optional[List[str]],
) -> bool:
    if not exclude_joint_keywords:
        return False
    lname = joint_name.lower()
    return any(k.lower() in lname for k in exclude_joint_keywords)


# -----------------------------
# urdf parsing
# -----------------------------

def parse_urdf_joints(
    urdf_path: str,
    include_fixed: bool = False,
    exclude_joint_keywords: Optional[List[str]] = None,
) -> List[JointInfo]:
    """
    解析 URDF 里所有 joint，并支持基于名字关键词过滤。
    """
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if not os.path.isfile(urdf_path):
        raise ValueError(f"Path is not a file: {urdf_path}")

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    if root.tag != "robot":
        raise ValueError(f"Root tag is not <robot>, got: {root.tag}")

    exclude_joint_keywords = exclude_joint_keywords or []
    joints: List[JointInfo] = []

    for joint_elem in root.findall("joint"):
        name = joint_elem.attrib["name"]
        joint_type = joint_elem.attrib.get("type", "fixed")

        if _should_exclude_joint(name, exclude_joint_keywords):
            continue

        if (not include_fixed) and joint_type == "fixed":
            continue

        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        origin_elem = joint_elem.find("origin")
        axis_elem = joint_elem.find("axis")
        limit_elem = joint_elem.find("limit")

        parent = parent_elem.attrib["link"] if parent_elem is not None else ""
        child = child_elem.attrib["link"] if child_elem is not None else ""

        if origin_elem is not None:
            origin_xyz = _parse_float_list(origin_elem.attrib.get("xyz"), 3, default=0.0)
            origin_rpy = _parse_float_list(origin_elem.attrib.get("rpy"), 3, default=0.0)
        else:
            origin_xyz = [0.0, 0.0, 0.0]
            origin_rpy = [0.0, 0.0, 0.0]

        if axis_elem is not None:
            axis = _parse_float_list(axis_elem.attrib.get("xyz"), 3, default=0.0)
        else:
            axis = [0.0, 0.0, 0.0]

        axis = _normalize_axis(axis)
        lower, upper = _safe_joint_limits(joint_type, limit_elem)

        joints.append(
            JointInfo(
                name=name,
                joint_type=joint_type,
                axis=axis,
                lower=lower,
                upper=upper,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                parent=parent,
                child=child,
            )
        )

    return joints


# -----------------------------
# chain selection
# -----------------------------

def select_arm_joints(
    joints: List[JointInfo],
    arm_joint_names: Optional[List[str]] = None,
) -> List[JointInfo]:
    """
    如果给了 arm_joint_names，就严格按这个顺序选主链。
    否则默认返回原 joints 顺序。
    """
    if arm_joint_names is None:
        return joints

    joint_map = {j.name: j for j in joints}
    selected: List[JointInfo] = []

    for name in arm_joint_names:
        if name not in joint_map:
            raise KeyError(
                f"Joint '{name}' not found in parsed URDF joints. "
                f"Available joints: {[j.name for j in joints]}"
            )
        selected.append(joint_map[name])

    return selected


# -----------------------------
# feature building
# -----------------------------

def encode_joint_feature(joint: JointInfo) -> np.ndarray:
    """
    单个 joint -> 17 dims
    """
    joint_type = np.asarray(
        JOINT_TYPE_TO_ONEHOT.get(joint.joint_type, [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float32
    )

    axis = np.asarray(joint.axis, dtype=np.float32)

    lower = float(joint.lower)
    upper = float(joint.upper)
    joint_range = upper - lower
    joint_center = 0.5 * (upper + lower)

    origin_xyz = np.asarray(joint.origin_xyz, dtype=np.float32)
    origin_rpy = np.asarray(joint.origin_rpy, dtype=np.float32)

    feat = np.concatenate([
        joint_type,  # 4
        axis,        # 3
        np.asarray([lower, upper, joint_range, joint_center], dtype=np.float32),  # 4
        origin_xyz,  # 3
        origin_rpy,  # 3
    ], axis=0)

    assert feat.shape[0] == JOINT_FEAT_DIM
    return feat.astype(np.float32)


def build_tool_offset_feature(
    tool_origin_xyz: Optional[List[float]],
    tool_origin_rpy: Optional[List[float]],
) -> np.ndarray:
    xyz = np.asarray(tool_origin_xyz if tool_origin_xyz is not None else [0.0, 0.0, 0.0], dtype=np.float32)
    rpy = np.asarray(tool_origin_rpy if tool_origin_rpy is not None else [0.0, 0.0, 0.0], dtype=np.float32)
    feat = np.concatenate([xyz, rpy], axis=0)
    assert feat.shape[0] == TOOL_OFFSET_DIM
    return feat.astype(np.float32)


def extract_robot_feature_from_joints(
    joints: List[JointInfo],
    max_joints: int = 16,
    include_tool_offset: bool = False,
    tool_origin_xyz: Optional[List[float]] = None,
    tool_origin_rpy: Optional[List[float]] = None,
) -> np.ndarray:
    """
    将主链 joints 编码成固定长度 robot_feature。

    输出维度:
        max_joints * 17 + 4 + (6 if include_tool_offset else 0)

    其中:
        - 4 维全局特征:
            dof
            total_link_length
            max_limit_range
            mean_limit_range
        - 6 维可选 tool offset:
            xyz + rpy
    """
    if len(joints) > max_joints:
        raise ValueError(
            f"Robot has {len(joints)} selected arm joints, exceeds max_joints={max_joints}. "
            f"Please increase max_joints."
        )

    joint_feats = [encode_joint_feature(j) for j in joints]
    dof = len(joint_feats)

    padded_joint_feats = np.zeros((max_joints, JOINT_FEAT_DIM), dtype=np.float32)
    for i, feat in enumerate(joint_feats):
        padded_joint_feats[i] = feat

    link_lengths = [
        float(np.linalg.norm(np.asarray(j.origin_xyz, dtype=np.float32)))
        for j in joints
    ]
    limit_ranges = [float(j.upper - j.lower) for j in joints] if len(joints) > 0 else [0.0]

    global_feat = np.asarray([
        float(dof),
        float(np.sum(link_lengths)) if len(link_lengths) > 0 else 0.0,
        float(np.max(limit_ranges)) if len(limit_ranges) > 0 else 0.0,
        float(np.mean(limit_ranges)) if len(limit_ranges) > 0 else 0.0,
    ], dtype=np.float32)

    parts = [
        padded_joint_feats.reshape(-1),
        global_feat,
    ]

    if include_tool_offset:
        tool_feat = build_tool_offset_feature(tool_origin_xyz, tool_origin_rpy)
        parts.append(tool_feat)

    robot_feat = np.concatenate(parts, axis=0)
    return robot_feat.astype(np.float32)


def extract_robot_feature_from_spec(
    spec: RobotSpec,
    max_joints: int = 16,
) -> np.ndarray:
    """
    推荐主入口：按 RobotSpec 提取 robot_feature
    """
    exclude_joint_keywords = (
        spec.exclude_joint_keywords
        if spec.exclude_joint_keywords is not None
        else DEFAULT_EXCLUDE_JOINT_KEYWORDS
    )

    joints = parse_urdf_joints(
        urdf_path=spec.urdf_path,
        include_fixed=spec.include_fixed,
        exclude_joint_keywords=exclude_joint_keywords,
    )

    arm_joints = select_arm_joints(
        joints=joints,
        arm_joint_names=spec.arm_joint_names,
    )

    feat = extract_robot_feature_from_joints(
        joints=arm_joints,
        max_joints=max_joints,
        include_tool_offset=spec.include_tool_offset,
        tool_origin_xyz=spec.tool_origin_xyz,
        tool_origin_rpy=spec.tool_origin_rpy,
    )
    return feat


# -----------------------------
# registry / batch utilities
# -----------------------------

def build_robot_feature_map(
    robot_specs: Dict[str, RobotSpec],
    max_joints: int = 16,
) -> Dict[str, np.ndarray]:
    """
    一次性构建所有机器人的 robot_feature_map
    """
    feature_map: Dict[str, np.ndarray] = {}
    for robot_name, spec in robot_specs.items():
        feat = extract_robot_feature_from_spec(spec, max_joints=max_joints)
        feature_map[robot_name] = feat
    return feature_map


def debug_print_selected_joints(spec: RobotSpec) -> None:
    exclude_joint_keywords = (
        spec.exclude_joint_keywords
        if spec.exclude_joint_keywords is not None
        else DEFAULT_EXCLUDE_JOINT_KEYWORDS
    )

    joints = parse_urdf_joints(
        urdf_path=spec.urdf_path,
        include_fixed=spec.include_fixed,
        exclude_joint_keywords=exclude_joint_keywords,
    )
    arm_joints = select_arm_joints(joints, spec.arm_joint_names)

    print(f"[DEBUG] robot_name={spec.robot_name}")
    print(f"[DEBUG] selected arm joints ({len(arm_joints)}):")
    for j in arm_joints:
        print(f"  - {j.name} ({j.joint_type})")


# -----------------------------
# example usage
# -----------------------------

if __name__ == "__main__":
    # 这里放一个示例，你自己改路径即可
    example_spec = RobotSpec(
        robot_name="airbot_right_arm",
        urdf_path="/path/to/play_g2_usb_cam.urdf",
        arm_joint_names=[
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ],
        include_fixed=False,
        exclude_joint_keywords=[
            "finger",
            "gripper",
            "jaw",
            "mimic",
            "cam",
            "camera",
            "g2_",
        ],
        include_tool_offset=False,
    )

    debug_print_selected_joints(example_spec)
    feat = extract_robot_feature_from_spec(example_spec, max_joints=16)
    print("feature shape:", feat.shape)
    print("feature dtype:", feat.dtype)
    print("first 40 dims:", feat[:40])