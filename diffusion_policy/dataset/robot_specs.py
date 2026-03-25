# specs for all robots to configure the robot_feature_utils.py.
from diffusion_policy.dataset.robot_feature_utils import RobotSpec
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_URDF_DIR = (_THIS_DIR / "../urdf_data").resolve()

ROBOT_SPECS = {
    "panda": RobotSpec(
        robot_name="panda",
        urdf_path=_URDF_DIR / "panda.urdf",
        arm_joint_names=[
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ],
        include_fixed=False,
        exclude_joint_keywords=["finger", "gripper", "jaw", "mimic", "hand"],
        include_tool_offset=False,
    ),
    "airbot_single_arm": RobotSpec(
        robot_name="airbot_single_arm",
        urdf_path=_URDF_DIR / "play_g2_usb_cam.urdf",
        arm_joint_names=[
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ],
        include_fixed=False,
        exclude_joint_keywords=["finger", "gripper", "jaw", "mimic", "cam", "camera", "g2_"],
        include_tool_offset=False,
    ),
    "piper_single_arm": RobotSpec(
        robot_name="piper_single_arm",
        urdf_path=_URDF_DIR / "piper.urdf",
        # piper.urdf: joint1–joint6 为臂部旋转关节；joint7/joint8 为夹爪 prismatic；相机为 fixed
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
            "mount",
        ],
        include_tool_offset=False,
    ),
}