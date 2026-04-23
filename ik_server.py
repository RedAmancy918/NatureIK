import os
import json
import time
import queue
import threading
import argparse
from collections import deque
import numpy as np
import pandas as pd
import torch
import uvicorn
from abc import ABC, abstractmethod
from datetime import datetime
from fastapi import FastAPI, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from termcolor import cprint

# --- 核心依赖加载 (含防御性导入) ---
try:
    import pybullet as p
    import pybullet_data
except ImportError:
    cprint("[!] 缺失 PyBullet，模式 4 将不可用", "yellow")

try:
    import pinocchio as pin
except ImportError:
    cprint("[!] 缺失 Pinocchio，模式 2 将不可用", "yellow")

try:
    from curobo.wrap.replan.arm_replan import ArmReplanner

    HAS_CUROBO = True
except ImportError:
    HAS_CUROBO = False

try:
    from diffusion_policy.workspace.base_workspace import BaseWorkspace
    from diffusion_policy.dataset.robot_feature_utils import build_robot_feature_map
    from diffusion_policy.dataset.robot_specs import ROBOT_SPECS
    from diffusion_policy.model.differentiable_ik_model import DifferentiableIKHelper, TTGLoss
    from diffusion_policy.gym_util.guidance_utils_ik import guided_inference
except ImportError as e:
    cprint(f"[!] NatureIK 核心库未安装: {e}", "red")

try:
    import sys
    # MIP repo path — adjust if installed elsewhere
    _MIP_ROOT = os.path.expanduser("../much-ado-about-noising")
    if _MIP_ROOT not in sys.path:
        sys.path.insert(0, _MIP_ROOT)
    from mip.agent import TrainingAgent
    from omegaconf import OmegaConf
    HAS_MIP = True
except ImportError as e:
    HAS_MIP = False
    cprint(f"[!] MIP 库未找到，模式 5 将不可用: {e}", "yellow")

# ==========================================
# 1. 异步实验日志记录器 (守护线程版)
# ==========================================
class ExperimentLogger:
    def __init__(self, log_dir="./experiment_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        cprint(f"[*] 实验日志系统已启动: {log_dir}", "cyan")

    def log(self, data: dict):
        self.queue.put(data)

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                batch = []
                data = self.queue.get(timeout=1.0)
                batch.append(data)
                while not self.queue.empty() and len(batch) < 100:
                    batch.append(self.queue.get_nowait())
                if batch:
                    file_path = os.path.join(
                        self.log_dir,
                        f"ik_log_{datetime.now().strftime('%Y%m%d')}.parquet",
                    )
                    df = pd.DataFrame(batch)
                    if os.path.exists(file_path):
                        try:
                            old_df = pd.read_parquet(file_path)
                            df = pd.concat([old_df, df], ignore_index=True)
                        except:
                            pass
                    df.to_parquet(file_path, index=False)
            except queue.Empty:
                continue


# ==========================================
# 2. 求解器策略接口与实现
# ==========================================
class BaseIKSolver(ABC):
    @abstractmethod
    def solve_arm(
        self, q_curr: np.ndarray, ee_curr: np.ndarray, ee_target: np.ndarray
    ) -> np.ndarray:
        pass
    #新增双臂推理接口
    def solve_dual_arm(self, q_l, ee_l, et_l, q_r, ee_r, et_r):
        return self.solve_arm(q_l, ee_l, et_l), self.solve_arm(q_r, ee_r, et_r)


# 四元数对齐逻辑：输入 7D eef 向量 [px, py, pz, qx, qy, qz, qw]，w 在索引 6
# 若 w < 0 则整体取反，保持单位四元数半球一致性
def _align_quaternions(p: np.ndarray):
    p = p.reshape(1, -1)
    mask = p[..., 6] < 0  # w < 0 → 翻转到上半球
    p[mask, 3:7] = -p[mask, 3:7]  # 翻转四元数部分 [qx, qy, qz, qw]
    return p[0]

class MIPIKSolver(BaseIKSolver):
    """基于 MIP（流匹配）模型的 IK 求解器。

    支持可选的测试时梯度投影（TTG Projection）：
        MIP 2步确定性推理 → delta_q → FK梯度投影（L_pose + L_hist + L_smooth）→ delta_q_refined

    从 train_ik.py 保存的推理包中加载：
        flow_map / encoder / *_ema 权重
        config  （OmegaConf 容器 → 重建 TrainingAgent）
        normalizer  （obs + action 归一化器，由训练数据拟合）

    观测历史：
        模型期望输入 obs_steps 帧的历史观测。由于服务端每次调用只提供当前帧，
        内部维护一个滚动 buffer；首次调用时用第一帧重复填充 buffer（冷启动）。
    """

    def __init__(
        self,
        ckpt: str,
        urdf_path: str | None = None,          # 提供时启用 TTG 梯度投影
        ee_link: str = "link_eef",
        use_ttg: bool = False,                  # 是否启用梯度投影
        ttg_lr: float = 0.02,                   # 投影步长
        ttg_steps: int = 5,                     # 投影迭代次数
        ttg_lambda_pose: float = 1.0,
        ttg_lambda_hist: float = 0.3,
        ttg_lambda_smooth: float = 0.05,
    ):
        if not HAS_MIP:
            raise RuntimeError("MIP library not available. Check sys.path.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── 加载推理包 ────────────────────────────────────────────────────────
        bundle = torch.load(ckpt, map_location="cpu", weights_only=False)

        # ── 重建 config 与 TrainingAgent ──────────────────────────────────────
        cfg_dict = bundle["config"]
        config = OmegaConf.create(cfg_dict)
        # 单样本推理不需要 compile / CUDA-graphs，关闭以避免额外开销
        config.optimization.device = str(self.device)
        config.optimization.use_compile = False
        config.optimization.use_cudagraphs = False
        config.optimization.compile_mode = None

        self.agent = TrainingAgent(config)
        # 分别加载 flow_map 与 encoder 的主参数及 EMA 参数
        self.agent.flow_map.load_state_dict(bundle["flow_map"])
        self.agent.encoder.load_state_dict(bundle["encoder"])
        self.agent.flow_map_ema.load_state_dict(bundle["flow_map_ema"])
        self.agent.encoder_ema.load_state_dict(bundle["encoder_ema"])
        self.agent.eval()  # 切换到推理模式，关闭 dropout 等

        # ── 归一化器 ──────────────────────────────────────────────────────────
        # obs_state: CompositeNormalizer（关节 MinMax + 四元数 Identity）
        # action:    MinMaxNormalizer（delta_joint 整体缩放到 [-1, 1]）
        # bundle["normalizer"] 由 convert_checkpoint.py / save_mip_bundle 写入：
        #   {"obs_state": CompositeNormalizer, "action": MinMaxNormalizer}
        self._norm_obs    = bundle["normalizer"]["obs_state"]
        self._norm_action = bundle["normalizer"]["action"]

        # ── 从 config 读取关键维度 ────────────────────────────────────────────
        self.obs_steps = int(config.task.obs_steps)  # encoder 接收的历史帧数
        self.horizon   = int(config.task.horizon)    # 动作预测窗口长度
        self.act_dim   = int(config.task.act_dim)    # 动作维度（delta_joint = 6）

        # 左右臂独立历史 buffer：各自保留最近 obs_steps 帧，互不干扰
        self._history_l: deque = deque(maxlen=self.obs_steps)
        self._history_r: deque = deque(maxlen=self.obs_steps)

        # 上一步 delta_q，用于 L_hist（左右臂独立）
        self._dq_prev_l: np.ndarray | None = None
        self._dq_prev_r: np.ndarray | None = None

        # ── TTG 梯度投影初始化 ────────────────────────────────────────────────
        self.use_ttg  = use_ttg and (urdf_path is not None)
        self.ttg_lr   = ttg_lr
        self.ttg_steps = ttg_steps
        self.ik_helper = None
        self.ttg_loss  = None
        if self.use_ttg:
            self.ik_helper = DifferentiableIKHelper(
                urdf_path=urdf_path,
                end_effector_link_name=ee_link,
                device=str(self.device),
            )
            self.ttg_loss = TTGLoss(
                lambda_pose=ttg_lambda_pose,
                lambda_hist=ttg_lambda_hist,
                lambda_smooth=ttg_lambda_smooth,
            )
            cprint(f"[*] MIP TTG 梯度投影已启用 (lr={ttg_lr}, steps={ttg_steps})", "cyan")

        cprint(
            f"[*] MIP IK Solver loaded | obs_steps={self.obs_steps} "
            f"horizon={self.horizon} act_dim={self.act_dim} device={self.device}",
            "magenta",
        )

    def _make_obs_frame(
        self,
        q_curr: np.ndarray,
        ee_curr: np.ndarray,
        ee_target: np.ndarray,
    ) -> np.ndarray:
        """拼接单帧 obs 向量，格式与训练时完全一致。

        输出维度：20D
        布局：[q_curr(6) | eef_curr_xyz(3) | eef_curr_quat(4) | eef_tgt_xyz(3) | eef_tgt_quat(4)]
        四元数均经过半球对齐（w < 0 时取反），避免符号不一致带来的跳变。
        """
        ec = _align_quaternions(ee_curr)[:7]    # 当前末端：xyz + quat（已对齐）
        et = _align_quaternions(ee_target)[:7]  # 目标末端：xyz + quat（已对齐）
        return np.concatenate([q_curr[:6], ec, et], axis=-1).astype(np.float32)

    def _infer_with_buf(
        self,
        buf: deque,
        frame: np.ndarray,
        q_curr: np.ndarray,
        ee_target: np.ndarray,
        dq_prev: np.ndarray | None = None,
    ) -> np.ndarray:
        """MIP 推理 + 可选 TTG 梯度投影，返回 delta_q (6,)。"""
        # ── 历史 buffer 更新 ──────────────────────────────────────────────────
        if len(buf) == 0:
            for _ in range(self.obs_steps):
                buf.append(frame.copy())
        else:
            buf.append(frame)

        obs_np   = np.stack(list(buf), axis=0)
        obs_norm = self._norm_obs.normalize(obs_np)
        obs_t    = torch.from_numpy(obs_norm).float().unsqueeze(0).to(self.device)

        # ── MIP 2步确定性推理 ─────────────────────────────────────────────────
        act_0 = torch.zeros(1, self.horizon, self.act_dim, device=self.device)
        with torch.no_grad():
            act_norm_t = self.agent.sample(act_0=act_0, obs=obs_t, num_steps=2, use_ema=True)

        act_norm_np = act_norm_t.cpu().numpy()[0, 0]
        delta_q     = self._norm_action.unnormalize(act_norm_np[np.newaxis])[0]  # (6,)

        # ── TTG 梯度投影（可选）──────────────────────────────────────────────
        if self.use_ttg and self.ik_helper is not None:
            delta_q = self._ttg_projection(
                delta_q    = delta_q,
                q_curr     = q_curr,
                ee_target  = ee_target,
                dq_prev    = dq_prev,
            )

        return delta_q

    def _ttg_projection(
        self,
        delta_q: np.ndarray,           # MIP 输出的 delta_q (6,)
        q_curr: np.ndarray,            # 当前关节角 (6,)
        ee_target: np.ndarray,         # 目标末端位姿 (7,) xyz+quat
        dq_prev: np.ndarray | None,    # 上一步 delta_q，用于 L_hist
    ) -> np.ndarray:
        """在 MIP 输出上做 FK 梯度投影，修正几何误差，保留风格。

        优化变量是 delta_q（而非关节角本身），保证修正量最小。
        """
        device = self.device
        dq = torch.from_numpy(delta_q).float().to(device).unsqueeze(0)   # (1, 6)
        q  = torch.from_numpy(q_curr[:6]).float().to(device).unsqueeze(0) # (1, 6)
        target_pos = torch.from_numpy(ee_target[:3]).float().to(device).unsqueeze(0)  # (1, 3)

        dq_prev_t = None
        if dq_prev is not None:
            dq_prev_t = torch.from_numpy(dq_prev).float().to(device).unsqueeze(0)

        dq = dq.requires_grad_(True)
        for _ in range(self.ttg_steps):
            q_pred = q + dq                                               # (1, 6)
            curr_pos, curr_rot = self.ik_helper.get_current_pose(q_pred)

            loss = self.ttg_loss.compute(
                curr_pos   = curr_pos,
                curr_rot   = curr_rot,
                target_pos = target_pos,
                dq_pred    = dq,
                dq_prev    = dq_prev_t,
            )
            if loss.item() < 1e-6:
                break
            grad = torch.autograd.grad(loss, dq)[0]
            # 梯度裁剪：防止奇异点附近梯度爆炸导致 dq 被推离正常范围
            grad_norm = grad.norm()
            if grad_norm > 1.0:
                grad = grad / grad_norm
            dq   = (dq - self.ttg_lr * grad).detach().requires_grad_(True)

        return dq.detach().squeeze(0).cpu().numpy()

    def solve_arm(
        self,
        q_curr: np.ndarray,
        ee_curr: np.ndarray,
        ee_target: np.ndarray,
    ) -> np.ndarray:
        """单臂推理：使用左臂 buffer。"""
        frame  = self._make_obs_frame(q_curr, ee_curr, ee_target)
        dq     = self._infer_with_buf(
            self._history_l, frame, q_curr, ee_target, self._dq_prev_l
        )
        self._dq_prev_l = dq.copy()
        return dq

    def solve_dual_arm(
        self,
        q_l: np.ndarray, ee_l: np.ndarray, et_l: np.ndarray,
        q_r: np.ndarray, ee_r: np.ndarray, et_r: np.ndarray,
    ):
        """双臂推理：左右臂各用独立 buffer 和独立 dq_prev。"""
        frame_l = self._make_obs_frame(q_l, ee_l, et_l)
        frame_r = self._make_obs_frame(q_r, ee_r, et_r)
        dq_l = self._infer_with_buf(self._history_l, frame_l, q_l, et_l, self._dq_prev_l)
        dq_r = self._infer_with_buf(self._history_r, frame_r, q_r, et_r, self._dq_prev_r)
        self._dq_prev_l = dq_l.copy()
        self._dq_prev_r = dq_r.copy()
        return dq_l, dq_r

class NatureIKSolver(BaseIKSolver):
    def __init__(
        self,
        ckpt,
        robot_name,
        urdf_path: str | None = None,         # 提供时启用 TTG
        ee_link: str = "link_eef",
        use_ttg: bool = False,                 # 是否启用 Diffusion TTG
        guidance_scale: float = 10.0,
        ttg_lambda_pose: float = 1.0,
        ttg_lambda_hist: float = 0.3,
        ttg_lambda_smooth: float = 0.05,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(open(ckpt, "rb"), pickle_module=__import__("dill"))
        self.cfg = payload["cfg"]
        cls = __import__("hydra").utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg)
        workspace.load_payload(payload)
        self.policy = (
            workspace.ema_model if self.cfg.training.use_ema else workspace.model
        )
        self.policy.to(self.device).eval()
        # TensorRT 级编译优化
        try:
            cprint("[*] 正在为 NatureIK 开启 TensorRT 级编译优化...", "yellow")
            self.policy = torch.compile(self.policy, mode="reduce-overhead")
            dummy_obs = torch.randn(2, self.policy.n_obs_steps, 20).to(self.device)
            with torch.no_grad():
                _ = self.policy.predict_action({"obs": dummy_obs})
            cprint("[*] NatureIK 编译预热完成！", "green")
        except Exception as e:
            cprint(f"[!] 编译失败（版本或环境不支持）: {e}", "red")

        self.robot_feature = None
        if self.cfg.task.dataset.get("use_robot_feature", False):
            f_map = build_robot_feature_map(ROBOT_SPECS, max_joints=16)
            self.robot_feature = f_map[robot_name].astype(np.float32)

        # ── TTG 初始化 ────────────────────────────────────────────────────────
        self.use_ttg       = use_ttg and (urdf_path is not None)
        self.guidance_scale = guidance_scale
        self.ik_helper     = None
        self.ttg_loss      = None
        if self.use_ttg:
            self.ik_helper = DifferentiableIKHelper(
                urdf_path=urdf_path,
                end_effector_link_name=ee_link,
                device=str(self.device),
            )
            self.ttg_loss = TTGLoss(
                lambda_pose=ttg_lambda_pose,
                lambda_hist=ttg_lambda_hist,
                lambda_smooth=ttg_lambda_smooth,
            )
            cprint(f"[*] NatureIK TTG 已启用 (scale={guidance_scale})", "cyan")

        # 左右臂各自维护上一步 delta_q，用于 L_hist
        self._dq_prev_l: np.ndarray | None = None
        self._dq_prev_r: np.ndarray | None = None

        cprint(f"[*] NatureIK (ResNet) 加载成功", "magenta")

    def _build_obs(self, q, ee, et):
        ec  = _align_quaternions(ee)[:7]
        et_ = _align_quaternions(et)[:7]
        obs = np.concatenate([q[:6], ec, et_], axis=-1)
        if self.robot_feature is not None:
            obs = np.concatenate([obs, self.robot_feature], axis=-1)
        return obs

    def _predict(self, obs_ts, ee_target_t, dq_prev_t):
        """统一推理入口：有 TTG 时走 guided_inference，否则走 predict_action。"""
        if self.use_ttg and self.ik_helper is not None:
            target_pos = torch.from_numpy(ee_target_t[:, :3]).float().to(self.device)
            target_pose_dict = {"pos": target_pos}
            action_stats = getattr(self.policy.normalizer, "action_stats", None) \
                           or self.policy.normalizer["action"].stats
            action = guided_inference(
                policy           = self.policy,
                obs_dict         = {"obs": obs_ts},
                ik_helper        = self.ik_helper,
                target_pose_dict = target_pose_dict,
                action_stats     = action_stats,
                guidance_scale   = self.guidance_scale,
                dq_prev          = dq_prev_t,
                ttg_loss         = self.ttg_loss,
                device           = str(self.device),
            )
            return action[:, 0, :].cpu().numpy()   # (B, act_dim)
        else:
            with torch.no_grad():
                res = self.policy.predict_action({"obs": obs_ts})
            actions = res["action_pred"].cpu().numpy()
            if actions.ndim == 3:
                actions = actions[:, 0, :]
            return actions                          # (B, act_dim)

    def solve_arm(self, q_curr, ee_curr, ee_target):
        obs    = self._build_obs(q_curr, ee_curr, ee_target)
        n_obs  = self.policy.n_obs_steps
        obs_ts = torch.from_numpy(obs).float().to(self.device)
        obs_ts = obs_ts.unsqueeze(0).unsqueeze(0).repeat(1, n_obs, 1)   # (1, n_obs, D)

        et_t      = torch.from_numpy(ee_target[:7]).float().unsqueeze(0).to(self.device)  # (1,7)
        dq_prev_t = None
        if self._dq_prev_l is not None:
            dq_prev_t = torch.from_numpy(self._dq_prev_l).float().unsqueeze(0).to(self.device)

        result         = self._predict(obs_ts, et_t.cpu().numpy(), dq_prev_t)
        dq             = result[0]                 # (act_dim,)
        self._dq_prev_l = dq.copy()
        return dq

    def solve_dual_arm(self, q_l, ee_l, et_l, q_r, ee_r, et_r):
        obs_l = self._build_obs(q_l, ee_l, et_l)
        obs_r = self._build_obs(q_r, ee_r, et_r)
        obs_batch = np.stack([obs_l, obs_r], axis=0)                     # (2, D)
        n_obs  = self.policy.n_obs_steps
        obs_ts = torch.from_numpy(obs_batch).float().to(self.device)
        obs_ts = obs_ts.unsqueeze(1).repeat(1, n_obs, 1)                 # (2, n_obs, D)

        et_batch = np.stack([et_l[:7], et_r[:7]], axis=0)                # (2, 7)
        dq_prev_t = None
        if self._dq_prev_l is not None and self._dq_prev_r is not None:
            dq_prev_t = torch.from_numpy(
                np.stack([self._dq_prev_l, self._dq_prev_r], axis=0)
            ).float().to(self.device)

        results = self._predict(obs_ts, et_batch, dq_prev_t)             # (2, act_dim)
        self._dq_prev_l = results[0].copy()
        self._dq_prev_r = results[1].copy()
        return results[0], results[1]

class PinocchioIKSolver(BaseIKSolver):
    def __init__(self, urdf_path, ee_link="link_eef"):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_id = self.model.getJointId(ee_link)

    def solve_arm(self, q_curr, ee_curr, ee_target):
        target_se3 = pin.XYZQUATToSE3(ee_target[:7])
        q = q_curr[:6].copy()
        for _ in range(15):
            pin.forwardKinematics(self.model, self.data, q)
            d_se3 = self.data.oMi[self.ee_id].inverse() * target_se3
            err = pin.log6(d_se3).np
            if np.linalg.norm(err) < 1e-4:
                break
            J = pin.computeJointJacobian(self.model, self.data, q, self.ee_id)
            q = pin.integrate(self.model, q, np.linalg.pinv(J) @ err)
        return q - q_curr[:6]


class PyBulletIKSolver(BaseIKSolver):
    def __init__(self, urdf_path):
        self.pb_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.pb_robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
        self.pb_movable_joints = []
        self.pb_ee_index = -1
        for i in range(p.getNumJoints(self.pb_robot_id)):
            info = p.getJointInfo(self.pb_robot_id, i)
            if info[2] != p.JOINT_FIXED:
                self.pb_movable_joints.append(i)
            l_name = info[12].decode("utf-8").lower()
            if any(x in l_name for x in ["end", "eef", "gripper"]):
                self.pb_ee_index = i
        if self.pb_ee_index == -1:
            self.pb_ee_index = self.pb_movable_joints[5]

    def solve_arm(self, q_curr, ee_curr, ee_target):
        for i in range(min(len(self.pb_movable_joints), 6)):
            p.resetJointState(self.pb_robot_id, self.pb_movable_joints[i], q_curr[i])
        ik_sol = p.calculateInverseKinematics(
            self.pb_robot_id,
            self.pb_ee_index,
            ee_target[:3],
            ee_target[3:7],
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        return np.array(ik_sol[:6]) - q_curr[:6]


class CuRoboSolver(BaseIKSolver):
    def __init__(self, urdf):
        cprint("[*] cuRobo 模式已选择，请确保初始化逻辑完整", "yellow")

    def solve_arm(self, q_curr, ee_curr, ee_target):
        return np.zeros(6)


# ==========================================
# 3. FastAPI 核心接口 (对齐机器人端)
# ==========================================
app = FastAPI()


@app.post("/predict")
async def predict_endpoint(
    background_tasks: BackgroundTasks,
    action_eef: str = Form(...),  # Pi0 输出 (16D)
    joint_now: str = Form(...),  # 机器人当前关节 (14D)
    eef_now: str = Form(...),  # 机器人当前末端 (16D)
):
    try:
        start_t = time.perf_counter()
        # 解析数据
        q_all = np.array(json.loads(joint_now), dtype=np.float32)
        ee_all = np.array(json.loads(eef_now), dtype=np.float32)
        pi_act = np.array(json.loads(action_eef), dtype=np.float32)

        ik: BaseIKSolver = app.state.solver

        # 双臂拆解:
        #   joint_now (14D): Left q[0:7], Right q[7:14]
        #   eef_now   (16D): Left ee[0:7] + Left g[7], Right ee[8:15] + Right g[15]
        #   action_eef(16D): Left ee[0:7] + Left g[7], Right ee[8:15] + Right g[15]
        q_l, q_r = q_all[0:7], q_all[7:14]
        ee_l, ee_r = ee_all[0:7], ee_all[8:15]  # 跳过 [7]=左夹爪，取右臂 ee

        # Pi0: [L_ee(7), L_g(1), R_ee(7), R_g(1)]
        et_l, g_l = pi_act[0:7], pi_act[7]
        et_r, g_r = pi_act[8:15], pi_act[15]

        # 核心求解 (Delta Q) 顺序推理（单臂推理）
        # dq_l = ik.solve_arm(q_l, ee_l, et_l)
        # dq_r = ik.solve_arm(q_r, ee_r, et_r)

        # NatureIKSolver 走 B=2 并行（双臂推理）；Pinocchio/PyBullet 自动兜底顺序调用（单臂推理） 
        # #新增双臂推理接口
        dq_l, dq_r = ik.solve_dual_arm(q_l, ee_l, et_l, q_r, ee_r, et_r)  # (6,) * 2

        # 积分叠加 (q_next = q_curr + delta_q)
        q_next_l = q_l.copy()
        q_next_l[:6] += dq_l[:6]
        q_next_r = q_r.copy()
        q_next_r[:6] += dq_r[:6]

        # 缝合 14 维全量信号: [L_j(6), L_g(1), R_j(6), R_g(1)]
        full_action = np.zeros(14)
        full_action[0:6] = q_next_l[:6]   # 左臂 6 关节
        full_action[6]   = float(g_l)     # 左夹爪
        full_action[7:13] = q_next_r[:6]  # 右臂 6 关节
        full_action[13]  = float(g_r)     # 右夹爪

        # 异步实验日志
        latency = (time.perf_counter() - start_t) * 1000
        app.state.logger.log(
            {
                "timestamp": datetime.now().isoformat(),
                "solver": ik.__class__.__name__,
                "latency_ms": latency,
                "q_now": q_all.tolist(),
                "ee_now": ee_all.tolist(),
                "ee_target": pi_act[
                    [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
                ].tolist(),
                "actions": full_action.tolist(),
            }
        )

        # 返回机器人端期望的 "actions" 字段 (14D: L_j×6 + L_g + R_j×6 + R_g)
        return {"success": True, "actions": full_action.tolist()}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/health")
def health():
    return {"status": "ok", "solver": app.state.solver.__class__.__name__}


# ==========================================
# 4. 启动入口
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="outputs/2026-03-26/00-46-32/checkpoints/epoch=0190-val_loss=0.006763.ckpt",
    )
    parser.add_argument(
        "--mip_ckpt",
        # 加载MIP模型
        default="../much-ado-about-noising/logs/models/model_final_bundle.pt",
        help="MIP checkpoint path (inference bundle saved by train_ik.py)",
    )
    parser.add_argument(
        "--urdf",
        default="diffusion_policy/urdf_data/play_g2_usb_cam/urdf/play_g2_usb_cam.urdf",
    )
    parser.add_argument("--robot",   default="airbot_single_arm")
    parser.add_argument("--ee_link", default="end_link", help="URDF 末端 link 名（默认 end_link）")
    parser.add_argument("--use_ttg", action="store_true", help="启用 Test-Time Guidance")
    parser.add_argument("--guidance_scale",    type=float, default=10.0)
    parser.add_argument("--ttg_lambda_pose",   type=float, default=1.0,  help="TTG L_pose 权重")
    parser.add_argument("--ttg_lambda_hist",   type=float, default=0.3,  help="TTG L_hist 权重（0 关闭）")
    parser.add_argument("--ttg_lambda_smooth", type=float, default=0.05, help="TTG L_smooth 权重（0 关闭）")
    parser.add_argument("--port", type=int, default=6162, help="服务端口（默认 6162）")
    args = parser.parse_args()

    if not os.path.exists(args.urdf):
        cprint(f"[!] URDF 不存在: {args.urdf}", "red")
        return

    cprint("\n--- AIRBOT IK Expert Server Launcher ---", "green", attrs=["bold"])
    print(
        " [1] NatureIK (Diffusion-ResNet)\n [2] Pinocchio (Numerical)\n [3] cuRobo (GPU)\n [4] PyBullet (DLS)\n [5] MIP (Flow-Matching)"
    )
    choice = input("\n请选择求解器编号: ").strip()

    try:
        if choice == "1":
            app.state.solver = NatureIKSolver(
                ckpt              = args.ckpt,
                robot_name        = args.robot,
                urdf_path         = args.urdf if args.use_ttg else None,
                ee_link           = args.ee_link,
                use_ttg           = args.use_ttg,
                guidance_scale    = args.guidance_scale,
                ttg_lambda_pose   = args.ttg_lambda_pose,
                ttg_lambda_hist   = args.ttg_lambda_hist,
                ttg_lambda_smooth = args.ttg_lambda_smooth,
            )
        elif choice == "2":
            app.state.solver = PinocchioIKSolver(args.urdf)
        elif choice == "3":
            app.state.solver = CuRoboSolver(args.urdf)
        elif choice == "4":
            app.state.solver = PyBulletIKSolver(args.urdf)
        elif choice == "5":
            if args.mip_ckpt is None:
                cprint(f"[!] 请通过 --mip_ckpt 指定 MIP checkpoint 路径", "red")
                return
            app.state.solver = MIPIKSolver(
                ckpt              = args.mip_ckpt,
                urdf_path         = args.urdf if args.use_ttg else None,
                ee_link           = args.ee_link,
                use_ttg           = args.use_ttg,
                ttg_lambda_pose   = args.ttg_lambda_pose,
                ttg_lambda_hist   = args.ttg_lambda_hist,
                ttg_lambda_smooth = args.ttg_lambda_smooth,
            )
        else:
            return
    except Exception as e:
        cprint(f"[!] 初始化失败: {e}", "red")
        return

    app.state.logger = ExperimentLogger()
    cprint(
        f"🚀 服务已启动 [Port {args.port}] | 模式: {app.state.solver.__class__.__name__} | TTG: {args.use_ttg}",
        "green",
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
