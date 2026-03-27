import os
import json
import time
import queue
import threading
import argparse
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
except ImportError as e:
    cprint(f"[!] NatureIK 核心库未安装: {e}", "red")

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
                    file_path = os.path.join(self.log_dir, f"ik_log_{datetime.now().strftime('%Y%m%d')}.parquet")
                    df = pd.DataFrame(batch)
                    if os.path.exists(file_path):
                        try:
                            old_df = pd.read_parquet(file_path)
                            df = pd.concat([old_df, df], ignore_index=True)
                        except: pass
                    df.to_parquet(file_path, index=False)
            except queue.Empty:
                continue

# ==========================================
# 2. 求解器策略接口与实现
# ==========================================
class BaseIKSolver(ABC):
    @abstractmethod
    def solve_arm(self, q_curr: np.ndarray, ee_curr: np.ndarray, ee_target: np.ndarray) -> np.ndarray:
        pass

def _align_quaternions(p: np.ndarray):
    """四元数对齐逻辑，w 在索引 6"""
    p = p.reshape(1, -1)
    mask = p[..., 6] < 0
    p[mask, 3:7] = -p[mask, 3:7]
    return p[0]

class NatureIKSolver(BaseIKSolver):
    def __init__(self, ckpt, robot_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(open(ckpt, "rb"), pickle_module=__import__('dill'))
        self.cfg = payload["cfg"]
        cls = __import__('hydra').utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg); workspace.load_payload(payload)
        self.policy = workspace.ema_model if self.cfg.training.use_ema else workspace.model
        self.policy.to(self.device).eval()
        self.robot_feature = None
        if self.cfg.task.dataset.get('use_robot_feature', False):
            f_map = build_robot_feature_map(ROBOT_SPECS, max_joints=16)
            self.robot_feature = f_map[robot_name].astype(np.float32)
        cprint(f"[*] NatureIK (ResNet) 加载成功", "magenta")

    def solve_arm(self, q_curr, ee_curr, ee_target):
        ec = _align_quaternions(ee_curr)[:7]
        et = _align_quaternions(ee_target)[:7]
        obs = np.concatenate([q_curr[:6], ec, et], axis=-1)
        if self.robot_feature is not None:
            obs = np.concatenate([obs, self.robot_feature], axis=-1)
        obs_ts = torch.from_numpy(obs).float().to(self.device).view(1, 1, -1)
        with torch.no_grad():
            res = self.policy.predict_action({"obs": obs_ts})
            return res['action_pred'].cpu().numpy()[0, 0]

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
            if np.linalg.norm(err) < 1e-4: break
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
            if info[2] != p.JOINT_FIXED: self.pb_movable_joints.append(i)
            l_name = info[12].decode("utf-8").lower()
            if any(x in l_name for x in ["end", "eef", "gripper"]): self.pb_ee_index = i
        if self.pb_ee_index == -1: self.pb_ee_index = self.pb_movable_joints[5]

    def solve_arm(self, q_curr, ee_curr, ee_target):
        for i in range(min(len(self.pb_movable_joints), 6)):
            p.resetJointState(self.pb_robot_id, self.pb_movable_joints[i], q_curr[i])
        ik_sol = p.calculateInverseKinematics(
            self.pb_robot_id, self.pb_ee_index, ee_target[:3], ee_target[3:7],
            maxNumIterations=100, residualThreshold=1e-5
        )
        return np.array(ik_sol[:6]) - q_curr[:6]

class CuRoboSolver(BaseIKSolver):
    def __init__(self, urdf): cprint("[*] cuRobo 模式已选择，请确保初始化逻辑完整", "yellow")
    def solve_arm(self, q_curr, ee_curr, ee_target): return np.zeros(6)

# ==========================================
# 3. FastAPI 核心接口 (对齐机器人端)
# ==========================================
app = FastAPI()

@app.post("/predict")
async def predict_endpoint(
    background_tasks: BackgroundTasks,
    action_eef: str = Form(...),   # Pi0 输出 (16D)
    joint_now: str = Form(...),    # 机器人当前关节 (14D)
    eef_now: str = Form(...)       # 机器人当前末端 (14D)
):
    try:
        start_t = time.perf_counter()
        # 解析数据
        q_all = np.array(json.loads(joint_now), dtype=np.float32)
        ee_all = np.array(json.loads(eef_now), dtype=np.float32)
        pi_act = np.array(json.loads(action_eef), dtype=np.float32)
        
        ik: BaseIKSolver = app.state.solver
        
        # 双臂拆解: Left [0:7], Right [7:14] (EEF) | Left [0:8], Right [8:16] (Pi0 Action)
        q_l, q_r = q_all[0:7], q_all[7:14]
        ee_l, ee_r = ee_all[0:7], ee_all[7:14]
        
        # Pi0: [L_ee(7), L_g(1), R_ee(7), R_g(1)]
        et_l, g_l = pi_act[0:7], pi_act[7]
        et_r, g_r = pi_act[8:15], pi_act[15]

        # 核心求解 (Delta Q)
        dq_l = ik.solve_arm(q_l, ee_l, et_l)
        dq_r = ik.solve_arm(q_r, ee_r, et_r)

        # 积分叠加 (q_next = q_curr + delta_q)
        q_next_l = q_l.copy(); q_next_l[:6] += dq_l[:6]
        q_next_r = q_r.copy(); q_next_r[:6] += dq_r[:6]

        # 缝合 16 维全量信号: [L_q(7), L_g(1), R_q(7), R_g(1)]
        full_action = np.zeros(16)
        full_action[0:7], full_action[7] = q_next_l, g_l
        full_action[8:15], full_action[15] = q_next_r, g_r

        # 异步实验日志
        latency = (time.perf_counter() - start_t) * 1000
        app.state.logger.log({
            "timestamp": datetime.now().isoformat(),
            "solver": ik.__class__.__name__,
            "latency_ms": latency,
            "q_now": q_all.tolist(),
            "ee_now": ee_all.tolist(),
            "ee_target": pi_act[[0,1,2,3,4,5,6, 8,9,10,11,12,13,14]].tolist(),
            "actions": full_action.tolist()
        })

        # 返回机器人端期望的 "actions" 字段
        return {"success": True, "actions": full_action.tolist()}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/health")
def health(): return {"status": "ok", "solver": app.state.solver.__class__.__name__}

# ==========================================
# 4. 启动入口
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/2026-03-26/00-46-32/checkpoints/epoch=0190-val_loss=0.006763.ckpt")
    parser.add_argument("--urdf", default="diffusion_policy/urdf_data/play_g2_usb_cam/urdf/play_g2_usb_cam.urdf")
    parser.add_argument("--robot", default="airbot_single_arm")
    args = parser.parse_args()

    if not os.path.exists(args.urdf):
        cprint(f"[!] URDF 不存在: {args.urdf}", "red"); return

    cprint("\n--- AIRBOT IK Expert Server Launcher ---", "green", attrs=["bold"])
    print(" [1] NatureIK (Diffusion-ResNet)\n [2] Pinocchio (Numerical)\n [3] cuRobo (GPU)\n [4] PyBullet (DLS)")
    choice = input("\n请选择求解器编号: ").strip()

    try:
        if choice == "1": app.state.solver = NatureIKSolver(args.ckpt, args.robot)
        elif choice == "2": app.state.solver = PinocchioIKSolver(args.urdf)
        elif choice == "3": app.state.solver = CuRoboSolver(args.urdf)
        elif choice == "4": app.state.solver = PyBulletIKSolver(args.urdf)
        else: return
    except Exception as e:
        cprint(f"[!] 初始化失败: {e}", "red"); return

    app.state.logger = ExperimentLogger()
    cprint(f"🚀 服务已启动 [Port 6162] | 模式: {app.state.solver.__class__.__name__}", "green")
    uvicorn.run(app, host="0.0.0.0", port=6162, log_level="warning")

if __name__ == "__main__":
    main()