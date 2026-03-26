import os
import json
import time
import queue
import threading
import argparse
import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data
import torch
import uvicorn
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from termcolor import cprint

# 在 ik_server.py 的顶部添加
try:
    from curobo.wrap.replan.arm_replan import ArmReplanner
    from curobo.types.base import TensorConfig
    from curobo.types.robot import RobotConfig
    HAS_CUROBO = True
except ImportError:
    HAS_CUROBO = False
    # 别哭，这说明你没装好 NVIDIA 的那一套环境
# NatureIK 核心依赖
try:
    import pinocchio as pin
    from diffusion_policy.workspace.base_workspace import BaseWorkspace
    from diffusion_policy.dataset.robot_feature_utils import build_robot_feature_map
    from diffusion_policy.dataset.robot_specs import ROBOT_SPECS
except ImportError as e:
    cprint(f"[!] 警告：部分核心库未安装，请检查环境: {e}", "red")

# ==========================================
# 1. 异步实验日志记录器 (Queue + Thread)
# ==========================================
class ExperimentLogger:
    def __init__(self, log_dir="./experiment_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        cprint(f"[*] 实验日志系统已就绪，保存路径: {log_dir}", "cyan")

    def log(self, data: dict):
        self.queue.put(data)

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                # 累积一秒钟的数据批量写入，减少磁盘IO压力
                batch = []
                data = self.queue.get(timeout=1.0)
                batch.append(data)
                while not self.queue.empty() and len(batch) < 50:
                    batch.append(self.queue.get_nowait())
                
                if batch:
                    self._write_to_disk(batch)
            except queue.Empty:
                continue

    def _write_to_disk(self, batch):
        file_path = os.path.join(self.log_dir, f"ik_log_{datetime.now().strftime('%Y%m%d')}.parquet")
        new_df = pd.DataFrame(batch)
        if os.path.exists(file_path):
            try:
                old_df = pd.read_parquet(file_path)
                new_df = pd.concat([old_df, new_df], ignore_index=True)
            except Exception: pass
        new_df.to_parquet(file_path, index=False, engine='fastparquet')

# ==========================================
# 2. 求解器策略接口与实现
# ==========================================
class BaseIKSolver(ABC):
    @abstractmethod
    def solve_arm(self, q_curr: np.ndarray, ee_curr: np.ndarray, ee_target: np.ndarray) -> np.ndarray:
        """返回 6 维关节增量 delta_q"""
        pass

def _align_quaternions(p: np.ndarray):
    """四元数对齐逻辑，维持数据连续性"""
    p = p.reshape(1, -1)
    mask = p[..., 6] < 0 # 假设 w 在第 7 位 (idx 6)
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
        cprint(f"[*] NatureIK 加载成功: {ckpt}", "magenta")

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

class PyBulletIKSolver(BaseIKSolver):
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.pb_movable_joints = []
        self.pb_ee_index = -1
        
        cprint(f"[*] [Mode: PyBullet] 正在初始化物理引擎... URDF: {urdf_path}", "cyan")
        
        # 1. 初始化 PyBullet 环境 (DIRECT 模式不显示窗口)
        try:
            self.pb_client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            
            # 加载机器人模型
            self.pb_robot_id = p.loadURDF(str(self.urdf_path), [0, 0, 0], useFixedBase=True)
            
            # 2. 自动搜索活动关节和末端 Link (沿用你的逻辑并优化)
            for i in range(p.getNumJoints(self.pb_robot_id)):
                info = p.getJointInfo(self.pb_robot_id, i)
                if info[2] != p.JOINT_FIXED:
                    self.pb_movable_joints.append(i)
                
                l_name = info[12].decode("utf-8").lower()
                if any(x in l_name for x in ["end", "eef", "gripper_link"]):
                    self.pb_ee_index = i
            
            # 兜底：如果没搜到末端，假设最后一位是活动末端
            if self.pb_ee_index == -1 and len(self.pb_movable_joints) >= 6:
                self.pb_ee_index = self.pb_movable_joints[5]
                
            cprint(f"[*] PyBullet IK 引擎就绪！活动关节数: {len(self.pb_movable_joints)}", "green")
        except Exception as e:
            cprint(f"[!] PyBullet 初始化失败: {e}", "red")

    def solve_arm(self, q_curr, ee_curr, ee_target):
        """
        核心求解逻辑
        q_curr: 当前关节角度 (7D)
        ee_target: Pi0 给出的目标 [x, y, z, qx, qy, qz, qw] (7D)
        """
        target_pos = ee_target[:3]
        target_quat = ee_target[3:7] # 注意：PyBullet 默认也是 [x,y,z,w] 顺序

        # 1. 同步当前状态作为 IK 种子 (防止多解跳变)
        # 假设 q_curr 的前 len(movable_joints) 是我们需要控制的臂关节
        num_movable = len(self.pb_movable_joints)
        for i in range(min(num_movable, 6)): # 通常只同步前 6 个自由度
            p.resetJointState(self.pb_robot_id, self.pb_movable_joints[i], q_curr[i])

        # 2. 调用 PyBullet 内部求解器
        # 这里建议加上 limit 约束，防止求出反人类的角度
        ik_sol = p.calculateInverseKinematics(
            self.pb_robot_id,
            self.pb_ee_index,
            targetPosition=target_pos,
            targetOrientation=target_quat,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        # 3. 计算增量 delta_q 并返回
        # 注意：返回的维度必须与 q_curr[:6] 对齐
        q_sol = np.array(ik_sol[:6])
        return q_sol - q_curr[:6]

class PinocchioIKSolver(BaseIKSolver):
    def __init__(self, urdf_path, ee_link="link_eef"):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_id = self.model.getJointId(ee_link)
        cprint(f"[*] Pinocchio 数值解准备就绪", "blue")

    def solve_arm(self, q_curr, ee_curr, ee_target):
        target_se3 = pin.XYZQUATToSE3(ee_target[:7])
        q = q_curr[:6].copy()
        for _ in range(15): # 数值迭代
            pin.forwardKinematics(self.model, self.data, q)
            d_se3 = self.data.oMi[self.ee_id].inverse() * target_se3
            err = pin.log6(d_se3).np
            if np.linalg.norm(err) < 1e-4: break
            J = pin.computeJointJacobian(self.model, self.data, q, self.ee_id)
            q = pin.integrate(self.model, q, np.linalg.pinv(J) @ err)
        return q - q_curr[:6]

# ==========================================
# 3. FastAPI 逻辑封装
# ==========================================
app = FastAPI()

@app.post("/solve")
async def solve_endpoint(
    qpos: str = Form(...), 
    ee_current: str = Form(...), 
    pi0_action: str = Form(...)
):
    try:
        start_t = time.perf_counter()
        q_all = np.array(json.loads(qpos), dtype=np.float32)
        ee_all = np.array(json.loads(ee_current), dtype=np.float32)
        pi_act = np.array(json.loads(pi0_action), dtype=np.float32)
        
        ik: BaseIKSolver = app.state.solver
        
        # 左右臂拆解 (Pi0格式: [L_ee(7), L_g(1), R_ee(7), R_g(1)])
        et_l, g_l = pi_act[0:7], pi_act[7]
        et_r, g_r = pi_act[8:15], pi_act[15]

        # 核心求解
        dq_l = ik.solve_arm(q_all[0:7], ee_all[0:7], et_l)
        dq_r = ik.solve_arm(q_all[7:14], ee_all[7:14], et_r)

        # 积分叠加: $q_{next} = q_{curr} + \Delta q$
        q_next_l = q_all[0:7].copy(); q_next_l[:6] += dq_l[:6]
        q_next_r = q_all[7:14].copy(); q_next_r[:6] += dq_r[:6]

        # 最终控制向量缝合
        full_action = np.zeros(16)
        full_action[0:7], full_action[7] = q_next_l, g_l
        full_action[8:15], full_action[15] = q_next_r, g_r

        # 异步落盘记录
        log_data = {
            "solver": ik.__class__.__name__,
            "latency": (time.perf_counter() - start_t)*1000,
            "q_curr": q_all.tolist(),
            "dq_l": dq_l.tolist(),
            "dq_r": dq_r.tolist(),
            "full_action": full_action.tolist()
        }
        app.state.logger.log(log_data)

        return {"success": True, "actions": full_action.tolist()}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/health")
def health(): return {"status": "ok", "solver": app.state.solver.__class__.__name__}

# ==========================================
# 4. 交互式启动与工厂逻辑
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/data/ckpt/nature_ik/latest.pt")
    parser.add_argument("--urdf", default="./assets/airbot.urdf")
    parser.add_argument("--robot", default="airbot_single_arm")
    args = parser.parse_args()
    
    if not os.path.exists(args.urdf):
        cprint(f"[!] URDF 文件不存在: {args.urdf}", "red"); return
    cprint("\n--- IK Solver Launcher ---", "green", attrs=["bold"])
    print(" [1] NatureIK (Diffusion-ResNet)")
    print(" [2] Pinocchio (Numerical Jacobian)")
    print(" [3] cuRobo (GPU Skeleton - Required CUDA)")
    print(" [4] PyBullet (PyBullet - Damped Least Squares IK)")
    choice = input("\n请选择求解器编号: ").strip()
    try:
        if choice == "1":
            app.state.solver = NatureIKSolver(args.ckpt, args.robot)
        elif choice == "2":
            app.state.solver = PinocchioIKSolver(args.urdf)
        elif choice == "3":
            app.state.solver = curoboIKSolver(args.urdf)
        elif choice == "4":
            app.state.solver = PyBulletIKSolver(args.urdf)
        else:
            cprint("[!] 暂不支持该模式或未接入具体的库，退出。", "red"); return
    except Exception as e:
        cprint(f"[!] 初始化失败: {e}", "red"); return
    
    app.state.logger = ExperimentLogger()
    uvicorn.run(app, host="0.0.0.0", port=6162, log_level="warning")

if __name__ == "__main__":
    main()