import torch
import numpy as np
import json
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from termcolor import cprint

"""
IK 在线推理服务（FastAPI）。

功能概览
--------
1) 启动时加载 diffusion policy checkpoint；
2) 暴露 /solve 接口，接收机器人当前状态与 Pi0 目标末端动作；
3) 通过单臂 IK policy 分别求解左右臂 delta_q；
4) 返回双臂 16 维绝对关节+夹爪控制信号。

输入/输出约定（非常关键）
-----------------------
- qpos:       14 维，左右臂各 7 维（前 6 维臂关节 + 第 7 维夹爪）
- ee_current: 14 维，左右臂各 7 维末端位姿（与训练数据保持一致）
- pi0_action: 16 维，格式为 [L_ee(7), L_g(1), R_ee(7), R_g(1)]

输出 actions: 16 维，格式为 [L_q(7), L_g(1), R_q(7), R_g(1)]
其中 q 是“绝对关节角”，由 q_next = q_curr + delta_q 计算得到。

注意
----
- 本服务假定模型是“单臂模型”，双臂通过调用两次 solve_arm 实现。
- 本服务强依赖训练时的特征构造规则（四元数对齐、切片维度等）。
"""

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.robot_feature_utils import build_robot_feature_map
from diffusion_policy.dataset.robot_specs import ROBOT_SPECS

app = FastAPI()

def _align_quaternions(pose_array: np.ndarray, w_idx: int = 6, quat_start: int = 3) -> np.ndarray:
    """
    将四元数对齐到同一半球（w >= 0），避免 q 与 -q 的双覆盖带来不稳定。

    与训练/评估脚本保持一致，确保线上推理和离线评估的数据预处理一致。
    """
    if pose_array.shape[-1] < 7: return pose_array
    neg_mask = pose_array[..., w_idx] < 0
    pose_array[neg_mask, quat_start:quat_start+4] = -pose_array[neg_mask, quat_start:quat_start+4]
    return pose_array

class NatureIKExpert:
    """
    封装模型加载与单臂 IK 推理。

    参数
    ----
    checkpoint_path: str
        训练产生的 .ckpt 路径（torch.load 可读取的 payload）。
    target_robot: str
        当 use_robot_feature=True 时，用于选择 ROBOT_SPECS 中对应结构编码。
    """
    def __init__(self, checkpoint_path, target_robot="airbot_single_arm"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1) 加载模型（严格对齐 eval_ik.py 的恢复逻辑）
        # payload 内含 cfg / 模型权重 / 优化器状态等。
        payload = torch.load(open(checkpoint_path, "rb"), pickle_module=__import__('dill'))
        self.cfg = payload["cfg"]
        
        # 根据 cfg._target_ 动态构建 workspace，再恢复参数。
        cls = __import__('hydra').utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg)
        workspace.load_payload(payload)

        # 推理优先使用 EMA 模型（若训练启用 use_ema）。
        self.policy = workspace.ema_model if self.cfg.training.use_ema else workspace.model
        self.policy.to(self.device).eval()
        
        # 2) 机器人结构特征编码（可选）
        # 若训练配置 use_robot_feature=True，则线上推理必须拼接同维度结构特征。
        self.use_robot_feature = self.cfg.task.dataset.get('use_robot_feature', False)
        self.robot_feature = None
        if self.use_robot_feature:
            max_joints = self.cfg.task.dataset.get('max_joints', 16)
            feature_map = build_robot_feature_map(ROBOT_SPECS, max_joints=max_joints)
            self.robot_feature = feature_map[target_robot].astype(np.float32)

        cprint(f"[*] NatureIK 专家节点启动。目标形态: {target_robot}", "green")

    def solve_arm(self, q_curr, ee_curr, ee_target):
        """
        单臂 IK 推理：输入当前关节/末端 + 目标末端，输出 delta_q。

        输入维度约定
        ----------
        q_curr:   >=7 维（只使用前 6 维作为臂关节）
        ee_curr:  >=7 维（只使用前 7 维）
        ee_target >=7 维（只使用前 7 维）

        返回
        ----
        delta_q: np.ndarray
            policy 预测的单步（或序列首步）关节增量向量。
        """
        # 四元数半球对齐（必须与训练预处理一致）
        ee_curr = _align_quaternions(ee_curr.reshape(1, -1))[0]
        ee_target = _align_quaternions(ee_target.reshape(1, -1))[0]
        
        # 组装基础观测：20 维 = q[:6] + ee_curr[:7] + ee_target[:7]
        obs_vec = np.concatenate([q_curr[:6], ee_curr[:7], ee_target[:7]], axis=-1)
        
        # 若开启结构编码，将固定 robot_feature 拼接到观测末尾。
        if self.use_robot_feature:
            obs_vec = np.concatenate([obs_vec, self.robot_feature], axis=-1)
            
        # 构造推理输入张量：(B=1, T=1, D)
        obs_ts = torch.from_numpy(obs_vec).float().to(self.device).view(1, 1, -1)
        
        with torch.no_grad():
            res = self.policy.predict_action({"obs": obs_ts})
            # 使用 action_pred 的首个时间步作为当前控制增量
            delta_q = res['action_pred'].cpu().numpy()[0, 0] 
            
        return delta_q

@app.post("/solve")
# async 接口可被并发调用；模型推理本身仍在单进程内执行。
async def solve(
    qpos: str = Form(...),          # 机器人端当前关节 (14D)
    ee_current: str = Form(...),    # 机器人端当前末端 (14D)
    pi0_action: str = Form(...)     # Pi0 返回的原始动作 (16D)
):
    try:
        # 1. 解析数据
        q_all = np.array(json.loads(qpos), dtype=np.float32)
        ee_all = np.array(json.loads(ee_current), dtype=np.float32)
        pi_act = np.array(json.loads(pi0_action), dtype=np.float32)
        
        # 从 app.state 取已初始化的 IK 专家实例
        ik: NatureIKExpert = app.state.ik_expert

        # 2) 数据拆分 (双臂 16 维动作定义：8D + 8D)
        # Left: EEF(7D) + G(1D) | Right: EEF(7D) + G(1D)
        ee_target_l, g_l = pi_act[0:7], pi_act[7]
        ee_target_r, g_r = pi_act[8:15], pi_act[15]

        # 3) 双臂求解
        # 注：这里是“逻辑并行”——单臂模型调用两次，分别处理左右臂。
        dq_l = ik.solve_arm(q_all[0:7], ee_all[0:7], ee_target_l)
        dq_r = ik.solve_arm(q_all[7:14], ee_all[7:14], ee_target_r)

        # 4) 缝合全量控制信号
        # 绝对关节角：q_next = q_curr + delta_q（仅更新每臂前 6 维）
        q_next_l = q_all[0:7].copy()
        q_next_l[:6] += dq_l[:6]
        
        q_next_r = q_all[7:14].copy()
        q_next_r[:6] += dq_r[:6]

        # 拼接最终 16D 信号 [L_q(7), L_g(1), R_q(7), R_g(1)]
        full_action = np.zeros(16)
        full_action[0:7] = q_next_l
        full_action[7]   = g_l
        full_action[8:15] = q_next_r
        full_action[15]  = g_r

        return {"success": True, "actions": full_action.tolist()}

    except Exception as e:
        # 将异常包装为 JSON，便于上游控制系统记录日志与重试。
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    # 直接运行脚本时使用该入口。
    # 若用 `uvicorn ik_server:app` 启动，需要在启动脚本中手动初始化 app.state.ik_expert。
    ckpt = "/data/ckpt/nature_ik/latest.pt"
    app.state.ik_expert = NatureIKExpert(ckpt)
    uvicorn.run(app, host="0.0.0.0", port=6162)