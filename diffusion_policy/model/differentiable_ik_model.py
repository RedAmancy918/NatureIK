import torch
import torch.nn.functional as F
import pytorch_kinematics as pk


# ─────────────────────────────────────────────────────────────────────────────
# TTGLoss: 统一的测试时引导损失，Diffusion TTG 和 MIP 梯度投影共用
# ─────────────────────────────────────────────────────────────────────────────
class TTGLoss:
    """Test-Time Guidance loss，三项组合：
        L_TTG = λ_pose  · L_pose           # FK 末端位置误差
              + λ_hist  · L_hist           # 历史 delta_q 方向一致性,在MIP中会失效
              + λ_smooth· L_smooth         # 步长抑制，防 guidance 过猛。这也是一个对于MIP没用的东西

    两个场景均可调用：
      - Diffusion TTG：在去噪循环里对 latent 求梯度（由 DifferentiableIKHelper 调用）
      - MIP 梯度投影：在推理输出后对 delta_q 直接求梯度
    """

    def __init__(
        self,
        lambda_pose: float = 1.0,
        lambda_hist: float = 0.3,
        lambda_smooth: float = 0.05,
    ):
        self.lambda_pose   = lambda_pose
        self.lambda_hist   = lambda_hist
        self.lambda_smooth = lambda_smooth

    def compute(
        self,
        curr_pos: torch.Tensor,          # [B, 3]  FK 得到的末端位置
        curr_rot: torch.Tensor,          # [B, 3, 3]  FK 得到的末端旋转矩阵
        target_pos: torch.Tensor,        # [B, 3]  目标末端位置
        dq_pred: torch.Tensor,           # [B, 6]  本步预测的 delta_q
        dq_prev: torch.Tensor | None = None,  # [B, 6]  上一步执行的 delta_q（来自历史 buffer）
        target_rot: torch.Tensor | None = None,  # [B, 3, 3]  目标旋转（可选）
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=curr_pos.device)

        # ── L_pose: 末端位置 MSE ──────────────────────────────────────────────
        loss_pose = F.mse_loss(curr_pos, target_pos)
        loss = loss + self.lambda_pose * loss_pose

        # ── L_pose_rot: 末端旋转（可选，用 Frobenius 距离）────────────────────
        if target_rot is not None:
            loss_rot = F.mse_loss(curr_rot, target_rot)
            loss = loss + self.lambda_pose * 0.3 * loss_rot

        # ── L_hist: 历史方向一致性 ────────────────────────────────────────────
        # 当前预测方向与上一步方向的余弦距离，鼓励延续人类示教的恢复风格
        if dq_prev is not None and self.lambda_hist > 0:
            cos_sim = F.cosine_similarity(dq_pred, dq_prev, dim=-1)  # [B]
            loss_hist = (1.0 - cos_sim).mean()
            loss = loss + self.lambda_hist * loss_hist

        # ── L_smooth: 步长抑制 ────────────────────────────────────────────────
        if self.lambda_smooth > 0:
            loss_smooth = dq_pred.pow(2).mean()
            loss = loss + self.lambda_smooth * loss_smooth

        return loss


class DifferentiableIKHelper:
    def __init__(self, urdf_path, end_effector_link_name, dof=6, device='cuda',
                 tool_offset_z=0.0, apply_z180=False,
                 base_offset=None):
        """
        :param tool_offset_z: 额外工具 Z 方向偏移（米），通常为 0
        :param apply_z180:    是否应用 Z 轴 180 度旋转修正（背对背安装）
        :param base_offset:   机器人基座在世界坐标系中的平移偏移 [x, y, z]（米）。
                              FK 输出在基座坐标系，加上此偏移后与数据集中的
                              world-frame eef_tgt 对齐。
                              例：airbot G2 实测值约 [0.097, -0.006, -0.091]
        """
        self.device = device
        self.dof = dof
        self.tool_offset_z = tool_offset_z
        self.apply_z180 = apply_z180

        # 基座世界坐标系偏移（常数平移）
        if base_offset is not None:
            self.base_offset = torch.tensor(base_offset, dtype=torch.float32, device=device)
        else:
            self.base_offset = None
        
        # 加载 URDF
        with open(urdf_path, 'rb') as f:
            urdf_data = f.read()
        self.chain = pk.build_chain_from_urdf(urdf_data).to(device=device)
        self.ee_link = end_effector_link_name
        # 查询运动链的可动关节总数，用于后续补零
        self._n_joints = len(self.chain.get_joint_parameter_names())

        # 预先定义旋转矩阵 [-1, 0, 0; 0, -1, 0; 0, 0, 1] 以节省计算
        if self.apply_z180:
            self.rot_180 = torch.tensor([
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], device=device)

    def _unnormalize_action(self, normalized_action, stats):
        if stats is None: return normalized_action
        if 'min' in stats and 'max' in stats:
            min_val = stats['min'].to(self.device)
            max_val = stats['max'].to(self.device)
            return (normalized_action + 1) / 2 * (max_val - min_val) + min_val
        elif 'mean' in stats and 'scale' in stats:
            mean = stats['mean'].to(self.device)
            scale = stats['scale'].to(self.device)
            return normalized_action * scale + mean
        return normalized_action

    def get_current_pose(self, joint_angles):
        # 截取前 dof 个受控关节
        if joint_angles.shape[-1] > self.dof:
            joint_angles_fk = joint_angles[..., :self.dof]
        else:
            joint_angles_fk = joint_angles

        # 若运动链关节数 > dof（如有手指/夹爪关节），补零对齐
        if self._n_joints > joint_angles_fk.shape[-1]:
            pad_size = self._n_joints - joint_angles_fk.shape[-1]
            pad = torch.zeros(
                *joint_angles_fk.shape[:-1], pad_size,
                dtype=joint_angles_fk.dtype,
                device=joint_angles_fk.device,
            )
            joint_angles_fk = torch.cat([joint_angles_fk, pad], dim=-1)

        # 计算 FK
        ret = self.chain.forward_kinematics(joint_angles_fk)
        if self.ee_link not in ret:
            raise ValueError(f"Link {self.ee_link} not found in URDF.")
            
        m = ret[self.ee_link].get_matrix() # [Batch, 4, 4]
        
        # 1. 提取原始旋转和平移
        pos = m[:, :3, 3]
        rot_mat = m[:, :3, :3]
        
        # 2. (可选) 加上夹爪 Offset (你设为0，这步其实不影响)
        if self.tool_offset_z != 0.0:
            offset_vec = torch.tensor([0.0, 0.0, self.tool_offset_z], device=self.device)
            pos += torch.matmul(rot_mat, offset_vec)

        # 3. (可选) 应用 Z轴 180 度旋转修正
        if self.apply_z180:
            pos = torch.matmul(pos, self.rot_180.T)
            rot_mat = torch.matmul(self.rot_180, rot_mat)

        # 4. 加上基座在世界坐标系中的平移偏移，将 FK 结果从基座坐标系转换到世界坐标系
        #    使 FK 输出与数据集中记录的 eef_tgt（世界坐标系）对齐
        if self.base_offset is not None:
            pos = pos + self.base_offset

        return pos, rot_mat

    def compute_guidance_gradient(
        self,
        latents,
        t,
        model,
        noise_scheduler,
        target_pose_dict,
        action_stats,
        global_cond=None,
        dq_prev: torch.Tensor | None = None,   # [B, 6] 上一步历史 delta_q
        ttg_loss: TTGLoss | None = None,        # 共享 TTGLoss 实例，None 时用默认
    ):
        """在 Diffusion 去噪循环中计算 TTG 梯度。

        Args:
            latents:          当前去噪 latent [B, T, act_dim]
            t:                当前 timestep
            model:            diffusion 网络
            noise_scheduler:  DDPM/DDIM scheduler
            target_pose_dict: {'pos': [B,3], 'rot': [B,3,3] (可选)}
            action_stats:     归一化统计量（用于还原物理单位）
            global_cond:      conditioning（Transformer 或 UNet global_cond）
            dq_prev:          历史 delta_q，用于 L_hist；None 时跳过该项
            ttg_loss:         TTGLoss 实例；None 时使用默认权重
        """
        if ttg_loss is None:
            ttg_loss = TTGLoss()

        with torch.enable_grad():
            latents = latents.detach().requires_grad_(True)

            # ── 1. 预测噪声，还原 x0 ─────────────────────────────────────────
            model_output = model(latents, t, global_cond=global_cond)
            if isinstance(model_output, dict):
                noise_pred = model_output['sample']
            else:
                noise_pred = model_output

            alpha_prod_t = noise_scheduler.alphas_cumprod[t]
            beta_prod_t  = 1 - alpha_prod_t
            alpha_prod_t = alpha_prod_t.to(latents.device).view(-1, 1, 1)
            beta_prod_t  = beta_prod_t.to(latents.device).view(-1, 1, 1)

            pred_normalized = (latents - (beta_prod_t ** 0.5) * noise_pred) / (alpha_prod_t ** 0.5)
            pred_physical   = self._unnormalize_action(pred_normalized, action_stats)

            # ── 2. 取最后一帧做约束（delta_q → 绝对关节角 → FK）────────────────
            dq_pred = pred_physical[:, -1, :]              # [B, act_dim]
            curr_pos, curr_rot = self.get_current_pose(dq_pred)

            # ── 3. 计算 TTG loss ──────────────────────────────────────────────
            target_rot = target_pose_dict.get('rot', None)
            loss = ttg_loss.compute(
                curr_pos   = curr_pos,
                curr_rot   = curr_rot,
                target_pos = target_pose_dict['pos'],
                dq_pred    = dq_pred,
                dq_prev    = dq_prev,
                target_rot = target_rot,
            )

            grads = torch.autograd.grad(loss, latents)[0]

        return grads