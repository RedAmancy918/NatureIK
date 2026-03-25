import torch
import torch.nn.functional as F
import pytorch_kinematics as pk

class DifferentiableIKHelper:
    def __init__(self, urdf_path, end_effector_link_name, dof=6, device='cuda', 
                 tool_offset_z=0.0, apply_z180=False):
        """
        :param tool_offset_z: 夹爪长度（米）。因为你没有夹爪，这里给 0.0
        :param apply_z180: 是否应用 Z 轴 180 度旋转修正（解决背对背安装问题）
        """
        self.device = device
        self.dof = dof
        self.tool_offset_z = tool_offset_z
        self.apply_z180 = apply_z180
        
        # 加载 URDF
        with open(urdf_path, 'rb') as f:
            urdf_data = f.read()
        self.chain = pk.build_chain_from_urdf(urdf_data).to(device=device)
        self.ee_link = end_effector_link_name

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
        # 截取前 6 个自由度
        if joint_angles.shape[-1] > self.dof:
            joint_angles_fk = joint_angles[..., :self.dof]
        else:
            joint_angles_fk = joint_angles

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

        # 3. (关键) 应用 Z轴 180 度旋转修正
        # 对应你之前代码的 R_FIX_180
        if self.apply_z180:
            # 坐标变换：P_new = R_180 @ P_old
            # PyTorch matmul 广播机制: (3,3) @ (B, 3, 1) -> (B, 3, 1)
            # 或者简单的: x'=-x, y'=-y
            pos = torch.matmul(pos, self.rot_180.T) # 注意转置以匹配 (B, 3) 形状
            # R_new = R_180 @ R_old
            rot_mat = torch.matmul(self.rot_180, rot_mat)

        return pos, rot_mat

    def compute_guidance_gradient(self, latents, t, model, noise_scheduler, 
                                  target_pose_dict, action_stats, global_cond=None,
                                  pos_weight=1.0, rot_weight=0.5):
        with torch.enable_grad():
            latents = latents.detach().requires_grad_(True)
            
            # 预测噪声
            model_output = model(latents, t, global_cond=global_cond)
            if isinstance(model_output, dict): noise_pred = model_output['sample']
            else: noise_pred = model_output

            # 还原 x0
            alpha_prod_t = noise_scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            alpha_prod_t = alpha_prod_t.to(latents.device).view(-1, 1, 1)
            beta_prod_t = beta_prod_t.to(latents.device).view(-1, 1, 1)
            
            pred_normalized_action = (latents - (beta_prod_t ** 0.5) * noise_pred) / (alpha_prod_t ** 0.5)
            pred_physical_action = self._unnormalize_action(pred_normalized_action, action_stats)

            # 取最后一帧进行约束
            final_pose_action = pred_physical_action[:, -1, :] 
            
            # 计算 FK (含 180 度修正)
            curr_pos, curr_rot = self.get_current_pose(final_pose_action)
            
            loss = 0.0
            target_pos = target_pose_dict['pos']
            loss_pos = F.mse_loss(curr_pos, target_pos)
            loss += loss_pos * pos_weight
            
            # 旋转约束 (如果有 quat)
            if 'quat' in target_pose_dict and rot_weight > 0:
                # 把 target quat 转成 matrix 来算 loss 会更稳定
                # 或者如果你不想引入 matrix 转换，可以用 pytorch_kinematics.transforms
                # 简单起见，这里如果不方便转 matrix，可以暂且只约束 pos
                pass 
            
            grads = torch.autograd.grad(loss, latents)[0]
            
        return grads