"""
TrainIKWorkspace
================
IK Diffusion Transformer 的训练 Workspace。

与 TrainDiffusionResNetLowdimWorkspace 的主要差异
-----------------------------------------------
1. 使用 IKDiffuserPolicy（cross-attention Transformer）替代 ResNet Policy。
2. batch 格式：obs = {'state_quat': (B,1,7)}，action = (B,1,6)
3. 混合归一化：四元数 [3:7] 强制 scale=1、offset=0，位置 [0:3] 正常归一化。
4. 无 env_runner（纯监督，不做在线 rollout）。
5. TopK checkpoint 按 val_loss 保存。

启动方式
--------
  cd /home/user/yjh/diffusion_policy
  python train.py --config-name=train_ik_dit \\
    task.dataset.data_dir=/path/to/data/chunk-000 \\
    task.dataset.scan_chunk_subdirs=true
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import os
import pathlib
import random

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.policy.ik_diffuser_policy import IKDiffuserPolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainIKWorkspace(BaseWorkspace):
    """
    IK Diffusion Transformer 训练 Workspace。
    """

    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 实例化 Policy
        self.model: IKDiffuserPolicy = hydra.utils.instantiate(cfg.policy)

        # EMA 模型（可选）
        self.ema_model: IKDiffuserPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # 优化器
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # 断点续训
        if cfg.training.resume:
            latest_ckpt = self.get_checkpoint_path()
            if latest_ckpt.is_file():
                print(f"[TrainIKWorkspace] Resuming from {latest_ckpt}")
                self.load_checkpoint(path=latest_ckpt)

        # -------- 数据集 --------
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # 设置归一化
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        # 混合归一化：四元数维度直通（scale=1, offset=0）
        self._apply_hybrid_quat_norm()

        # -------- LR 调度 --------
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs
            ) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        # -------- EMA --------
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # -------- WandB --------
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update({"output_dir": self.output_dir})

        # -------- Checkpoint --------
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # -------- 设备转移 --------
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        # -------- 训练循环 --------
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
                step_log = {}
                train_losses = []

                self.model.train()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if cfg.training.use_ema:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == len(train_dataloader) - 1
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if cfg.training.max_train_steps is not None and \
                                batch_idx >= cfg.training.max_train_steps - 1:
                            break

                step_log['train_loss'] = np.mean(train_losses)

                # -------- 验证 --------
                if (self.epoch % cfg.training.val_every) == 0:
                    policy = self.ema_model if cfg.training.use_ema else self.model
                    policy.eval()
                    with torch.no_grad():
                        val_losses = []
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Val epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as vepoch:
                            for batch_idx, batch in enumerate(vepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                val_losses.append(self.model.compute_loss(batch))
                                if cfg.training.max_val_steps is not None and \
                                        batch_idx >= cfg.training.max_val_steps - 1:
                                    break
                        if val_losses:
                            step_log['val_loss'] = torch.mean(
                                torch.stack(val_losses)
                            ).item()
                    policy.train()

                # -------- Checkpoint --------
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = {k.replace('/', '_'): v for k, v in step_log.items()}
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

    def _apply_hybrid_quat_norm(self):
        """
        混合归一化：四元数维度（obs/state_quat 的 [3:7]）强制 scale=1、offset=0，
        位置维度（[0:3]）保留 dataset 统计量。
        """
        for key in self.model.normalizer.params_dict:
            if 'state_quat' in key:
                stat = self.model.normalizer[key]
                stat.params_dict['scale'][3:] = 1.0
                stat.params_dict['offset'][3:] = 0.0
                print(f"[HybridNorm] Applied to '{key}': "
                      f"pos scale={stat.params_dict['scale'][:3].tolist()}, "
                      f"quat scale forced to 1.0")
        # EMA 同步
        if self.ema_model is not None:
            self.ema_model.set_normalizer(self.model.normalizer)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainIKWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
