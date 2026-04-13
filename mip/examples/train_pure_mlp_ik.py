"""Training pipeline for a pure supervised MLP IK baseline.

This baseline reuses the exact same MIP IK dataset interface:
    obs  = dataset["obs"]["state"]  -> normalized observation history
    act  = dataset["action"]        -> normalized target action sequence

Unlike train_ik.py, this script does not use flow matching. It simply learns
the direct mapping obs -> action with a plain MLP regressor.

Usage:
    python examples/train_pure_mlp_ik.py
    python examples/train_pure_mlp_ik.py task=ik_delta_joint
    python examples/train_pure_mlp_ik.py task=ik_absolute optimization.device=cpu
    python examples/train_pure_mlp_ik.py task=ik_delta_joint network.hidden_dims=[2048,1024,512]
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import hydra
import loguru
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from mip.dataset_utils import loop_dataloader
from mip.datasets.ik_parquet_dataset import IKParquetDataset
from mip.logger import Logger
from mip.networks.pure_mlp_ik import PureMLPIK
from mip.torch_utils import limit_threads, set_seed

torch.set_float32_matmul_precision("high")


@contextmanager
def timed(section: str, record_dict: dict[str, list[float]]):
    start = time.perf_counter()
    yield
    record_dict[section].append(time.perf_counter() - start)


def make_ik_dataset(task_config, split: str = "train") -> IKParquetDataset:
    data_dir = os.path.expanduser(task_config.dataset_path)
    if task_config.env_name == "ik_delta_joint":
        mode = "delta_joint"
    elif task_config.env_name == "ik_absolute":
        mode = "absolute"
    else:
        raise ValueError(f"Unknown IK env_name: {task_config.env_name!r}")

    return IKParquetDataset(
        data_dir=data_dir,
        mode=mode,
        horizon=task_config.horizon,
        obs_steps=task_config.obs_steps,
        val_ratio=getattr(task_config, "val_dataset_percentage", 0.1),
        split=split,
    )


def build_loss_fn(loss_name: str):
    loss_name = loss_name.lower()
    if loss_name in {"l2", "mse"}:
        return F.mse_loss
    if loss_name == "l1":
        return F.l1_loss
    if loss_name == "smooth_l1":
        return F.smooth_l1_loss
    raise ValueError(f"Unsupported regression loss: {loss_name}")


def update_ema(model_ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    if model_ema is None:
        return
    with torch.no_grad():
        for ema_param, param in zip(model_ema.parameters(), model.parameters(), strict=True):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)


def save_pure_mlp_bundle(
    path: Path,
    model: PureMLPIK,
    model_ema: PureMLPIK | None,
    config,
    normalizer: dict,
    metadata: dict | None = None,
) -> None:
    bundle = {
        "framework": "pure_mlp_ik",
        "config": OmegaConf.to_container(config, resolve=True),
        "normalizer": {
            "obs_state": normalizer["obs"]["state"],
            "action": normalizer["action"],
        },
        "model": model.state_dict(),
        "model_ema": model_ema.state_dict() if model_ema is not None else None,
        "metadata": metadata or {},
    }
    torch.save(bundle, path)


def make_model(config) -> PureMLPIK:
    hidden_dims = getattr(config.network, "hidden_dims", [1024, 1024, 512])
    return PureMLPIK(
        obs_dim=int(config.task.obs_dim),
        obs_steps=int(config.task.obs_steps),
        act_dim=int(config.task.act_dim),
        horizon=int(config.task.horizon),
        hidden_dims=hidden_dims,
        dropout=float(getattr(config.network, "dropout", 0.1)),
        activation=str(getattr(config.network, "activation", "gelu")),
        use_layernorm=bool(getattr(config.network, "use_layernorm", False)),
    )


def compute_val_metrics(
    model: PureMLPIK,
    val_loader: torch.utils.data.DataLoader,
    config,
    loss_fn,
    action_normalizer,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses = []
    maes = []
    rmses = []

    with torch.no_grad():
        for batch in val_loader:
            obs = batch["obs"]["state"].to(config.optimization.device)
            obs = obs[:, : config.task.obs_steps, :]
            target = batch["action"].to(config.optimization.device)
            target = target[:, : config.task.horizon, :]

            pred = model(obs)
            losses.append(loss_fn(pred, target).item())

            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            pred_denorm = action_normalizer.unnormalize(pred_np)
            target_denorm = action_normalizer.unnormalize(target_np)
            diff = pred_denorm - target_denorm
            maes.append(float(np.mean(np.abs(diff))))
            rmses.append(float(np.sqrt(np.mean(np.square(diff)))))

    if was_training:
        model.train()
    else:
        model.eval()
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_mae_denorm": float(np.mean(maes)) if maes else float("nan"),
        "val_rmse_denorm": float(np.mean(rmses)) if rmses else float("nan"),
    }


@hydra.main(config_path="configs", config_name="pure_mlp_ik", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    config = cfg

    limit_threads(1)
    set_seed(config.optimization.seed)
    os.makedirs(config.log.log_dir, exist_ok=True)

    device = torch.device(config.optimization.device)
    loss_fn = build_loss_fn(config.optimization.norm_type)

    loguru.logger.info(f"Task:   {config.task.env_name}")
    loguru.logger.info(f"Device: {device}")
    loguru.logger.info("Loading IK datasets ...")

    dataset = make_ik_dataset(config.task, split="train")
    val_dataset = make_ik_dataset(config.task, split="val")
    val_dataset._normalizer = dataset._normalizer

    if len(dataset) == 0:
        raise ValueError("Training dataset is empty. Check dataset_path / val split.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty. Check dataset_path / val split.")

    sample = dataset[0]
    actual_obs_dim = int(sample["obs"]["state"].shape[-1])
    if actual_obs_dim != int(config.task.obs_dim):
        loguru.logger.warning(
            f"obs_dim mismatch: config={config.task.obs_dim}, actual data={actual_obs_dim}. "
            f"Overriding with data value."
        )
        config.task.obs_dim = actual_obs_dim

    actual_act_dim = int(sample["action"].shape[-1])
    if actual_act_dim != int(config.task.act_dim):
        loguru.logger.warning(
            f"act_dim mismatch: config={config.task.act_dim}, actual data={actual_act_dim}. "
            f"Overriding with data value."
        )
        config.task.act_dim = actual_act_dim

    model = make_model(config).to(device)
    model_ema = deepcopy(model).eval().requires_grad_(False) if config.optimization.ema_rate < 1 else None

    total_params = sum(p.numel() for p in model.parameters())
    loguru.logger.info(f"PureMLPIK parameters: {total_params:,}")
    loguru.logger.info(
        f"Train samples: {len(dataset):,} | Val samples: {len(val_dataset):,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.lr,
        weight_decay=config.optimization.weight_decay,
    )
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.optimization.gradient_steps,
    )

    pin_memory = device.type == "cuda"
    train_num_workers = int(getattr(config.optimization, "train_num_workers", 4))
    val_num_workers = int(getattr(config.optimization, "val_num_workers", 2))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=train_num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        persistent_workers=train_num_workers > 0,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.optimization.batch_size,
        num_workers=val_num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        persistent_workers=val_num_workers > 0,
        drop_last=False,
    )
    loop_loader = loop_dataloader(dataloader)

    action_normalizer = dataset.get_normalizer()["action"]
    run_logger = Logger(config)

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = Path(os.getcwd()) / "checkpoints" / run_timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    exp_tag = getattr(config.log, "exp_tag", "") or ""
    ckpt_suffix = "_pure_mlp_ik"
    if exp_tag:
        ckpt_suffix += f"_{exp_tag}"

    loguru.logger.info(
        f"Checkpoints will be saved to: {checkpoint_dir}  (suffix: '{ckpt_suffix}')"
    )

    best_val_loss = float("inf")
    perf_times = {
        "data_load": [],
        "update": [],
        "total_step": [],
    }
    train_info_list = []
    start_time = time.time()

    pbar = tqdm(range(config.optimization.gradient_steps), dynamic_ncols=True)
    for step in pbar:
        with timed("total_step", perf_times):
            with timed("data_load", perf_times):
                batch = next(loop_loader)
                obs = batch["obs"]["state"].to(device)
                obs = obs[:, : config.task.obs_steps, :]
                target = batch["action"].to(device)
                target = target[:, : config.task.horizon, :]

            with timed("update", perf_times):
                pred = model(obs)
                loss = loss_fn(pred, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if config.optimization.grad_clip_norm:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.optimization.grad_clip_norm,
                    )
                else:
                    grad_norm = torch.tensor(0.0, device=device)

                optimizer.step()
                lr_scheduler.step()
                update_ema(model_ema, model, config.optimization.ema_rate)

            train_info_list.append(
                {
                    "loss": float(loss.item()),
                    "grad_norm": float(grad_norm.item()),
                }
            )

        if (step + 1) % config.log.log_freq == 0:
            freq = config.log.log_freq
            metrics = {
                "step": step,
                "total_time": time.time() - start_time,
                "lr": lr_scheduler.get_last_lr()[0],
                "loss": float(np.mean([d["loss"] for d in train_info_list])),
                "grad_norm": float(np.mean([d["grad_norm"] for d in train_info_list])),
                "perf/data_load_ms": np.mean(perf_times["data_load"][-freq:]) * 1000,
                "perf/update_ms": np.mean(perf_times["update"][-freq:]) * 1000,
                "perf/steps_per_sec": 1.0 / np.mean(perf_times["total_step"][-freq:]),
            }
            run_logger.log(metrics, category="train")
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}", lr=f"{metrics['lr']:.2e}", step=step + 1)
            train_info_list = []

        if (step + 1) % config.log.eval_freq == 0:
            eval_model = model_ema if model_ema is not None else model
            val_metrics = compute_val_metrics(
                eval_model,
                val_loader,
                config,
                loss_fn,
                action_normalizer,
            )
            run_logger.log({"step": step, **val_metrics}, category="eval")
            loguru.logger.info(
                f"[Step {step + 1:>7d}] val_loss = {val_metrics['val_loss']:.6f} | "
                f"val_mae_denorm = {val_metrics['val_mae_denorm']:.6f}"
            )

            val_loss = val_metrics["val_loss"]
            best_name = f"step={step + 1:07d}-val_loss={val_loss:.6f}{ckpt_suffix}.pt"
            save_pure_mlp_bundle(
                checkpoint_dir / best_name,
                model=model,
                model_ema=model_ema,
                config=config,
                normalizer=dataset.get_normalizer(),
                metadata={"step": step + 1, **val_metrics},
            )
            loguru.logger.info(f"Saved checkpoint -> {best_name}")

            existing = list(checkpoint_dir.glob(f"step=*-val_loss=*{ckpt_suffix}.pt"))

            def parse_loss(path: Path) -> float:
                try:
                    return float(path.stem.split("-val_loss=")[1].split(ckpt_suffix)[0])
                except Exception:
                    return float("inf")

            existing.sort(key=parse_loss)
            for old_ckpt in existing[3:]:
                old_ckpt.unlink()
                loguru.logger.info(f"Removed old checkpoint -> {old_ckpt.name}")

            best_val_loss = min(best_val_loss, val_loss)
            pbar.set_postfix(val_loss=f"{val_loss:.5f}", step=step + 1)

        if (step + 1) % config.log.save_freq == 0:
            latest_name = f"latest{ckpt_suffix}.pt"
            save_pure_mlp_bundle(
                checkpoint_dir / latest_name,
                model=model,
                model_ema=model_ema,
                config=config,
                normalizer=dataset.get_normalizer(),
                metadata={"step": step + 1},
            )
            loguru.logger.info(f"Latest checkpoint saved at step {step + 1} -> {latest_name}")

    if best_val_loss == float("inf"):
        eval_model = model_ema if model_ema is not None else model
        val_metrics = compute_val_metrics(
            eval_model,
            val_loader,
            config,
            loss_fn,
            action_normalizer,
        )
        best_val_loss = val_metrics["val_loss"]
        loguru.logger.info(
            "No validation was run during training; computed final validation metrics "
            f"at step {config.optimization.gradient_steps}: {val_metrics}"
        )

    final_name = (
        f"step={config.optimization.gradient_steps:07d}"
        f"-val_loss={best_val_loss:.6f}"
        f"{ckpt_suffix}-final.pt"
    )
    save_pure_mlp_bundle(
        checkpoint_dir / final_name,
        model=model,
        model_ema=model_ema,
        config=config,
        normalizer=dataset.get_normalizer(),
        metadata={"step": config.optimization.gradient_steps, "best_val_loss": best_val_loss},
    )
    loguru.logger.info(f"Training complete -> {final_name}")
    run_logger.finish(agent=None)


if __name__ == "__main__":
    main()
