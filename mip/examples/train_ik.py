"""Training pipeline for NatureIK dataset using MIP flow matching.

No simulation environment — pure offline imitation learning from Parquet data.
Validation loss is computed every eval_freq steps using the EMA model (no backward).

Usage:
    python examples/train_ik.py task=ik_delta_joint
    python examples/train_ik.py task=ik_absolute
    python examples/train_ik.py task=ik_delta_joint network.emb_dim=256 optimization.batch_size=512
"""

import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import hydra
import loguru
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from mip.agent import TrainingAgent
from mip.config import Config
from mip.dataset_utils import loop_dataloader
from mip.datasets.ik_parquet_dataset import IKParquetDataset
from mip.logger import Logger
from mip.scheduler import WarmupAnnealingScheduler
from mip.torch_utils import limit_threads, set_seed

torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def timed(section: str, record_dict: dict):
    """Records wall-clock time of a code block into record_dict[section]."""
    start = time.perf_counter()
    yield
    record_dict[section].append(time.perf_counter() - start)


def save_mip_bundle(
    path: Path,
    agent: TrainingAgent,
    config,
    normalizer: dict,
) -> None:
    """Save a self-contained inference bundle for MIPIKSolver.

    Bundle layout:
        config      - OmegaConf config dict (serialisable, no Python objects)
        normalizer  - {"obs": {"state": CompositeNormalizer},
                       "action": MinMaxNormalizer}
        flow_map / encoder / *_ema  - model state dicts (EMA weights used at inference)
    """
    bundle = {
        "config":     OmegaConf.to_container(config, resolve=True),
        "normalizer": {
            "obs_state": normalizer["obs"]["state"],  # CompositeNormalizer
            "action":    normalizer["action"],         # MinMaxNormalizer
        },
        "flow_map":       agent.flow_map.state_dict(),
        "encoder":        agent.encoder.state_dict(),
        "flow_map_ema":   agent.flow_map_ema.state_dict(),
        "encoder_ema":    agent.encoder_ema.state_dict(),
    }
    torch.save(bundle, path)


def make_ik_dataset(task_config, split: str = "train") -> IKParquetDataset:
    """Instantiate IKParquetDataset from Hydra task config."""
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


def compute_val_loss(
    agent: TrainingAgent,
    val_loader: torch.utils.data.DataLoader,
    config: Config,
    delta_t_scalar: float,
) -> float:
    """Compute mean validation loss with EMA model — no backward pass."""
    device = config.optimization.device
    agent.eval()
    val_losses = []

    with torch.no_grad():
        for val_batch in val_loader:
            v_obs = val_batch["obs"]["state"].to(device)
            v_obs = v_obs[:, : config.task.obs_steps, :]   # (B, To, obs_dim)
            v_act = val_batch["action"].to(device)
            v_act = v_act[:, : config.task.horizon, :]     # (B, Ta, act_dim)
            B = v_act.shape[0]
            v_delta_t = torch.full((B,), delta_t_scalar, device=device)

            # Direct loss_fn call: forward only, no optimizer step
            loss, _ = agent.loss_fn(
                config.optimization,
                agent.flow_map_ema,
                agent.encoder_ema,
                agent.interpolant,
                v_act,
                v_obs,
                v_delta_t,
            )
            val_losses.append(loss.item())

    agent.train()
    return float(np.mean(val_losses))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    config: Config,
    dataset: IKParquetDataset,
    val_dataset: IKParquetDataset,
    agent: TrainingAgent,
    run_logger: Logger,
    checkpoint_dir: Path = None,
    normalizer: dict = None,
):
    device = config.optimization.device

    # ---- DataLoaders ----
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,   # Required for CUDA graphs (static shapes)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.optimization.batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )
    loop_loader = loop_dataloader(dataloader)

    # ---- Schedulers ----
    lr_scheduler = CosineAnnealingLR(
        agent.optimizer,
        T_max=config.optimization.gradient_steps,
    )
    warmup_scheduler = WarmupAnnealingScheduler(
        max_steps=config.optimization.gradient_steps,
        warmup_ratio=config.optimization.warmup_ratio,
        rampup_ratio=config.optimization.rampup_ratio,
        min_value=config.optimization.min_value,
        max_value=config.optimization.max_value,
    )

    # ---- Checkpoint directory (Hydra changes CWD to outputs/{date}/{time}/) ----
    if checkpoint_dir is None:
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_dir = Path(os.getcwd()) / "checkpoints" / run_timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---- Checkpoint name suffix: _{network_type}[_{exp_tag}] ----
    network_type = getattr(config.network, "network_type", "")
    exp_tag      = getattr(config.log,     "exp_tag",      "") or ""
    ckpt_suffix  = f"_{network_type}" if network_type else ""
    if exp_tag:
        ckpt_suffix += f"_{exp_tag}"

    loguru.logger.info(
        f"Checkpoints will be saved to: {checkpoint_dir}  (suffix: '{ckpt_suffix}')"
    )

    best_val_loss = float("inf")
    info_list = []
    start_time = time.time()
    perf_times = {
        "data_load":  [],
        "preprocess": [],
        "update":     [],
        "total_step": [],
    }

    loguru.logger.info(
        f"Starting training | steps={config.optimization.gradient_steps} | "
        f"batch={config.optimization.batch_size} | device={device}"
    )

    pbar = tqdm(range(config.optimization.gradient_steps), dynamic_ncols=True)
    for step in pbar:
        with timed("total_step", perf_times):

            # ---- Load ----
            with timed("data_load", perf_times):
                batch = next(loop_loader)

            # ---- Preprocess ----
            with timed("preprocess", perf_times):
                obs = batch["obs"]["state"].to(device)
                obs = obs[:, : config.task.obs_steps, :]   # (B, To, obs_dim)
                act = batch["action"].to(device)
                act = act[:, : config.task.horizon, :]     # (B, Ta, act_dim)

            # ---- Update ----
            with timed("update", perf_times):
                delta_t_scalar = warmup_scheduler(step)
                B = act.shape[0]
                delta_t = torch.full((B,), delta_t_scalar, device=device)
                info = agent.update(act, obs, delta_t)
                lr_scheduler.step()

            for k, v in info.items():
                if isinstance(v, torch.Tensor):
                    info[k] = v.item()
            info_list.append(info)

        # ---- Train logging ----  category must be "train" or "eval"
        if (step + 1) % config.log.log_freq == 0:
            freq = config.log.log_freq
            metrics = {
                "step":       step,
                "total_time": time.time() - start_time,
                "lr":         lr_scheduler.get_last_lr()[0],
                "delta_t":    delta_t_scalar,
            }
            for key in info:
                try:
                    metrics[key] = float(np.nanmean([d[key] for d in info_list]))
                except Exception:
                    metrics[key] = float("nan")
            metrics["perf/data_load_ms"]  = np.mean(perf_times["data_load"][-freq:])  * 1000
            metrics["perf/preprocess_ms"] = np.mean(perf_times["preprocess"][-freq:]) * 1000
            metrics["perf/update_ms"]     = np.mean(perf_times["update"][-freq:])     * 1000
            metrics["perf/steps_per_sec"] = 1.0 / np.mean(perf_times["total_step"][-freq:])
            run_logger.log(metrics, category="train")
            # Update progress bar with latest loss
            loss_val = metrics.get("loss", float("nan"))
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{metrics['lr']:.2e}", step=step + 1)
            info_list = []

        # ---- Validation + Best-model checkpoint ----
        if (step + 1) % config.log.eval_freq == 0:
            val_loss = compute_val_loss(agent, val_loader, config, delta_t_scalar)
            run_logger.log({"step": step, "val_loss": val_loss}, category="eval")
            loguru.logger.info(f"[Step {step + 1:>7d}] val_loss = {val_loss:.5f}")
            pbar.set_postfix(val_loss=f"{val_loss:.5f}", step=step + 1)

            # Save best model (by val_loss)，保留 val_loss 最小的 top-3
            _TOP_K = 3
            best_name = f"step={step + 1:07d}-val_loss={val_loss:.6f}{ckpt_suffix}.pt"
            if normalizer is not None:
                save_mip_bundle(checkpoint_dir / best_name, agent, config, normalizer)
            else:
                agent.save(checkpoint_dir / best_name)
            loguru.logger.info(f"Saved checkpoint → {best_name}")

            # 找出所有同 suffix 的最优 checkpoint，按 val_loss 升序，超出 top-k 的删掉
            pattern = f"step=*-val_loss=*{ckpt_suffix}.pt"
            existing = list(checkpoint_dir.glob(pattern))
            def _parse_loss(p):
                try:
                    return float(p.stem.split("-val_loss=")[1].split(ckpt_suffix)[0])
                except Exception:
                    return float("inf")
            existing.sort(key=_parse_loss)
            for old in existing[_TOP_K:]:
                old.unlink()
                loguru.logger.info(f"Removed old checkpoint → {old.name}")

            best_val_loss = _parse_loss(existing[0]) if existing else val_loss

        # ---- Latest checkpoint (periodic) ----
        if (step + 1) % config.log.save_freq == 0:
            latest_name = f"latest{ckpt_suffix}.pt"
            if normalizer is not None:
                save_mip_bundle(checkpoint_dir / latest_name, agent, config, normalizer)
            else:
                agent.save(checkpoint_dir / latest_name)
            loguru.logger.info(f"Latest checkpoint saved at step {step + 1} → {latest_name}")

    # Final save
    final_name = (
        f"step={config.optimization.gradient_steps:07d}"
        f"-val_loss={best_val_loss:.6f}"
        f"{ckpt_suffix}-final.pt"
    )
    if normalizer is not None:
        save_mip_bundle(checkpoint_dir / final_name, agent, config, normalizer)
    else:
        agent.save(checkpoint_dir / final_name)
    loguru.logger.info(f"Training complete → {final_name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="main", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    config = cfg  # DictConfig supports dot-access; OmegaConf.to_object returns plain dict

    limit_threads(1)
    set_seed(config.optimization.seed)
    os.makedirs(config.log.log_dir, exist_ok=True)

    loguru.logger.info(f"Task:   {config.task.env_name}")
    loguru.logger.info(f"Device: {config.optimization.device}")

    loguru.logger.info("Loading IK datasets ...")
    dataset     = make_ik_dataset(config.task, split="train")
    val_dataset = make_ik_dataset(config.task, split="val")
    # IMPORTANT: val must use the SAME normalizer as train (fit on train data only).
    # Re-fitting on val episodes gives different statistics → model sees wrong-scale inputs.
    val_dataset._normalizer = dataset._normalizer
    loguru.logger.info(
        f"Train samples: {len(dataset):,} | Val samples: {len(val_dataset):,}"
    )

    # Auto-detect actual obs_dim from the first data sample (overrides yaml value)
    sample = dataset[0]
    actual_obs_dim = sample["obs"]["state"].shape[-1]  # (obs_steps, obs_dim) → last dim
    if actual_obs_dim != config.task.obs_dim:
        loguru.logger.warning(
            f"obs_dim mismatch: config={config.task.obs_dim}, actual data={actual_obs_dim}. "
            f"Overriding with data value."
        )
        config.task.obs_dim = actual_obs_dim

    agent      = TrainingAgent(config)
    run_logger = Logger(config)

    train(config, dataset, val_dataset, agent, run_logger, normalizer=dataset._normalizer)


if __name__ == "__main__":
    main()
