"""Convert old-format MIP checkpoint to self-contained inference bundle.

Old format (logs/models/model_*.pt):
    {flow_map, encoder, encoder_ema, flow_map_ema, optimizer}

New format (inference bundle):
    {flow_map, encoder, encoder_ema, flow_map_ema, config, normalizer}

Usage:
    python examples/convert_checkpoint.py \
        --ckpt    logs/models/model_final.pt \
        --output  logs/models/model_final_bundle.pt \
        task=ik_delta_joint network=sudeepdit
"""

import os
import sys
from pathlib import Path

import hydra
import loguru
import torch
from omegaconf import OmegaConf


@hydra.main(config_path="configs", config_name="main", version_base=None)
def main(cfg):
    import argparse

    # ── Parse extra args (hydra consumes its own, we need ours) ──────────────
    # Hydra strips unknown args, so we read from sys.argv manually before hydra
    # This is a workaround: set them as env vars before calling this script.
    ckpt_path  = os.environ.get("CONVERT_CKPT", "logs/models/model_final.pt")
    out_path   = os.environ.get("CONVERT_OUT",  "logs/models/model_bundle.pt")

    OmegaConf.resolve(cfg)
    config = cfg
    config.optimization.use_compile   = False
    config.optimization.use_cudagraphs = False

    # ── Load normalizer from dataset ──────────────────────────────────────────
    # We need to rebuild the normalizer from the training data.
    loguru.logger.info("Loading training dataset to rebuild normalizer ...")
    from mip.datasets.ik_parquet_dataset import IKParquetDataset

    data_dir = os.path.expanduser(config.task.dataset_path)
    mode = "delta_joint" if config.task.env_name == "ik_delta_joint" else "absolute"

    dataset = IKParquetDataset(
        data_dir=data_dir,
        mode=mode,
        horizon=config.task.horizon,
        obs_steps=config.task.obs_steps,
        val_ratio=getattr(config.task, "val_dataset_percentage", 0.1),
        split="train",
    )
    loguru.logger.info(f"Dataset loaded: {len(dataset):,} samples")

    # Auto-detect actual obs_dim
    sample = dataset[0]
    actual_obs_dim = sample["obs"]["state"].shape[-1]
    if actual_obs_dim != config.task.obs_dim:
        loguru.logger.warning(
            f"obs_dim mismatch: config={config.task.obs_dim}, data={actual_obs_dim}. Overriding."
        )
        config.task.obs_dim = actual_obs_dim

    normalizer = dataset.get_normalizer()

    # ── Reconstruct agent & load old weights ──────────────────────────────────
    from mip.agent import TrainingAgent

    loguru.logger.info(f"Loading checkpoint: {ckpt_path}")
    agent = TrainingAgent(config)
    agent.load(ckpt_path, load_optimizer=False)
    agent.eval()

    # ── Save inference bundle ─────────────────────────────────────────────────
    bundle = {
        "flow_map":     agent.flow_map.state_dict(),
        "encoder":      agent.encoder.state_dict(),
        "flow_map_ema": agent.flow_map_ema.state_dict(),
        "encoder_ema":  agent.encoder_ema.state_dict(),
        "config":       OmegaConf.to_container(config, resolve=True),
        "normalizer": {
            "obs_state": normalizer["obs"]["state"],
            "action":    normalizer["action"],
        },
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    loguru.logger.info(f"Inference bundle saved → {out}")
    loguru.logger.info("Done. Use this bundle with MIPIKSolver in ik_server.py")


if __name__ == "__main__":
    main()
