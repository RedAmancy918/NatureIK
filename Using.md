# diffusion_policy 使用说明（IK）

本文档整理了当前项目内 IK 相关训练、评估、服务部署与加速测试的常用流程。

## 1. 主要配置与文件对应关系

- **ResNet IK（DeltaJoint）**
  - 训练配置：`diffusion_policy/config/train_ik_resnet_DeltaJoint.yaml`
  - 任务配置：`diffusion_policy/config/task/ik_parquet_DeltaJoint.yaml`
  - Policy：`diffusion_policy/policy/diffusion_resnet_lowdim_policy.py`
  - Workspace：`diffusion_policy/workspace/train_diffusion_resnet_lowdim_workspace.py`

- **ResNet IK + 结构编码（URDF feature）**
  - 训练配置：`diffusion_policy/config/train_ik_resnet_DeltaJoint_stru.yaml`
  - 任务配置：`diffusion_policy/config/task/ik_parquet_DeltaJoint_stru.yaml`
  - 结构特征：`diffusion_policy/dataset/robot_feature_utils.py`
  - 机器人规格：`diffusion_policy/dataset/robot_specs.py`
  - URDF 目录：`diffusion_policy/urdf_data`

- **UNet IK（DeltaJoint）**
  - 训练配置：`diffusion_policy/config/train_ik_unet_DeltaJoint.yaml`
  - 任务配置：`diffusion_policy/config/task/ik_parquet_DeltaJoint.yaml`
  - Policy：`diffusion_policy/policy/diffusion_unet_lowdim_policy.py`
  - Workspace：`diffusion_policy/workspace/train_diffusion_unet_lowdim_workspace.py`

- **UNet IK + 结构编码（URDF feature）**
  - 训练配置：`diffusion_policy/config/train_ik_unet_lowdim_stru.yaml`
  - 任务配置：`diffusion_policy/config/task/ik_parquet_lowdim_stru.yaml`
  - Policy：`diffusion_policy/policy/diffusion_unet_lowdim_policy.py`
  - Workspace：`diffusion_policy/workspace/train_diffusion_unet_lowdim_workspace.py`

- **IK Diffusion Transformer（DiT）**
  - 训练配置：`diffusion_policy/config/train_ik_dit.yaml`
  - 任务配置：`diffusion_policy/config/task/ik_parquet_dit.yaml`
  - 模型：`diffusion_policy/model/diffusion/ik_transformer.py`
  - Policy：`diffusion_policy/policy/ik_diffuser_policy.py`
  - Workspace：`diffusion_policy/workspace/train_ik_workspace.py`
  - 数据集：`diffusion_policy/dataset/ik_parquet_dataset.py`

- **Flow Matching ResNet IK（DeltaJoint）**
  - 训练配置：`diffusion_policy/config/train_ik_flow_resnet_DeltaJoint.yaml`
  - 任务配置：`diffusion_policy/config/task/ik_parquet_DeltaJoint.yaml`
  - Policy：`diffusion_policy/policy/flow_resnet_lowdim_policy.py`
  - Workspace：`diffusion_policy/workspace/train_flow_resnet_workspace.py`
  - 说明：保留现有 ResNet + global_cond 结构，训练目标改为速度场，推理改为固定步数 ODE 积分
## 2. 数据集组织方式

### 2.1 单文件夹训练（只读一个 chunk）

当数据在 `.../chunk-000/episode_*.parquet`：

```bash
python train.py --config-name=train_ik_resnet_DeltaJoint \
  task.dataset.data_dir=/home/user/Data/.../data/chunk-000 \
  task.dataset.scan_chunk_subdirs=false
```

### 2.2 多 chunk 训练（自动读 chunk-000...chunk-00N）

当目录为 `.../data/chunk-000`、`.../data/chunk-001`、...：

- `task.dataset.data_dir` 设为包含所有 `chunk-*` 的父目录（`.../data`）
- `task.dataset.scan_chunk_subdirs=true`

```bash
python train.py --config-name=train_ik_resnet_DeltaJoint \
  task.dataset.data_dir=/home/user/Data/.../data \
  task.dataset.scan_chunk_subdirs=true
```

多 chunk 路径收集实现见：`diffusion_policy/dataset/parquet_path_utils.py`。

## 3. 训练命令

### 3.1 ResNet IK（无结构编码）

```bash
python train.py --config-name=train_ik_resnet_DeltaJoint \
  task.dataset.data_dir=/home/user/Data/.../data/chunk-000
```

### 3.2 ResNet IK（带结构编码）

```bash
python train.py --config-name=train_ik_resnet_DeltaJoint_stru \
  task.dataset.data_dir=/home/user/Data/.../data/chunk-000
```

> 若不使用结构编码，请确保 task 配置中 `use_robot_feature: false`。

### 3.3 UNet IK（无结构编码）

```bash
python train.py --config-name=train_ik_unet_DeltaJoint \
  task.dataset.data_dir=/home/user/Data/.../data/chunk-000
```

### 3.4 UNet IK（带结构编码）

```bash
python train.py --config-name=train_ik_unet_lowdim_stru \
  task.dataset.data_dir=/home/user/Data/.../data/chunk-000
```

### 3.5 IK DiT

```bash
python train.py --config-name=train_ik_dit \
  task.dataset.data_dir=/home/user/Data/.../data/chunk-000
```

## 4. 推理步数与速度

- `policy.num_inference_steps`：推理扩散步数，步数越小通常越快。
- 当该值为 `null` 时，默认使用 `noise_scheduler.num_train_timesteps`。

例如（`train_ik_resnet_DeltaJoint_stru.yaml`）：

```yaml
policy:
  noise_scheduler:
    num_train_timesteps: 100
```

常见实践：
- 调速优先改 `policy.num_inference_steps`（例如 100 -> 20 或 10）
- 改模型结构（`hidden_dim`、`n_blocks`）需要重新训练
- 参考：
    * model:
    * hidden_dim: 256   # 每层宽度，↑ 更强但更慢，常用 128/256/512
    * n_blocks: 6       # ResNet 层数，↑ 更深但更慢，常用 4/6/8
    * input_dim: 6      # 动作维度，一般不动
    * global_cond_dim:  # = n_obs_steps × obs_dim，由上面自动算，不用手改

## 5. 评估命令

```bash
python eval_ik.py \
  -c /home/user/yjh/diffusion_policy/outputs/2026-03-26/00-46-32/checkpoints/epoch=0190-val_loss=0.006763.ckpt \
  -o /home/user/yjh/diffusion_policy/outputs/eval_results_0190 \
  -d /home/user/Data/.../data/chunk-000 \
  --device cuda:0
```

## 6. IK 服务启动

`ik_server.py` 通过 `--ckpt` 指定权重路径；未传时使用代码中的默认路径。

```bash
python ik_server.py \
  --ckpt /home/user/yjh/diffusion_policy/outputs/.../checkpoints/latest.ckpt \
  --robot airbot_single_arm \
  --mode 1
```

## 7. 性能测试（Test-Time Guidance）

```bash
python eval_Test_Time_Guidance.py
```

该脚本用于测试：
- `num_inference_steps` 对延迟的影响
- 顺序双臂 vs 批并行双臂的速度差异

## 8. 版本与注意事项

- 当前环境中如需 `torch.compile` 等算子级优化，通常需要更高版本 PyTorch。
- 四元数存在双覆盖问题（`q` 与 `-q` 等价），已在代码中做了半球对齐；如后续追求更稳训练，可考虑 6D 旋转表示。

## 9. 变更记录

- 详细开发修改记录见：`experiment_logs/dev_log_20260329.md`。

