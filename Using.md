# 训练启动方法
* 如果数据集当中都是panda的机械臂
``dataset = ParquetLowdimDataset(
    data_dir="/path/to/panda_dataset",
    horizon=16,
    obs_key="observation.state_quat",
    action_key="action",
    use_robot_feature=True,
    robot_name="panda",
    max_joints=16,
)``
# 配置
目前以下几种配置
1. 无时序的unet
2. 带有t与t+1时刻的eef对应DeltaJoint
3. 在2的基础上加上MLP
4. 使用机器人urdf的结构编码 

task config ik_xxx.yaml train_ikxxxx.yaml
dataset parquet_DeltaJoint_dataset.py parquet_lowdim_dataset.py 
用于机器人结构编码：robot_feature_utils.py robot_specs.py /urdf_data
policy diffusion_unet_lowdim_mlp_policy.py
eval eval_ik

2026.3.23新增了一个对比resnet的结构编码的对比模型。
2026.3.26 新增 IK Diffusion Transformer (DiT) 方案（cross-attention Transformer 去噪网络）。
  - 模型：IKDiffuserModel（ik_transformer.py）+ IKDiffuserPolicy（ik_diffuser_policy.py）
  - 数据集：IKParquetDataset（ik_parquet_dataset.py），obs=末端位姿(7D)，action=绝对关节角(6D)
  - 配置：train_ik_dit.yaml + task/ik_parquet_dit.yaml
  - 训练：python train.py --config-name=train_ik_dit task.dataset.data_dir=.../chunk-000
如果不使用机器人结构那么

  use_robot_feature: true
  robot_name: airbot_single_arm
  max_joints: 16
需要修改为false

# 四元数也有双覆盖的问题，可能要改成6d表示。

## LeRobot 多 chunk 数据（自动读 chunk-000 … chunk-00N）
数据集目录形如 `.../data/chunk-000/episode_*.parquet`、`chunk-001/...` 时：
- `data_dir` 设为 **`data` 父目录**（即包含所有 `chunk-*` 的那一层）
- 在 task 的 `dataset` 里设 **`scan_chunk_subdirs: true`**

命令行覆盖示例：
```bash
python -m diffusion_policy.workspace.train_diffusion_resnet_lowdim_workspace \
  --config-name=train_ik_resnet_DeltaJoint \
  dataset.data_dir=/home/user/Data/data/airbot/2026_0322_airbot_ALL_SplitNoRGB/data \
  dataset.scan_chunk_subdirs=true
```
实现见 `diffusion_policy/dataset/parquet_path_utils.py`（不依赖 lerobot 包，仅匹配路径布局）。

```bash eval脚本
python eval_ik.py   -c /home/user/yjh/diffusion_policy/outputs/2026-03-26/00-46-32/checkpoints/epoch=0190-val_loss=0.006763.ckpt   -o /home/user/yjh/diffusion_policy/outputs/eval_results_0190   -d /home/user/Data/data/airbot/2026_0322_airbot_ALL_SplitNoRGB/data/chunk-000   --device cuda:0
```