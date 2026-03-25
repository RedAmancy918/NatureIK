from diffusion_policy.dataset.parquet_DeltaJoint_dataset import ParquetDeltaJointDataset

dataset = ParquetDeltaJointDataset(
    data_dir="/project/data/airbot_tok4_split_IK/data/chunk-000",
    horizon=20,
    use_robot_feature=False,
)

item = dataset[0]
print("obs shape:", item["obs"].shape)       # 预期 [20, 20]
print("action shape:", item["action"].shape) # 预期 [20, 6]