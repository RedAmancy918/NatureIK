import pandas as pd
import numpy as np

# p = "/home/user/Data/data/airbot/2026_0322_airbot_ALL_SplitNoRGB/data/chunk-000/episode_000000.parquet"
p = "ErrorData/error_data_euler/data/chunk-000/episode_000001.parquet"
df = pd.read_parquet(p)

print("==== columns ====")
for c in df.columns:
    print(c)

print("\n==== dataset length ====")
print(len(df))


def show_shape(col):
    x = df.iloc[0][col]
    arr = np.array(x)
    print(f"\n{col}")
    print("shape:", arr.shape)
    print("first values:", arr.reshape(-1)[:10])


# 基础字段检查
for col in [
    "observation.state",
    "action",
    "observation.state_quat",
    "action_quat"
]:
    if col in df.columns:
        show_shape(col)


print("\n============================")
print("Check joint relation")
print("============================")

for i in range(5):
    q = np.array(df.iloc[i]["observation.state"])
    a = np.array(df.iloc[i]["action"])
    diff = a - q

    print(f"\nrow {i}")
    print("q:", q)
    print("a:", a)
    print("a - q:", diff)
    print("mean |diff|:", np.mean(np.abs(diff)))


print("\n============================")
print("Check EEF relation")
print("============================")

if "observation.state_quat" in df.columns and "action_quat" in df.columns:
    for i in range(5):
        eef = np.array(df.iloc[i]["observation.state_quat"])
        eef_next = np.array(df.iloc[i]["action_quat"])

        diff = eef_next - eef

        print(f"\nrow {i}")
        print("eef:", eef)
        print("eef_next:", eef_next)
        print("diff:", diff)
        print("mean |diff|:", np.mean(np.abs(diff)))


print("\n============================")
print("Check time continuity")
print("============================")

for i in range(3):
    q_next = np.array(df.iloc[i]["action"])
    q_next_row = np.array(df.iloc[i+1]["observation.state"])

    print(f"\nrow {i} -> {i+1}")
    print("action[i]        :", q_next)
    print("observation[i+1] :", q_next_row)
    print("difference:", q_next_row - q_next)