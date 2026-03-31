"""
DiffusionIK 推理性能 Benchmark（ResNet 版）
============================================
基于 benchmark_diffusion_ik.py 修改，将测试对象从 IKDiffuserModel（DiT）
替换为你实际使用的 ConditionalResNet1D + DiffusionResNetLowdimPolicy。

运行方式（无需 checkpoint，直接用随机权重测速）：
    python benchmark_diffusion_ik.py

测试维度与原版一致：
    1. 分段计时 —— 找到单次推理的瓶颈在哪一步
    2. 推理步数扫参 —— num_inference_steps 对延迟的影响
    3. 调度器对比 —— DDPM vs DDIM
    4. 模型规模扫参 —— hidden_dim / n_blocks 的影响
    5. torch.compile 效果
    6. 单臂 vs 双臂（顺序 vs B=2 并行）
    7. torch.profiler 详细 kernel 分析
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import time
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

try:
    from termcolor import cprint
except ImportError:
    def cprint(msg, *args, **kwargs):
        print(msg)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from diffusion_policy.model.diffusion.conditional_resnet1d import ConditionalResNet1D


# ==============================================================================
# 默认参数（对应 train_ik_resnet_DeltaJoint.yaml）
# ==============================================================================
OBS_DIM      = 20   # 单帧观测维度
N_OBS_STEPS  = 3    # 历史帧数
ACTION_DIM   = 6    # 动作维度
N_ACT_STEPS  = 1    # 推理时执行步数（pred_action_steps_only=true）

# global_cond_dim = n_obs_steps × obs_dim
GLOBAL_COND_DIM = N_OBS_STEPS * OBS_DIM   # 60


# ==============================================================================
# 辅助函数
# ==============================================================================

def format_ms(t_sec: float) -> str:
    return f"{t_sec * 1000:.3f} ms"


def build_model_and_scheduler(
    hidden_dim: int = 256,
    n_blocks: int = 6,
    global_cond_dim: int = GLOBAL_COND_DIM,
    num_train_timesteps: int = 100,
    scheduler_type: str = "ddpm",
    device: torch.device = torch.device("cuda"),
    dtype=torch.float32,
):
    model = ConditionalResNet1D(
        input_dim=ACTION_DIM,
        global_cond_dim=global_cond_dim,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
    ).to(device, dtype=dtype).eval()

    if scheduler_type == "ddpm":
        scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            prediction_type="epsilon",
        )
    elif scheduler_type == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
    else:
        raise ValueError(f"未知调度器类型: {scheduler_type}")

    return model, scheduler


def single_arm_inference(
    model: ConditionalResNet1D,
    scheduler,
    num_inference_steps: int,
    global_cond: torch.Tensor,   # (B, N_OBS_STEPS * OBS_DIM)
    device: torch.device,
    dtype,
):
    """
    完整单臂推理，对应 DiffusionResNetLowdimPolicy.conditional_sample() 的核心逻辑。
    global_cond: (B, 60) = 3帧 obs 展平
    """
    B = global_cond.shape[0]
    trajectory = torch.randn((B, N_ACT_STEPS, ACTION_DIM), device=device, dtype=dtype)
    scheduler.set_timesteps(num_inference_steps)
    for t in scheduler.timesteps:
        model_output = model(trajectory, t, global_cond=global_cond)
        trajectory = scheduler.step(model_output, t, trajectory).prev_sample
    return trajectory


# ==============================================================================
# 测试 1：分段计时（找单次推理瓶颈）
# ==============================================================================

def test_stage_timing(device, dtype, warmup=10, repeat=200):
    cprint("\n" + "=" * 60, "yellow")
    cprint("  测试 1：分段计时（100 步 DDPM，单臂）", "yellow")
    cprint("=" * 60, "yellow")

    model, scheduler = build_model_and_scheduler(
        hidden_dim=256, n_blocks=6,
        scheduler_type="ddpm",
        device=device, dtype=dtype,
    )
    num_inference_steps = 100
    gc = torch.randn(1, GLOBAL_COND_DIM, device=device, dtype=dtype)
    traj_init = torch.randn(1, N_ACT_STEPS, ACTION_DIM, device=device, dtype=dtype)
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            single_arm_inference(model, scheduler, num_inference_steps, gc, device, dtype)
    torch.cuda.synchronize()

    # ---------- (A) 端到端总时间 ----------
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            single_arm_inference(model, scheduler, num_inference_steps, gc, device, dtype)
    torch.cuda.synchronize()
    t_total = (time.perf_counter() - t0) / repeat

    # ---------- (B) 仅模型 forward × 100 步 ----------
    with torch.no_grad():
        for _ in range(warmup):
            for t in timesteps:
                _ = model(traj_init, t, global_cond=gc)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            for t in timesteps:
                _ = model(traj_init, t, global_cond=gc)
    torch.cuda.synchronize()
    t_model_only = (time.perf_counter() - t0) / repeat

    # ---------- (C) 仅 scheduler.step × 100 步 ----------
    dummy_pred = torch.zeros(1, N_ACT_STEPS, ACTION_DIM, device=device, dtype=dtype)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            traj = traj_init.clone()
            for t in timesteps:
                traj = scheduler.step(dummy_pred, t, traj).prev_sample
    t_scheduler_only = (time.perf_counter() - t0) / repeat

    # ---------- (D) 单步 forward（平均每步代价） ----------
    t_mid = timesteps[50]
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(traj_init, t_mid, global_cond=gc)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat * 10):
        with torch.no_grad():
            _ = model(traj_init, t_mid, global_cond=gc)
    torch.cuda.synchronize()
    t_single_step = (time.perf_counter() - t0) / (repeat * 10)

    cprint(f"\n  端到端推理（100步）      : {format_ms(t_total)}", "cyan")
    cprint(f"  纯模型 forward × 100    : {format_ms(t_model_only)}", "cyan")
    cprint(f"  纯 scheduler.step × 100 : {format_ms(t_scheduler_only)}", "cyan")
    cprint(f"  单步 forward（平均）     : {format_ms(t_single_step)}", "cyan")
    cprint(f"\n  模型占总时间比例         : {t_model_only / t_total * 100:.1f}%", "magenta")
    cprint(f"  调度器占总时间比例       : {t_scheduler_only / t_total * 100:.1f}%", "magenta")
    cprint(f"  其余开销（循环/同步等）  : {(t_total - t_model_only - t_scheduler_only) / t_total * 100:.1f}%", "magenta")

    return {
        "total_ms": t_total * 1000,
        "model_ms": t_model_only * 1000,
        "scheduler_ms": t_scheduler_only * 1000,
        "single_step_ms": t_single_step * 1000,
    }


# ==============================================================================
# 测试 2：推理步数扫参
# ==============================================================================

def test_inference_steps_sweep(device, dtype, warmup=5, repeat=100):
    cprint("\n" + "=" * 60, "yellow")
    cprint("  测试 2：推理步数扫参（DDPM）", "yellow")
    cprint("=" * 60, "yellow")

    steps_list = [1, 2, 5, 10, 20, 50, 100]
    model, _ = build_model_and_scheduler(
        hidden_dim=256, n_blocks=6,
        scheduler_type="ddpm", device=device, dtype=dtype,
    )
    gc = torch.randn(1, GLOBAL_COND_DIM, device=device, dtype=dtype)

    results = []
    cprint(f"\n  {'步数':>6}  {'延迟(ms)':>12}  {'vs 100步':>10}", "white")
    cprint("  " + "-" * 34, "white")

    baseline_ms = None
    for n_steps in steps_list:
        scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            prediction_type="epsilon",
        )
        with torch.no_grad():
            for _ in range(warmup):
                single_arm_inference(model, scheduler, n_steps, gc, device, dtype)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeat):
                single_arm_inference(model, scheduler, n_steps, gc, device, dtype)
        torch.cuda.synchronize()
        avg_ms = (time.perf_counter() - t0) / repeat * 1000

        if n_steps == 100:
            baseline_ms = avg_ms
        ratio = f"{baseline_ms / avg_ms:.1f}x 快" if baseline_ms and n_steps < 100 else "baseline"
        cprint(f"  {n_steps:>6}  {avg_ms:>11.2f}ms  {ratio:>10}", "cyan")
        results.append({"steps": n_steps, "ms": avg_ms})

    return results


# ==============================================================================
# 测试 3：调度器对比（DDPM vs DDIM）
# ==============================================================================

def test_scheduler_comparison(device, dtype, warmup=5, repeat=100):
    cprint("\n" + "=" * 60, "yellow")
    cprint("  测试 3：DDPM vs DDIM（相同步数对比）", "yellow")
    cprint("=" * 60, "yellow")

    steps_list = [5, 10, 20, 50]
    gc = torch.randn(1, GLOBAL_COND_DIM, device=device, dtype=dtype)

    cprint(f"\n  {'步数':>6}  {'DDPM(ms)':>12}  {'DDIM(ms)':>12}  {'DDIM加速':>10}", "white")
    cprint("  " + "-" * 46, "white")

    results = []
    for n_steps in steps_list:
        timings = {}
        for sched_type in ["ddpm", "ddim"]:
            model, scheduler = build_model_and_scheduler(
                hidden_dim=256, n_blocks=6,
                scheduler_type=sched_type, device=device, dtype=dtype,
            )
            with torch.no_grad():
                for _ in range(warmup):
                    single_arm_inference(model, scheduler, n_steps, gc, device, dtype)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(repeat):
                    single_arm_inference(model, scheduler, n_steps, gc, device, dtype)
            torch.cuda.synchronize()
            timings[sched_type] = (time.perf_counter() - t0) / repeat * 1000

        speedup = timings["ddpm"] / timings["ddim"]
        cprint(
            f"  {n_steps:>6}  {timings['ddpm']:>11.2f}ms  {timings['ddim']:>11.2f}ms  {speedup:>9.2f}x",
            "cyan",
        )
        results.append({"steps": n_steps, "ddpm_ms": timings["ddpm"], "ddim_ms": timings["ddim"]})

    return results


# ==============================================================================
# 测试 4：模型规模扫参（hidden_dim / n_blocks）
# ==============================================================================

def test_model_size_sweep(device, dtype, n_steps=10, warmup=5, repeat=100):
    cprint("\n" + "=" * 60, "yellow")
    cprint(f"  测试 4：模型规模扫参（固定 {n_steps} 步 DDPM）", "yellow")
    cprint("=" * 60, "yellow")

    configs = list(itertools.product(
        [128, 256, 512],   # hidden_dim
        [2, 4, 6, 8],      # n_blocks
    ))

    gc = torch.randn(1, GLOBAL_COND_DIM, device=device, dtype=dtype)

    cprint(f"\n  {'hidden_dim':>12}  {'n_blocks':>8}  {'参数量':>12}  {'延迟(ms)':>12}", "white")
    cprint("  " + "-" * 52, "white")

    results = []
    for hidden_dim, n_blocks in configs:
        model, scheduler = build_model_and_scheduler(
            hidden_dim=hidden_dim, n_blocks=n_blocks,
            scheduler_type="ddpm", device=device, dtype=dtype,
        )
        total_params = sum(p.numel() for p in model.parameters())

        with torch.no_grad():
            for _ in range(warmup):
                single_arm_inference(model, scheduler, n_steps, gc, device, dtype)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeat):
                single_arm_inference(model, scheduler, n_steps, gc, device, dtype)
        torch.cuda.synchronize()
        avg_ms = (time.perf_counter() - t0) / repeat * 1000

        params_k = total_params / 1000
        cprint(
            f"  {hidden_dim:>12}  {n_blocks:>8}  {params_k:>10.1f}K  {avg_ms:>11.2f}ms",
            "cyan",
        )
        results.append({"hidden_dim": hidden_dim, "n_blocks": n_blocks, "params_k": params_k, "ms": avg_ms})

    return results


# ==============================================================================
# 测试 5：torch.compile 效果
# ==============================================================================

def test_torch_compile(device, dtype, n_steps=10, warmup=15, repeat=100):
    cprint("\n" + "=" * 60, "yellow")
    cprint(f"  测试 5：torch.compile 加速（{n_steps} 步 DDPM）", "yellow")
    cprint("=" * 60, "yellow")

    gc = torch.randn(1, GLOBAL_COND_DIM, device=device, dtype=dtype)

    timings = {}
    for mode in ["原始模型", "compile(reduce-overhead)", "compile(max-autotune)"]:
        model, scheduler = build_model_and_scheduler(
            hidden_dim=256, n_blocks=6,
            scheduler_type="ddpm", device=device, dtype=dtype,
        )
        if mode == "compile(reduce-overhead)":
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                cprint(f"  torch.compile 失败: {e}", "red")
                continue
        elif mode == "compile(max-autotune)":
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception as e:
                cprint(f"  torch.compile 失败: {e}", "red")
                continue

        cprint(f"  正在 warmup: {mode} ...", "white")
        with torch.no_grad():
            for _ in range(warmup):
                single_arm_inference(model, scheduler, n_steps, gc, device, dtype)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeat):
                single_arm_inference(model, scheduler, n_steps, gc, device, dtype)
        torch.cuda.synchronize()
        avg_ms = (time.perf_counter() - t0) / repeat * 1000
        timings[mode] = avg_ms

    baseline = timings.get("原始模型", 1.0)
    cprint(f"\n  {'模式':<30}  {'延迟(ms)':>12}  {'加速比':>8}", "white")
    cprint("  " + "-" * 56, "white")
    for mode, ms in timings.items():
        speedup = baseline / ms
        cprint(f"  {mode:<30}  {ms:>11.2f}ms  {speedup:>7.2f}x", "cyan")

    return timings


# ==============================================================================
# 测试 6：单臂 vs 双臂
# ==============================================================================

def test_single_vs_dual_arm(device, dtype, n_steps=10, warmup=5, repeat=100):
    cprint("\n" + "=" * 60, "yellow")
    cprint(f"  测试 6：单臂 vs 双臂端到端延迟（{n_steps} 步 DDPM）", "yellow")
    cprint("=" * 60, "yellow")

    model, scheduler = build_model_and_scheduler(
        hidden_dim=256, n_blocks=6,
        scheduler_type="ddpm", device=device, dtype=dtype,
    )
    gc_l = torch.randn(1, GLOBAL_COND_DIM, device=device, dtype=dtype)
    gc_r = torch.randn(1, GLOBAL_COND_DIM, device=device, dtype=dtype)
    gc_both = torch.cat([gc_l, gc_r], dim=0)  # (2, 60)

    # 单臂
    with torch.no_grad():
        for _ in range(warmup):
            single_arm_inference(model, scheduler, n_steps, gc_l, device, dtype)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            single_arm_inference(model, scheduler, n_steps, gc_l, device, dtype)
    torch.cuda.synchronize()
    t_single = (time.perf_counter() - t0) / repeat * 1000

    # 双臂顺序
    with torch.no_grad():
        for _ in range(warmup):
            single_arm_inference(model, scheduler, n_steps, gc_l, device, dtype)
            single_arm_inference(model, scheduler, n_steps, gc_r, device, dtype)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            single_arm_inference(model, scheduler, n_steps, gc_l, device, dtype)
            single_arm_inference(model, scheduler, n_steps, gc_r, device, dtype)
    torch.cuda.synchronize()
    t_dual = (time.perf_counter() - t0) / repeat * 1000

    # 双臂并行 B=2
    with torch.no_grad():
        for _ in range(warmup):
            single_arm_inference(model, scheduler, n_steps, gc_both, device, dtype)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            single_arm_inference(model, scheduler, n_steps, gc_both, device, dtype)
    torch.cuda.synchronize()
    t_batch = (time.perf_counter() - t0) / repeat * 1000

    saving = (t_dual - t_batch) / t_dual * 100
    cprint(f"\n  单臂（B=1）                : {t_single:.2f} ms", "cyan")
    cprint(f"  双臂顺序（2 × 单臂）       : {t_dual:.2f} ms   (理论 2x = {t_single*2:.2f} ms)", "cyan")
    cprint(f"  双臂并行（B=2，单次推理）  : {t_batch:.2f} ms   → 节省 {saving:.1f}%", "green")

    if saving > 5:
        cprint("  ✅ 并行有效，建议保留", "green")
    elif saving > 0:
        cprint("  ⚠️  收益较小，保留也无害", "yellow")
    else:
        cprint("  ❌ 无收益，建议回滚", "red")

    return {"single_ms": t_single, "dual_sequential_ms": t_dual, "dual_batch_ms": t_batch}


# ==============================================================================
# 测试 7：torch.profiler 详细 kernel 分析
# ==============================================================================

def test_profiler(device, dtype, n_steps=10, save_trace=True):
    cprint("\n" + "=" * 60, "yellow")
    cprint(f"  测试 7：torch.profiler 详细分析（{n_steps} 步 DDPM）", "yellow")
    cprint("=" * 60, "yellow")

    try:
        from torch.profiler import profile, record_function, ProfilerActivity
    except ImportError:
        cprint("  torch.profiler 不可用，跳过", "red")
        return

    model, scheduler = build_model_and_scheduler(
        hidden_dim=256, n_blocks=6,
        scheduler_type="ddpm", device=device, dtype=dtype,
    )
    gc = torch.randn(1, GLOBAL_COND_DIM, device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(5):
            single_arm_inference(model, scheduler, n_steps, gc, device, dtype)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        with torch.no_grad():
            with record_function("single_arm_inference"):
                single_arm_inference(model, scheduler, n_steps, gc, device, dtype)

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    cprint(f"\n  Top 15 耗时算子（按 {sort_key} 排序）：", "white")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

    if save_trace:
        trace_path = "benchmark_ik_trace.json"
        prof.export_chrome_trace(trace_path)
        cprint(f"\n  Chrome trace 已保存: {trace_path}", "green")
        cprint("  用 chrome://tracing 或 perfetto.dev 打开可视化", "green")


# ==============================================================================
# 汇总报告
# ==============================================================================

def print_summary(device, dtype, stage_result, steps_result, sched_result):
    cprint("\n" + "=" * 60, "green")
    cprint("  汇总：瓶颈诊断 & 优化建议", "green")
    cprint("=" * 60, "green")

    total_ms  = stage_result["total_ms"]
    model_ms  = stage_result["model_ms"]
    sched_ms  = stage_result["scheduler_ms"]

    cprint(f"\n  当前端到端（100步，单臂）: {total_ms:.2f} ms", "white")

    if sched_ms / total_ms > 0.3:
        cprint("\n  ⚠  调度器开销占比较高 → 建议优先尝试 DDIM", "red")
    if model_ms / total_ms > 0.6:
        cprint("\n  ⚠  模型 forward 是主要瓶颈 → 建议 torch.compile 或减小模型", "red")

    best_steps = min(steps_result, key=lambda x: x["ms"])
    cprint(f"\n  最快步数配置: {best_steps['steps']} 步 → {best_steps['ms']:.2f} ms", "green")
    cprint(f"  vs 100步的加速比: {total_ms / best_steps['ms']:.1f}x", "green")

    cprint("\n  推荐优先级排序：", "yellow")
    cprint("    1. 减少 num_inference_steps（yaml 改为 10~20 步）", "yellow")
    cprint("    2. 切换到 DDIM 调度器", "yellow")
    cprint("    3. 双臂改为 batch_size=2 并行推理（修改 ik_server.py）", "yellow")
    cprint("    4. torch.compile(mode='reduce-overhead')", "yellow")
    cprint("    5. 缩小 hidden_dim / n_blocks（需重训）", "yellow")


# ==============================================================================
# 主入口
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32

    cprint(f"\n{'='*60}", "green")
    cprint(f"  DiffusionIK 推理性能 Benchmark（ResNet 版）", "green")
    cprint(f"  设备: {device}  |  精度: {dtype}", "green")
    cprint(f"  模型: ConditionalResNet1D  hidden=256  blocks=6", "green")
    cprint(f"  输入: obs({GLOBAL_COND_DIM}D) → action({ACTION_DIM}D)", "green")
    cprint(f"{'='*60}", "green")

    stage_result = test_stage_timing(device, dtype)
    steps_result = test_inference_steps_sweep(device, dtype)
    sched_result = test_scheduler_comparison(device, dtype)
    _            = test_model_size_sweep(device, dtype, n_steps=10)
    _            = test_torch_compile(device, dtype, n_steps=10)
    _            = test_single_vs_dual_arm(device, dtype, n_steps=10)
    test_profiler(device, dtype, n_steps=10)

    print_summary(device, dtype, stage_result, steps_result, sched_result)
