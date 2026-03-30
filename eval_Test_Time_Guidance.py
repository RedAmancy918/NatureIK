"""
DiffusionIK 推理性能 Benchmark
================================
参考 vla_scripts 中 flowmatching 的多配置测试思路，针对 IKDiffuserPolicy 的
推理瓶颈进行系统性诊断和量化。

运行方式（无需 checkpoint，直接用随机权重测速）：
    python benchmark_diffusion_ik.py

主要测试维度：
    1. 分段计时 —— 找到单次推理的瓶颈在哪一步
    2. 推理步数扫参 —— num_inference_steps 对延迟的影响
    3. 调度器对比 —— DDPM vs DDIM（步数相同时的速度差异）
    4. 模型规模扫参 —— n_layer / n_embd 的影响
    5. torch.compile 效果
    6. 单臂 vs 双臂（ik_server.py 里 solve_arm 调用 2 次的实际代价）
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import time
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

try:
    from termcolor import cprint
except ImportError:
    def cprint(msg, *args, **kwargs):
        print(msg)

from tqdm import tqdm

# 把项目根目录加入路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from diffusion_policy.model.diffusion.ik_transformer import IKDiffuserModel
from diffusion_policy.model.common.normalizer import LinearNormalizer


# ==============================================================================
# 辅助函数
# ==============================================================================

def format_ms(t_sec: float) -> str:
    return f"{t_sec * 1000:.3f} ms"


def make_dummy_normalizer(device):
    """构造一个直通归一化器（scale=1, offset=0），不需要真实数据。"""
    norm = LinearNormalizer()
    # 手动填充 obs/state_quat 和 action 两个 key
    for key, dim in [("obs/state_quat", 7), ("action", 6)]:
        norm.params_dict[key] = {
            "scale": torch.ones(dim, device=device),
            "offset": torch.zeros(dim, device=device),
            "input_stats": {
                "min": torch.zeros(dim, device=device),
                "max": torch.ones(dim, device=device),
                "mean": torch.zeros(dim, device=device),
                "std": torch.ones(dim, device=device),
            },
        }
    return norm


def build_model_and_scheduler(
    n_embd: int,
    n_layer: int,
    n_head: int,
    num_train_timesteps: int = 100,
    scheduler_type: str = "ddpm",
    device: torch.device = torch.device("cuda"),
    dtype=torch.float32,
):
    model = IKDiffuserModel(
        joint_dim=6,
        ee_dim=7,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        dropout=0.0,  # 推理时关闭 dropout
    ).to(device, dtype=dtype).eval()

    if scheduler_type == "ddpm":
        scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
        )
    elif scheduler_type == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
        )
    else:
        raise ValueError(f"未知调度器类型: {scheduler_type}")

    return model, scheduler


def single_arm_inference(
    model: IKDiffuserModel,
    scheduler,
    num_inference_steps: int,
    ee_cond: torch.Tensor,   # (1, 1, 7)
    device: torch.device,
    dtype,
):
    """完整单臂推理，对应 IKDiffuserPolicy.predict_action() 的核心逻辑。"""
    B = ee_cond.shape[0]
    nq = torch.randn((B, 1, 6), device=device, dtype=dtype)
    scheduler.set_timesteps(num_inference_steps)
    for t in scheduler.timesteps:
        batched_t = torch.full((B,), t.item(), device=device, dtype=torch.long)
        pred = model(nq, batched_t, cond=ee_cond)
        nq = scheduler.step(pred, t, nq).prev_sample
    return nq


# ==============================================================================
# 测试 1：分段计时（找单次推理瓶颈）
# ==============================================================================

def test_stage_timing(device, dtype, warmup=10, repeat=200):
    cprint("\n" + "=" * 60, "yellow")
    cprint("  测试 1：分段计时（100 步 DDPM，单臂）", "yellow")
    cprint("=" * 60, "yellow")

    model, scheduler = build_model_and_scheduler(
        n_embd=256, n_layer=4, n_head=8,
        num_train_timesteps=100,
        scheduler_type="ddpm",
        device=device, dtype=dtype,
    )
    num_inference_steps = 100
    ee_cond = torch.randn(1, 1, 7, device=device, dtype=dtype)
    nq_init = torch.randn(1, 1, 6, device=device, dtype=dtype)
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            single_arm_inference(model, scheduler, num_inference_steps, ee_cond, device, dtype)
    torch.cuda.synchronize()

    # ---------- (A) 端到端总时间 ----------
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            single_arm_inference(model, scheduler, num_inference_steps, ee_cond, device, dtype)
    torch.cuda.synchronize()
    t_total = (time.perf_counter() - t0) / repeat

    # ---------- (B) 仅模型 forward × 100 步 ----------
    with torch.no_grad():
        nq = nq_init.clone()
        for _ in range(warmup):
            for t in timesteps:
                bt = torch.full((1,), t.item(), device=device, dtype=torch.long)
                _ = model(nq, bt, cond=ee_cond)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            nq = nq_init.clone()
            for t in timesteps:
                bt = torch.full((1,), t.item(), device=device, dtype=torch.long)
                _ = model(nq, bt, cond=ee_cond)
    torch.cuda.synchronize()
    t_model_only = (time.perf_counter() - t0) / repeat

    # ---------- (C) 仅 scheduler.step × 100 步（用零张量模拟） ----------
    dummy_pred = torch.zeros(1, 1, 6, device=device, dtype=dtype)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            nq = nq_init.clone()
            for t in timesteps:
                nq = scheduler.step(dummy_pred, t, nq).prev_sample
    t_scheduler_only = (time.perf_counter() - t0) / repeat

    # ---------- (D) 单步 forward（平均每步代价） ----------
    t = timesteps[50]  # 取中间步
    bt = torch.full((1,), t.item(), device=device, dtype=torch.long)
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(nq_init, bt, cond=ee_cond)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat * 10):
        with torch.no_grad():
            _ = model(nq_init, bt, cond=ee_cond)
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
        n_embd=256, n_layer=4, n_head=8,
        scheduler_type="ddpm", device=device, dtype=dtype,
    )
    ee_cond = torch.randn(1, 1, 7, device=device, dtype=dtype)

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
        )
        # warmup
        with torch.no_grad():
            for _ in range(warmup):
                single_arm_inference(model, scheduler, n_steps, ee_cond, device, dtype)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeat):
                single_arm_inference(model, scheduler, n_steps, ee_cond, device, dtype)
        torch.cuda.synchronize()
        avg_ms = (time.perf_counter() - t0) / repeat * 1000

        if n_steps == 100:
            baseline_ms = avg_ms
        ratio = f"{baseline_ms / avg_ms:.1f}x 快" if baseline_ms and n_steps < 100 else "baseline"
        cprint(f"  {n_steps:>6}  {avg_ms:>11.2f}ms  {ratio:>10}", "cyan")
        results.append({"steps": n_steps, "ms": avg_ms})

    return results


# ==============================================================================
# 测试 3：调度器对比（DDPM vs DDIM，同步数）
# ==============================================================================

def test_scheduler_comparison(device, dtype, warmup=5, repeat=100):
    cprint("\n" + "=" * 60, "yellow")
    cprint("  测试 3：DDPM vs DDIM（相同步数对比）", "yellow")
    cprint("=" * 60, "yellow")

    steps_list = [5, 10, 20, 50]
    ee_cond = torch.randn(1, 1, 7, device=device, dtype=dtype)

    cprint(f"\n  {'步数':>6}  {'DDPM(ms)':>12}  {'DDIM(ms)':>12}  {'DDIM加速':>10}", "white")
    cprint("  " + "-" * 46, "white")

    results = []
    for n_steps in steps_list:
        timings = {}
        for sched_type in ["ddpm", "ddim"]:
            model, scheduler = build_model_and_scheduler(
                n_embd=256, n_layer=4, n_head=8,
                scheduler_type=sched_type, device=device, dtype=dtype,
            )
            with torch.no_grad():
                for _ in range(warmup):
                    single_arm_inference(model, scheduler, n_steps, ee_cond, device, dtype)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(repeat):
                    single_arm_inference(model, scheduler, n_steps, ee_cond, device, dtype)
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
# 测试 4：模型规模扫参
# ==============================================================================

def test_model_size_sweep(device, dtype, n_steps=10, warmup=5, repeat=100):
    cprint("\n" + "=" * 60, "yellow")
    cprint(f"  测试 4：模型规模扫参（固定 {n_steps} 步 DDPM）", "yellow")
    cprint("=" * 60, "yellow")

    configs = list(itertools.product(
        [128, 256, 512],   # n_embd
        [2, 4, 6, 8],      # n_layer
    ))

    ee_cond = torch.randn(1, 1, 7, device=device, dtype=dtype)

    cprint(f"\n  {'n_embd':>8}  {'n_layer':>8}  {'参数量':>12}  {'延迟(ms)':>12}", "white")
    cprint("  " + "-" * 48, "white")

    results = []
    for n_embd, n_layer in configs:
        n_head = 8 if n_embd >= 256 else 4
        model, scheduler = build_model_and_scheduler(
            n_embd=n_embd, n_layer=n_layer, n_head=n_head,
            scheduler_type="ddpm", device=device, dtype=dtype,
        )
        total_params = sum(p.numel() for p in model.parameters())

        with torch.no_grad():
            for _ in range(warmup):
                single_arm_inference(model, scheduler, n_steps, ee_cond, device, dtype)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeat):
                single_arm_inference(model, scheduler, n_steps, ee_cond, device, dtype)
        torch.cuda.synchronize()
        avg_ms = (time.perf_counter() - t0) / repeat * 1000

        params_k = total_params / 1000
        cprint(
            f"  {n_embd:>8}  {n_layer:>8}  {params_k:>10.1f}K  {avg_ms:>11.2f}ms",
            "cyan",
        )
        results.append({"n_embd": n_embd, "n_layer": n_layer, "params_k": params_k, "ms": avg_ms})

    return results


# ==============================================================================
# 测试 5：torch.compile 效果
# ==============================================================================

def test_torch_compile(device, dtype, n_steps=10, warmup=15, repeat=100):
    cprint("\n" + "=" * 60, "yellow")
    cprint(f"  测试 5：torch.compile 加速（{n_steps} 步 DDPM）", "yellow")
    cprint("=" * 60, "yellow")

    ee_cond = torch.randn(1, 1, 7, device=device, dtype=dtype)

    timings = {}
    for mode in ["原始模型", "compile(reduce-overhead)", "compile(max-autotune)"]:
        model, scheduler = build_model_and_scheduler(
            n_embd=256, n_layer=4, n_head=8,
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

        # 较多 warmup（compile 需要 trace）
        cprint(f"  正在 warmup: {mode} ...", "white")
        with torch.no_grad():
            for _ in range(warmup):
                single_arm_inference(model, scheduler, n_steps, ee_cond, device, dtype)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeat):
                single_arm_inference(model, scheduler, n_steps, ee_cond, device, dtype)
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
# 测试 6：单臂 vs 双臂（ik_server.py 中 solve_arm 调用 2 次的实际代价）
# ==============================================================================

def test_single_vs_dual_arm(device, dtype, n_steps=10, warmup=5, repeat=100):
    cprint("\n" + "=" * 60, "yellow")
    cprint(f"  测试 6：单臂 vs 双臂端到端延迟（{n_steps} 步 DDPM）", "yellow")
    cprint("=" * 60, "yellow")

    model, scheduler = build_model_and_scheduler(
        n_embd=256, n_layer=4, n_head=8,
        scheduler_type="ddpm", device=device, dtype=dtype,
    )
    ee_l = torch.randn(1, 1, 7, device=device, dtype=dtype)
    ee_r = torch.randn(1, 1, 7, device=device, dtype=dtype)

    # 单臂
    with torch.no_grad():
        for _ in range(warmup):
            single_arm_inference(model, scheduler, n_steps, ee_l, device, dtype)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            single_arm_inference(model, scheduler, n_steps, ee_l, device, dtype)
    torch.cuda.synchronize()
    t_single = (time.perf_counter() - t0) / repeat * 1000

    # 双臂（顺序调用，和 ik_server.py 一致）
    with torch.no_grad():
        for _ in range(warmup):
            single_arm_inference(model, scheduler, n_steps, ee_l, device, dtype)
            single_arm_inference(model, scheduler, n_steps, ee_r, device, dtype)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            single_arm_inference(model, scheduler, n_steps, ee_l, device, dtype)
            single_arm_inference(model, scheduler, n_steps, ee_r, device, dtype)
    torch.cuda.synchronize()
    t_dual = (time.perf_counter() - t0) / repeat * 1000

    # 双臂并行（batch_size=2，同时处理左右臂）
    ee_both = torch.cat([ee_l, ee_r], dim=0)  # (2, 1, 7)
    with torch.no_grad():
        for _ in range(warmup):
            single_arm_inference(model, scheduler, n_steps, ee_both, device, dtype)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            single_arm_inference(model, scheduler, n_steps, ee_both, device, dtype)
    torch.cuda.synchronize()
    t_batch = (time.perf_counter() - t0) / repeat * 1000

    cprint(f"\n  单臂（B=1）                : {t_single:.2f} ms", "cyan")
    cprint(f"  双臂顺序（2 × 单臂）       : {t_dual:.2f} ms   (理论 2x = {t_single*2:.2f} ms)", "cyan")
    cprint(f"  双臂并行（B=2，单次推理）  : {t_batch:.2f} ms   → 节省 {(t_dual - t_batch)/t_dual*100:.1f}%", "green")

    return {"single_ms": t_single, "dual_sequential_ms": t_dual, "dual_batch_ms": t_batch}


# ==============================================================================
# 测试 7：torch.profiler 详细 kernel 分析（可选，输出 chrome trace）
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
        n_embd=256, n_layer=4, n_head=8,
        scheduler_type="ddpm", device=device, dtype=dtype,
    )
    ee_cond = torch.randn(1, 1, 7, device=device, dtype=dtype)

    # warmup
    with torch.no_grad():
        for _ in range(5):
            single_arm_inference(model, scheduler, n_steps, ee_cond, device, dtype)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            with record_function("single_arm_inference"):
                single_arm_inference(model, scheduler, n_steps, ee_cond, device, dtype)

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

    total_ms = stage_result["total_ms"]
    model_ms = stage_result["model_ms"]
    sched_ms = stage_result["scheduler_ms"]

    cprint(f"\n  当前端到端（100步，单臂）: {total_ms:.2f} ms", "white")

    if sched_ms / total_ms > 0.3:
        cprint("\n  ⚠  调度器开销占比较高 → 建议优先尝试 DDIM", "red")
    if model_ms / total_ms > 0.6:
        cprint("\n  ⚠  模型 forward 是主要瓶颈 → 建议 torch.compile 或减小模型", "red")

    # 最快步数配置
    best_steps = min(steps_result, key=lambda x: x["ms"])
    cprint(f"\n  最快步数配置: {best_steps['steps']} 步 → {best_steps['ms']:.2f} ms", "green")
    cprint(f"  vs 100步的加速比: {total_ms / best_steps['ms']:.1f}x", "green")

    cprint("\n  推荐优先级排序：", "yellow")
    cprint("    1. 减少 num_inference_steps（改 yaml → 10~20 步）", "yellow")
    cprint("    2. 切换到 DDIM 调度器（需重新训练或直接替换 scheduler）", "yellow")
    cprint("    3. 双臂改为 batch_size=2 并行推理（修改 ik_server.py）", "yellow")
    cprint("    4. torch.compile(mode='reduce-overhead')", "yellow")
    cprint("    5. 蒸馏到 1~5 步（Consistency Distillation）", "yellow")


# ==============================================================================
# 主入口
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # ik_server 里用的 float32

    cprint(f"\n{'='*60}", "green")
    cprint(f"  DiffusionIK 推理性能 Benchmark", "green")
    cprint(f"  设备: {device}  |  精度: {dtype}", "green")
    cprint(f"{'='*60}", "green")

    # ---- 运行各项测试 ----
    stage_result  = test_stage_timing(device, dtype)
    steps_result  = test_inference_steps_sweep(device, dtype)
    sched_result  = test_scheduler_comparison(device, dtype)
    _             = test_model_size_sweep(device, dtype, n_steps=10)
    _             = test_torch_compile(device, dtype, n_steps=10)
    _             = test_single_vs_dual_arm(device, dtype, n_steps=10)
    test_profiler(device, dtype, n_steps=10)

    # ---- 汇总报告 ----
    print_summary(device, dtype, stage_result, steps_result, sched_result)
