# -*- coding: utf-8 -*-
# https://chatgpt.com/share/68c798e3-ff68-8005-b4cc-af36d0699836
# https://chatgpt.com/s/t_68c8236029c48191b795565cb20b4cdc
# https://chatgpt.com/s/t_68c7bd06e2788191824c62455e7bc64a
import argparse
import os
import numpy as np
from typing import List, Dict, Any
from model import ALGORITHM_REGISTRY
from viz import save_scatter_snapshot
from metrics import compute_sse

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

def gen_stream(
    n_points: int,
    low: float = -100.0,
    high: float = 100.0,
    seed: int = 42,
    k: int = 6,
    sigma_range: tuple = (5.0, 12.0),
    outlier_rate: float = 0.02,
    drift_per_step: float = 0.0,
    margin: float = 20.0,
    return_label: bool = False,
):
    """
    生成“更有聚集性”的流式二维数据：k 个高斯簇 + 少量离群点（可选中心缓慢漂移）。
    - 输出维度固定为 2D，范围约束在 [low, high]^2（clip）。
    - 与原框架接口兼容：默认 yield 点坐标；若 return_label=True 则 yield (点, 真标签)。
      真标签 ∈ {0..k-1}，离群点标签为 k。
    """
    rng = np.random.default_rng(seed)
    # 在边界内留出 margin，避免中心靠边
    centers = rng.uniform(low + margin, high - margin, size=(k, 2))
    # 每簇方差不同，增强“明显的中心”与聚集感
    sigmas = rng.uniform(sigma_range[0], sigma_range[1], size=(k,))
    # 不同簇大小（权重）不同
    weights = rng.dirichlet(np.ones(k))

    for _ in range(n_points):
        if rng.random() < outlier_rate:
            pt = rng.uniform(low, high, size=(2,))
            label = k  # 离群点
        else:
            j = int(rng.choice(k, p=weights))
            pt = centers[j] + rng.normal(0.0, sigmas[j], size=(2,))
            label = j

        # 可选：让中心缓慢漂移，模拟概念漂移场景
        if drift_per_step > 0.0:
            centers += rng.normal(0.0, drift_per_step, size=centers.shape)
            centers = np.clip(centers, low + margin, high - margin)

        # 保证点不出界
        pt = np.clip(pt, low, high)
        yield (pt, label) if return_label else pt

def build_algorithms(names: List[str]):
    algos = {}
    for name in names:
        if name not in ALGORITHM_REGISTRY:
            raise ValueError(f"Algorithm '{name}' not found. Available: {list(ALGORITHM_REGISTRY.keys())}")
        algos[name] = ALGORITHM_REGISTRY[name](dim=2, name=name)
    return algos

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algos", type=str, default="floc,online_kmeans,dp_means,demo_random,demo_grid,minibatch_kmeans,denstream,clustream,streamkmpp,",
                        help="Comma-separated algorithm names from model.ALGORITHM_REGISTRY")
    parser.add_argument("--n_points", type=int, default=10000)
    parser.add_argument("--snapshot_every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low", type=float, default=-100.0)
    parser.add_argument("--high", type=float, default=100.0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    algo_names = [s.strip() for s in args.algos.split(",") if s.strip()]
    algos = build_algorithms(algo_names)

    labels: Dict[str, List[int]] = {name: [] for name in algo_names}
    points: List[np.ndarray] = []
    
    for step, x in enumerate(gen_stream(args.n_points, args.low, args.high, seed=args.seed), start=1):
        points.append(x)
        for name, algo in algos.items():
            label = algo.partial_fit(x)
            labels[name].append(int(label))

        if (step % args.snapshot_every == 0) or (step == args.n_points):
            P = np.vstack(points)
            for name, algo in algos.items():
                lbl = np.array(labels[name], dtype=int)
                state = getattr(algo, "get_state", lambda: {})() or {}
                centroids = state.get("centroids", None)
                save_scatter_snapshot(P, lbl, centroids,
                                      out_path=os.path.join(OUTPUT_DIR, f"{name}_step_{step}.png"),
                                      title=f"{name} - step {step}")

    metrics = {}
    P = np.vstack(points)
    for name, algo in algos.items():
        lbl = np.array(labels[name], dtype=int)
        state = getattr(algo, "get_state", lambda: {})() or {}
        centroids = state.get("centroids", None)
        k_from_state = state.get("k", None)
        k_from_labels = int(lbl.max()) + 1 if lbl.size > 0 else 0
        k = int(k_from_state) if isinstance(k_from_state, (int, np.integer)) else k_from_labels

        sse = None
        if centroids is not None:
            sse = float(compute_sse(P, lbl, np.asarray(centroids)))

        save_scatter_snapshot(P, lbl, centroids,
                              out_path=os.path.join(OUTPUT_DIR, f"{name}_final.png"),
                              title=f"{name} - final (n={args.n_points})")

        metrics[name] = {"k": k, "sse": sse, "loss": state.get("loss", None)}

    import json
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


    print("Done. See outputs/ for PNG snapshots and metrics.json.")
    print(algo.get_state())

if __name__ == "__main__":
    main()
