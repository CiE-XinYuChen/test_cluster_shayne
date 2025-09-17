# -*- coding: utf-8 -*-
# https://chatgpt.com/share/68c798e3-ff68-8005-b4cc-af36d0699836
# https://chatgpt.com/s/t_68c8236029c48191b795565cb20b4cdc
# https://chatgpt.com/s/t_68c7bd06e2788191824c62455e7bc64a
import argparse
import os
import sys
import time
import glob
import numpy as np
from typing import List, Dict, Any, Iterable, Optional
from model import ALGORITHM_REGISTRY
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


def _iter_npy_files(root: str) -> Iterable[str]:
    """Yield .npy file paths under `root` (recursively), sorted deterministically."""
    root = os.path.abspath(root)
    # Prefer a 3-level pattern like dev-clean/*/*/*.npy; fallback to recursive walk.
    pattern = os.path.join(root, "**", "*.npy")
    files = sorted(glob.glob(pattern, recursive=True))
    for p in files:
        if os.path.isfile(p) and p.lower().endswith(".npy"):
            yield p


def infer_hubert_dim(root: str) -> int:
    """Infer feature dimension from the first .npy file under `root`.
    Expects arrays of shape (T, D) or (D,). Returns D.
    """
    for p in _iter_npy_files(root):
        a = np.load(p, mmap_mode="r")
        if a.ndim == 1:
            return int(a.shape[0])
        elif a.ndim == 2:
            return int(a.shape[1])
        else:
            raise ValueError(f"Unsupported npy shape {a.shape} in {p}")
    raise FileNotFoundError(f"No .npy files found under {root}")


def hubert_stream(root: str) -> Iterable[np.ndarray]:
    """Stream 1D feature frames from a folder of .npy files.

    Each file may be (T, D) or (D,). This yields row vectors (D,) sequentially
    across files in sorted order.
    """
    for p in _iter_npy_files(root):
        a = np.load(p)
        if a.ndim == 1:
            yield a.astype(float).reshape(-1)
        elif a.ndim == 2:
            # Stream per-frame
            for i in range(a.shape[0]):
                yield a[i].astype(float).reshape(-1)
        else:
            raise ValueError(f"Unsupported npy shape {a.shape} in {p}")


def count_hubert_frames(root: str) -> int:
    """Count total number of frames across all .npy files under root.
    (T,D) contributes T; (D,) contributes 1.
    """
    total = 0
    for p in _iter_npy_files(root):
        a = np.load(p, mmap_mode="r")
        if a.ndim == 1:
            total += 1
        elif a.ndim == 2:
            total += int(a.shape[0])
        else:
            raise ValueError(f"Unsupported npy shape {a.shape} in {p}")
    return int(total)


class ProgressBar:
    def __init__(self, total: Optional[int] = None, width: int = 40, label: str = "progress"):
        self.total = total if (total is None or total > 0) else None
        self.width = max(10, int(width))
        self.label = label
        self.n = 0
        self._last_draw = 0.0
        self._spinner = ['-', '\\', '|', '/']
        self._spin_idx = 0

    def _draw(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_draw) < 0.05:
            return
        self._last_draw = now
        if self.total is None:
            ch = self._spinner[self._spin_idx % len(self._spinner)]
            self._spin_idx += 1
            line = f"[{ch}] {self.label}: {self.n}"
        else:
            ratio = min(1.0, self.n / max(self.total, 1))
            filled = int(ratio * self.width)
            bar = '#' * filled + '-' * (self.width - filled)
            line = f"[{bar}] {self.label}: {self.n}/{self.total} ({ratio*100:5.1f}%)"
        sys.stdout.write('\r' + line)
        sys.stdout.flush()

    def step(self, inc: int = 1):
        self.n += int(inc)
        self._draw()

    def close(self):
        self._draw(force=True)
        sys.stdout.write('\n')
        sys.stdout.flush()

def build_algorithms(names: List[str], dim: int):
    algos = {}
    for name in names:
        if name not in ALGORITHM_REGISTRY:
            raise ValueError(
                f"Algorithm '{name}' not found. Available: {list(ALGORITHM_REGISTRY.keys())}"
            )
        try:
            algos[name] = ALGORITHM_REGISTRY[name](dim=int(dim), name=name)
        except AssertionError as e:
            # Gracefully skip algos that assert on dim (e.g., demo_grid needs 2D)
            print(f"[warn] Skip algo '{name}' for dim={dim}: {e}")
        except Exception as e:
            raise
    if not algos:
        raise RuntimeError("No algorithms instantiated — check names and dim compatibility.")
    return algos

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algos",
        type=str,
        default=(
            "floc,online_kmeans,dp_means,demo_random,demo_grid,minibatch_kmeans,denstream,clustream,streamkmpp,"
        ),
        help="Comma-separated algorithm names from model.ALGORITHM_REGISTRY",
    )
    parser.add_argument("--n_points", type=int, default=10000, help="Number of points/frames to process (cap). Use <=0 for no cap in HuBERT mode.")
    # Alias for convenience/mistype
    parser.add_argument("--n_point", type=int, dest="n_points", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot_every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low", type=float, default=-100.0)
    parser.add_argument("--high", type=float, default=100.0)
    parser.add_argument(
        "--hubert",
        type=str,
        default=None,
        help="Path to folder of .npy HuBERT-like features (frames x D). If set, overrides synthetic stream.",
    )
    parser.add_argument(
        "--center_every",
        type=int,
        default=5000,
        help="HuBERT mode: sample centers every N frames to track change curves (<=0 disables tracking).",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar logging during processing.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    algo_names = [s.strip() for s in args.algos.split(",") if s.strip()]

    # Choose data source and dimension
    use_hubert = bool(args.hubert)
    if use_hubert:
        dim = infer_hubert_dim(args.hubert)
        stream = hubert_stream(args.hubert)
        total_available = None
        if not args.no_progress:
            try:
                total_available = count_hubert_frames(args.hubert)
            except Exception as e:
                print(f"[warn] Could not pre-count frames: {e}")
        if args.n_points and args.n_points > 0:
            cap = int(args.n_points)
            total_display = min(cap, total_available) if total_available else cap
            print(f"[info] HuBERT mode: dim={dim}, root={args.hubert}, cap={cap} frames (total≈{total_available if total_available is not None else 'unknown'})")
        else:
            total_display = total_available
            print(f"[info] HuBERT mode: dim={dim}, root={args.hubert}, no cap (process all frames, total≈{total_available if total_available is not None else 'unknown'})")
    else:
        dim = 2
        stream = gen_stream(args.n_points, args.low, args.high, seed=args.seed)
        print(f"[info] Synthetic mode: 2D stream, seed={args.seed}")

    algos = build_algorithms(algo_names, dim=dim)

    labels: Optional[Dict[str, List[int]]] = None if use_hubert else {name: [] for name in algos}
    points: Optional[List[np.ndarray]] = None if use_hubert else []

    # HuBERT center change tracking
    center_every = int(args.center_every)
    center_trackers: Dict[str, Dict[str, Any]] = {name: {
        'next_id': 0,
        'prev_centers': None,
        'prev_track_ids': None,
        'series': {},  # track_id -> {'steps': [], 'delta': []}
    } for name in algos} if use_hubert and center_every != 0 else {}

    def _match_and_update_tracks(name: str, step: int, centers: np.ndarray):
        tr = center_trackers[name]
        C = np.asarray(centers, dtype=float)
        if C.size == 0:
            return
        if tr['prev_centers'] is None:
            # initialize tracks
            track_ids = []
            for j in range(C.shape[0]):
                tid = tr['next_id']; tr['next_id'] += 1
                tr['series'][tid] = {'steps': [step], 'delta': [0.0]}
                track_ids.append(tid)
            tr['prev_centers'] = C.copy(); tr['prev_track_ids'] = track_ids
            return
        P = tr['prev_centers']; prev_ids = tr['prev_track_ids']
        # Greedy bipartite matching by nearest distances
        import itertools
        pairs = []
        for i in range(P.shape[0]):
            diffs = C - P[i][None, :]
            d2 = np.einsum('ij,ij->i', diffs, diffs)
            for j in range(C.shape[0]):
                pairs.append((float(d2[j]), i, j))
        pairs.sort(key=lambda t: t[0])
        used_prev = set(); used_curr = set(); match = []
        for d, i, j in pairs:
            if i in used_prev or j in used_curr:
                continue
            used_prev.add(i); used_curr.add(j)
            match.append((i, j, float(np.sqrt(d))))
            if len(used_prev) == P.shape[0] or len(used_curr) == C.shape[0]:
                break
        # Update matched tracks
        curr_track_ids = [None] * C.shape[0]
        for i, j, dist in match:
            tid = prev_ids[i]
            curr_track_ids[j] = tid
            series = tr['series'].setdefault(tid, {'steps': [], 'delta': []})
            series['steps'].append(step)
            series['delta'].append(dist)
        # New tracks for unmatched current centers
        for j in range(C.shape[0]):
            if curr_track_ids[j] is None:
                tid = tr['next_id']; tr['next_id'] += 1
                tr['series'][tid] = {'steps': [step], 'delta': [0.0]}
                curr_track_ids[j] = tid
        tr['prev_centers'] = C.copy(); tr['prev_track_ids'] = curr_track_ids

    processed = 0
    bar = None if args.no_progress else ProgressBar(total=total_display if use_hubert else (args.n_points or None), label="processing")
    for x in stream:
        processed += 1
        if not use_hubert:
            points.append(x)
        for name, algo in algos.items():
            label = algo.partial_fit(x)
            if labels is not None:
                labels[name].append(int(label))
        if bar: bar.step()

        if (not use_hubert) and ((processed % args.snapshot_every == 0) or (processed == args.n_points)):
            P = np.vstack(points)
            for name, algo in algos.items():
                lbl = np.array(labels[name], dtype=int)
                state = getattr(algo, "get_state", lambda: {})() or {}
                centroids = state.get("centroids", None)
                # Only visualize when dim==2
                if P.shape[1] == 2:
                    from viz import save_scatter_snapshot  # lazy import to avoid MPL in non-2D mode
                    save_scatter_snapshot(
                        P,
                        lbl,
                        centroids,
                        out_path=os.path.join(OUTPUT_DIR, f"{name}_step_{processed}.png"),
                        title=f"{name} - step {processed}",
                    )
        elif use_hubert and center_trackers and (processed % max(1, center_every) == 0):
            # sample centers for change tracking
            for name, algo in algos.items():
                state = getattr(algo, "get_state", lambda: {})() or {}
                C = state.get("centroids", None)
                if C is None:
                    continue
                _match_and_update_tracks(name, processed, np.asarray(C))

        # Cap by n_points if provided
        if args.n_points and args.n_points > 0 and processed >= args.n_points:
            break

    # Final metrics and optional visualization (2D only)
    if bar: bar.close()
    metrics: Dict[str, Dict[str, Any]] = {}
    if not use_hubert:
        P = np.vstack(points) if points else np.empty((0, dim), dtype=float)
        for name, algo in algos.items():
            lbl = np.array(labels[name], dtype=int) if labels is not None else np.zeros((P.shape[0],), dtype=int)
            state = getattr(algo, "get_state", lambda: {})() or {}
            centroids = state.get("centroids", None)
            k_from_state = state.get("k", None)
            k_from_labels = int(lbl.max()) + 1 if lbl.size > 0 else 0
            k = int(k_from_state) if isinstance(k_from_state, (int, np.integer)) else k_from_labels

            sse = None
            if centroids is not None and P.size > 0:
                sse = float(compute_sse(P, lbl, np.asarray(centroids)))

            if P.shape[1] == 2:
                from viz import save_scatter_snapshot  # lazy import to avoid MPL in non-2D mode
                save_scatter_snapshot(
                    P,
                    lbl,
                    centroids,
                    out_path=os.path.join(OUTPUT_DIR, f"{name}_final.png"),
                    title=f"{name} - final (n={processed})",
                )

            metrics[name] = {"k": k, "sse": sse, "loss": state.get("loss", None)}
    else:
        # In HuBERT mode, export centroids to JSON and print K/shape info
        centers_payload: Dict[str, Dict[str, Any]] = {}
        for name, algo in algos.items():
            state = getattr(algo, "get_state", lambda: {})() or {}
            C = state.get("centroids", None)
            counts = state.get("counts", None)
            radii = state.get("radii", None)
            if C is None:
                k = 0
                dim_c = int(dim)
                centers_list = []
            else:
                C = np.asarray(C)
                k = int(C.shape[0])
                dim_c = int(C.shape[1]) if C.ndim == 2 else int(dim)
                centers_list = C.astype(float).tolist()
            centers_payload[name] = {
                "k": k,
                "dim": dim_c,
                "centroids": centers_list,
                "counts": counts if counts is not None else [],
                "radii": radii if radii is not None else [],
            }
            metrics[name] = {"k": k, "sse": None, "loss": state.get("loss", None)}

        import json
        centers_path = os.path.join(OUTPUT_DIR, "centroids.json")
        with open(centers_path, "w", encoding="utf-8") as f:
            json.dump(centers_payload, f, ensure_ascii=False)
        for name, info in centers_payload.items():
            print(f"[centers] {name}: k={info['k']}, dim={info['dim']} -> saved to {centers_path}")

        # Plot center change curves
        if center_trackers:
            plot_root = os.path.join(OUTPUT_DIR, "center_changes")
            os.makedirs(plot_root, exist_ok=True)
            try:
                import matplotlib.pyplot as plt
            except Exception as e:
                print(f"[warn] Could not import matplotlib for center change plots: {e}")
                plt = None
            if plt is not None:
                for name, tr in center_trackers.items():
                    algo_dir = os.path.join(plot_root, name)
                    os.makedirs(algo_dir, exist_ok=True)
                    for tid, series in tr['series'].items():
                        steps = series.get('steps', [])
                        delta = series.get('delta', [])
                        if not steps:
                            continue
                        plt.figure(figsize=(6, 3))
                        plt.plot(steps, delta, label=f"center {tid}")
                        plt.xlabel("frame")
                        plt.ylabel("center shift (L2)")
                        plt.title(f"{name} center {tid} change")
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        outp = os.path.join(algo_dir, f"center_{tid}.png")
                        plt.savefig(outp, dpi=140)
                        plt.close()

    import json
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


    print("Done. See outputs/ for PNG snapshots (2D) and metrics.json.")

if __name__ == "__main__":
    main()
