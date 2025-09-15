# Stream Clustering Framework (with FLOC)

本仓库提供一套**流式聚类对比与可视化框架**，内置多种在线/流式聚类算法，并特别实现了**FLOC（Fused‑Loss Online Clustering）**：
- 支持在**数据流**场景中动态增减簇；
- 具有**可闭式计算**的总体损失（Loss），便于调参与比较；
- 内置**合并机制**与**遗忘机制**，更适应概念漂移/噪声。

> 代码结构：
> - `run.py`：生成二维随机点流（默认 10,000 点，范围 `[-100, 100]^2`），将点逐一喂给算法，定期保存可视化。
> - `model.py`：算法实现与注册（含 `floc` 与其他基线）。
> - `viz.py` / `metrics.py`：绘图与基本指标（SSE）。
> - `outputs/`：保存快照与最终指标 `metrics.json`（若算法提供 `loss` 也会记录）。

---

## 快速开始

```bash
python run.py --algos floc,dp_means,denstream
python run.py --algos online_kmeans,minibatch_kmeans,streamkmpp
```

运行结束后，查看：
- `outputs/<algo>_step_<t>.png`：第 `<t>` 步的散点快照（按簇着色，若有质心则叠加“x”）。
- `outputs/<algo>_final.png`：最终散点。
- `outputs/metrics.json`：最终指标（`k`、`sse`、可选 `loss`）。

> 注：当前 `run.py` 未将命令行参数直传到构造器，若需调参可在 `model.py` 中各类构造器调整默认值，或扩展 `run.py` 支持 `--algo_kwargs`。

---

## 已实现的算法（注册名）

- `floc`：**Fused‑Loss Online Clustering（FLOC）**
- `online_kmeans`：在线 K‑Means（MacQueen；增量均值/EMA）
- `minibatch_kmeans`：Mini‑Batch K‑Means（k‑means++ 暖启动 + 小批量）
- `dp_means`：DP‑Means（阈值新簇）
- `denstream`：DenStream‑Lite（密度微簇 + 衰减 + 提升/降级）
- `clustream`：CluStream‑Lite（CF 微簇 + 衰减 + 容量合并）
- `streamkmpp`：StreamKM++（reservoir coreset + 周期重聚类）
- `demo_random` / `demo_grid`：演示占位（用于跑通管线与可视化）

---

## FLOC：设计原理与数学方法

### CF 统计与可闭式量
- 质心：`c = LS / N`
- SSE：`SSE = sum(SS) - N * ||c||^2`
- 半径：`R^2 = SSE / N`
- 指数遗忘：`(N,LS,SS) ← ρ^Δt * (N,LS,SS)`（惰性应用）

### 在线分配打分 + 新簇判定
`E_j(x) = ||x - c_j||^2 + β R_j^2 - γ log(N_j + 1)`  
若 `min_j E_j(x) > λ_new` → 新建簇。

### 全局损失（闭式）
`L_total = Σ_j ( SSE_j + α + β * R_j^2 ) = Σ_j ( SSE_j + α + β * SSE_j / N_j )`

### 合并机制（以损失为准则）
- 候选对 `(i,j)` 的收益：`ΔL = L(i∪j) - [L(i)+L(j)]`
- 若 `ΔL < 0`（或 `K > max_k`）则贪心合并；同时清理小权重簇（`N < min_weight`）。
- 合并后的 CF：`N=N_i+N_j, LS=LS_i+LS_j, SS=SS_i+SS_j`（闭式重算 SSE/半径/损失）。

### 复杂度
- 在线分配 `O(K)`；周期合并近似 `O(K^2)`（限制每次最大合并数）。

### 调参提示
- `α`（每簇复杂度）：越大→越少簇；
- `β`（紧凑性）：越大→半径更受惩罚；
- `γ`（在线偏好）：越大→更偏向“稳定大簇”；
- `λ_new`（新簇门槛）：越大→越难新建簇；
- `ρ`（遗忘）：越小→更快跟踪漂移；
- `merge_every/max_k/min_weight` 控制合并频率、容量、清理强度。

---

## 其他算法要点

- **Online K‑Means**：前 K 点为中心；增量均值/EMA 更新；`K` 固定。
- **Mini‑Batch K‑Means**：k‑means++ 暖启动 + 小批量更新；`K` 固定。
- **DP‑Means**：距离阈值触发新簇；易炸簇，暂无合并。
- **DenStream‑Lite**：潜在/离群微簇 + 衰减；密度门槛提升；抗噪。
- **CluStream‑Lite**：CF 微簇 + 衰减；容量满时合并最近对。
- **StreamKM++**：reservoir 代表集；周期重聚类（k‑means++ + 少量 Lloyd）。

---

## 数据流（建议替换）

高斯簇 + 少量离群 + 中心缓慢漂移：
```python
def gen_stream(n_points, low=-100, high=100, seed=42,
               k=6, sigma_range=(5,12), outlier_rate=0.02,
               drift_per_step=0.0, margin=20):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(low+margin, high-margin, size=(k, 2))
    sigmas = rng.uniform(sigma_range[0], sigma_range[1], size=(k,))
    weights = rng.dirichlet(np.ones(k))
    for _ in range(n_points):
        if rng.random() < outlier_rate:
            pt = rng.uniform(low, high, size=(2,))
        else:
            j = int(rng.choice(k, p=weights))
            pt = centers[j] + rng.normal(0.0, sigmas[j], size=(2,))
        if drift_per_step > 0.0:
            centers += rng.normal(0.0, drift_per_step, size=centers.shape)
            centers = np.clip(centers, low+margin, high-margin)
        yield np.clip(pt, low, high)
```

---

## 目录结构

```
stream_clustering/
├── run.py
├── model.py
├── viz.py
├── metrics.py
├── README.md
└── outputs/
```

---

## 许可

本实现用于研究/教学演示，欢迎修改与扩展；如在论文/报告中使用 FLOC 思路，请注明来源与改动。
