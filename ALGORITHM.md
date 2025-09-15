# FLOC Cheat Sheet (Fused‑Loss Online Clustering)

This is a compact reference for the FLOC algorithm used in this repo. It summarizes the key formulas, rules, and defaults implemented in `model.py` and described in `README.md`.

## Core CF Stats (per micro‑cluster)

- Sufficient statistics: `N, LS, SS`.
- Centroid: `c = LS / N`.
- SSE: `SSE = sum(SS) - N * ||c||^2`.
- Radius: `R^2 = SSE / N`.
- Lazy exponential forgetting: when current time is `t` and CF last updated at `t_cf`, apply `Δt = t − t_cf`, then `(N, LS, SS) ← ρ^Δt * (N, LS, SS)` and set `t_cf = t`.

## Online Assignment Score

- For an incoming point `x`, each existing cluster `j` gets score:
  `E_j(x) = ||x − c_j||^2 + β * R_j^2 − γ * log(N_j + 1)`
- Pick `j* = argmin_j E_j(x)`.

## New Cluster Rule

- If `min_j E_j(x) > λ_new` and current number of micro‑clusters `K < max_k * 3`, create a new micro‑cluster initialized at `x`.

## Update Existing Cluster

- Otherwise, decay the chosen CF to now, then update:
  `N ← N + 1;  LS ← LS + x;  SS ← SS + x ⊙ x`.

## Global Loss (closed form)

- Per cluster loss: `L_j = SSE_j + α + β * R_j^2 = SSE_j + α + β * SSE_j / N_j`.
- Total loss: `L_total = Σ_j L_j`.

## Merge & Cleanup

- Cleanup small clusters: drop CFs with `N < min_weight` (after decay).
- Merge criterion (greedy): for candidate pair `(i, j)`, compute
  `ΔL = L(i ∪ j) − [L(i) + L(j)]` using merged CF (`N, LS, SS` additively).
- Merge if `ΔL < 0` or if `K > max_k`. Limit merges per cleanup by `max_merges_per_cleanup`.
- Cleanup/merge runs every `merge_every` points.

## Complexity

- Assignment per point: `O(K)`.
- Periodic merge: approx `O(K^2)` per cleanup (bounded by max merges).

## Default Parameters (model.py)

- `alpha = 100.0`
- `beta = 0.2`
- `gamma = 20.0`
- `lambda_new = 2000.0`
- `rho = 0.999`
- `merge_every = 200`
- `max_k = 100`
- `min_weight = 1.2`
- `max_merges_per_cleanup = 20`

## Tuning Notes

- `α` (per‑cluster complexity): larger → fewer clusters.
- `β` (compactness penalty): larger → penalizes large radii more.
- `γ` (online stability bias): larger → favors bigger/stable clusters.
- `λ_new` (new‑cluster threshold): larger → harder to create new clusters.
- `ρ` (forgetting): smaller → faster adaptation to drift/noise.
- `merge_every`, `max_k`, `min_weight`, `max_merges_per_cleanup`: control merge frequency, capacity pressure, and cleanup strength.

## Pointers

- Implementation: `model.py`, class `FLOC`; registry key: `"floc"`.
- Run demo: `python run.py --algos floc`.
- Outputs: snapshots in `outputs/`, final metrics in `outputs/metrics.json` (includes `loss` for FLOC).
