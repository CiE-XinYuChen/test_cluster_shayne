# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

class StreamClusterer(ABC):
    def __init__(self, dim: int = 2, name: str = "base"):
        self.dim = int(dim)
        self.name = name

    @abstractmethod
    def partial_fit(self, x: np.ndarray) -> int:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((X.shape[0],), dtype=int)

    def get_state(self) -> Dict[str, Any]:
        return {}

########################################
# DEMO
########################################

class DemoRandomClusterer(StreamClusterer):
    def __init__(self, dim: int = 2, name: str = "demo_random", K: int = 5, seed: int = 2025):
        super().__init__(dim=dim, name=name)
        self.K = int(K)
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros((self.K,), dtype=float)
        self.sums = np.zeros((self.K, dim), dtype=float)

    def partial_fit(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float).reshape(-1)
        label = int(self.rng.integers(0, self.K))
        self.counts[label] += 1.0
        self.sums[label] += x
        return label

    def get_state(self) -> Dict[str, Any]:
        centroids = np.zeros((self.K, self.dim), dtype=float)
        for k in range(self.K):
            if self.counts[k] > 0:
                centroids[k] = self.sums[k] / self.counts[k]
        return {"k": int(self.K), "centroids": centroids, "counts": self.counts.copy().tolist()}

class DemoGridClusterer(StreamClusterer):
    def __init__(self, dim: int = 2, name: str = "demo_grid", cell: float = 20.0):
        super().__init__(dim=dim, name=name)
        assert dim == 2
        self.cell = float(cell)
        self.bin_to_id: Dict[Any, int] = {}
        self.id_to_sum = []
        self.id_to_count = []

    def _bin(self, x: np.ndarray):
        gx = int(np.floor(x[0] / self.cell))
        gy = int(np.floor(x[1] / self.cell))
        return (gx, gy)

    def partial_fit(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float).reshape(-1)
        key = self._bin(x)
        if key not in self.bin_to_id:
            new_id = len(self.bin_to_id)
            self.bin_to_id[key] = new_id
            self.id_to_sum.append(np.zeros((self.dim,), dtype=float))
            self.id_to_count.append(0.0)
        cid = self.bin_to_id[key]
        self.id_to_sum[cid] += x
        self.id_to_count[cid] += 1.0
        return int(cid)

    def get_state(self) -> Dict[str, Any]:
        k = len(self.bin_to_id)
        if k == 0:
            return {"k": 0, "centroids": None, "counts": []}
        centroids = np.zeros((k, self.dim), dtype=float)
        for i in range(k):
            c = self.id_to_count[i]
            centroids[i] = self.id_to_sum[i] / max(c, 1.0)
        return {"k": int(k), "centroids": centroids, "counts": [float(v) for v in self.id_to_count]}

########################################
# Online K-Means
########################################
import math
class OnlineKMeans(StreamClusterer):
    def __init__(self, dim: int = 2, name: str = "online_kmeans",
                 K: int = 30, update: str = "count", alpha: float = 0.05, distance: str = "cosine"):
        super().__init__(dim=dim, name=name)
        self.K = int(K)
        self.update = str(update)
        self.alpha = float(alpha)
        self.centroids = np.empty((0, dim), dtype=float)
        self.counts = np.empty((0,), dtype=float)
        self.distance = str(distance)

    def cosine_distances(self, x: np.ndarray) -> np.ndarray:
        """Cosine distance (1 - cosine similarity) to all centers."""
        # 计算余弦相似度：点积 / (L2范数的乘积)
        cosine_sim = np.dot(self.centroids, x) / (
            np.linalg.norm(self.centroids, axis=1) * np.linalg.norm(x)
        )
        # 余弦距离 = 1 - 余弦相似度
        return 1 - cosine_sim
    
    def euclidean_distances(self, x: np.ndarray) -> np.ndarray:
        """Euclidean distance to all centers."""
        diffs = self.centroids - x
        return np.einsum('ij,ij->i', diffs, diffs)

    def partial_fit(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float).reshape(-1)
        # if self.centroids.shape[0] < self.K:
        #     self.centroids = np.vstack([self.centroids, x[None, :]])
        #     self.counts = np.append(self.counts, 1.0)
        #     return int(self.centroids.shape[0] - 1)
        rng = np.random.default_rng(42)
        while self.centroids.shape[0] < self.K:
            tmp = rng.uniform(-1, 1, size=x.shape)
            self.centroids = np.vstack([self.centroids, tmp[None, :]])
            self.counts = np.append(self.counts, 1.0)
            

        d2 = getattr(self, f"{self.distance}_distances")(x)
        j = int(np.argmin(d2))

        if self.update == "ema":
            eta = self.alpha  #*(2-d2[j])
            self.centroids[j] = (1.0 - eta) * self.centroids[j] + eta * x
            self.counts[j] += 1.0
        else:
            self.counts[j] += 1.0
            eta =  (1.0 / (1+self.counts[j]))**0.4 #*(2-d2[j])#  math.sqrt(1.0 / self.counts[j])
            self.centroids[j] = self.centroids[j] + eta * (x - self.centroids[j])
        return j

    def get_state(self):
        k = int(self.centroids.shape[0])
        return {"k": k, "centroids": self.centroids.copy() if k > 0 else None, "counts": self.counts.copy().tolist() if k > 0 else []}

########################################
# Mini-Batch K-Means
########################################

class MiniBatchKMeans(StreamClusterer):
    def __init__(self, dim: int = 2, name: str = "minibatch_kmeans",
                 K: int = 8, batch_size: int = 256, seed: int = 2025):
        super().__init__(dim=dim, name=name)
        self.K = int(K)
        self.batch_size = int(batch_size)
        self._rng = np.random.default_rng(seed)
        self.centroids = np.empty((0, dim), dtype=float)
        self._buf = []
        self._init_buf = []
        self.counts = np.empty((0,), dtype=float)

    def _kmeanspp_init(self, X: np.ndarray, K: int):
        n = X.shape[0]
        idx0 = int(self._rng.integers(0, n))
        centers = [X[idx0].copy()]
        for _ in range(1, K):
            diffs = X[:, None, :] - np.stack(centers, axis=0)[None, :, :]
            d2 = np.sum(diffs * diffs, axis=2).min(axis=1)
            probs = d2 / max(d2.sum(), 1e-12)
            new_id = self._rng.choice(n, p=probs)
            centers.append(X[int(new_id)].copy())
        return np.stack(centers, axis=0)

    def _mini_update(self, Xb: np.ndarray):
        diffs = Xb[:, None, :] - self.centroids[None, :, :]
        assign = np.sum(diffs * diffs, axis=2).argmin(axis=1)
        sums = np.zeros_like(self.centroids)
        counts = np.zeros((self.centroids.shape[0],), dtype=float)
        for k in range(self.centroids.shape[0]):
            mask = (assign == k)
            if np.any(mask):
                sums[k] = Xb[mask].sum(axis=0)
                counts[k] = float(mask.sum())
        nz = counts > 0
        self.centroids[nz] = sums[nz] / counts[nz, None]

    def partial_fit(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float).reshape(-1)
        if self.centroids.shape[0] < self.K:
            self._init_buf.append(x.copy())
            if len(self._init_buf) >= max(5 * self.K, 128):
                Xw = np.stack(self._init_buf, axis=0)
                self.centroids = self._kmeanspp_init(Xw, self.K)
                self.counts = np.zeros((self.centroids.shape[0],), dtype=float)
                self._mini_update(Xw)
                self._init_buf.clear()
            return 0

        diffs = self.centroids - x
        j = int(np.argmin(np.einsum('ij,ij->i', diffs, diffs)))
        self._buf.append(x.copy())
        if self.counts.shape[0] == 0 and self.centroids.shape[0] > 0:
            self.counts = np.zeros((self.centroids.shape[0],), dtype=float)
        if self.counts.shape[0] > 0:
            self.counts[j] += 1.0
        if len(self._buf) >= self.batch_size:
            Xb = np.stack(self._buf, axis=0)
            self._mini_update(Xb)
            self._buf.clear()
        return j

    def get_state(self):
        if self.centroids.shape[0] == 0:
            return {"k": 0, "centroids": None, "counts": []}
        return {"k": int(self.centroids.shape[0]), "centroids": self.centroids.copy(), "counts": self.counts.copy().tolist() if self.counts.size > 0 else [0.0] * int(self.centroids.shape[0])}

########################################
# DP-Means
########################################

class DPMeans(StreamClusterer):
    def __init__(self, dim: int = 2, name: str = "dp_means",
                 lambda0: float = 400.0, max_k: int = 100,
                 growth: float = 9.0, power: float = 2.0):
        super().__init__(dim=dim, name=name)
        self.lambda0 = float(lambda0)
        self.max_k = int(max_k)
        self.growth = float(growth)
        self.power = float(power)
        self.centroids = np.empty((0, dim), dtype=float)
        self.counts = np.empty((0,), dtype=float)

    def _threshold(self):
        k = self.centroids.shape[0]
        if k == 0:
            return self.lambda0
        return self.lambda0 * (1.0 + self.growth * ((k / max(self.max_k, 1)) ** self.power))

    def partial_fit(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float).reshape(-1)
        if self.centroids.shape[0] == 0:
            self.centroids = x[None, :]
            self.counts = np.array([1.0])
            return 0

        diffs = self.centroids - x
        d2 = np.einsum('ij,ij->i', diffs, diffs)
        j = int(np.argmin(d2))
        if (d2[j] > self._threshold()) and (self.centroids.shape[0] < self.max_k):
            self.centroids = np.vstack([self.centroids, x[None, :]])
            self.counts = np.append(self.counts, 1.0)
            return int(self.centroids.shape[0] - 1)
        self.counts[j] += 1.0
        eta = 1.0 / self.counts[j]
        self.centroids[j] = self.centroids[j] + eta * (x - self.centroids[j])
        return j

    def get_state(self):
        k = int(self.centroids.shape[0])
        return {"k": k, "centroids": self.centroids.copy() if k > 0 else None, "counts": self.counts.copy().tolist() if k > 0 else []}

########################################
# DenStream-Lite
########################################

class DenStreamLite(StreamClusterer):
    class _MC:
        __slots__ = ("c", "w", "t", "id")
        def __init__(self, c, w, t, mc_id):
            self.c = c.astype(float)
            self.w = float(w)
            self.t = int(t)
            self.id = int(mc_id)

    def __init__(self, dim: int = 2, name: str = "denstream",
                 eps: float = 8.0, mu: float = 5.0, beta: float = 0.3,
                 lambd: float = 0.001, cleanup_interval: int = 200):
        super().__init__(dim=dim, name=name)
        self.eps = float(eps)
        self.mu = float(mu)
        self.beta = float(beta)
        self.lambd = float(lambd)
        self.cleanup_interval = int(cleanup_interval)
        self.t = 0
        self._next_id = 0
        self.p_list = []
        self.o_list = []

    def _decay(self, mc, t_now):
        if t_now == mc.t:
            return
        dt = t_now - mc.t
        mc.w *= (2.0 ** (-self.lambd * dt))
        mc.t = t_now

    def _nearest(self, x, mcs):
        if not mcs:
            return None, None, None
        C = np.stack([mc.c for mc in mcs], axis=0)
        diffs = C - x[None, :]
        d2 = np.einsum('ij,ij->i', diffs, diffs)
        j = int(np.argmin(d2))
        return mcs[j], float(np.sqrt(d2[j])), j

    def _new_mc(self, x, w, t_now, to_p=False):
        mc = DenStreamLite._MC(x.copy(), w, t_now, self._next_id); self._next_id += 1
        (self.p_list if to_p else self.o_list).append(mc)
        return mc

    def _cleanup(self):
        keep_p = []
        for mc in self.p_list:
            self._decay(mc, self.t)
            if mc.w >= self.beta * self.mu:
                keep_p.append(mc)
            else:
                if mc.w > 1e-3:
                    self.o_list.append(mc)
        self.p_list = keep_p

        keep_o = []
        for mc in self.o_list:
            self._decay(mc, self.t)
            if mc.w >= self.beta * self.mu:
                self.p_list.append(mc)
            elif mc.w > 1e-3:
                keep_o.append(mc)
        self.o_list = keep_o

    def partial_fit(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float).reshape(-1)
        self.t += 1

        pmc, d, _ = self._nearest(x, self.p_list)
        if (pmc is not None) and (d <= self.eps):
            self._decay(pmc, self.t)
            pmc.c = (pmc.w * pmc.c + x) / (pmc.w + 1.0)
            pmc.w += 1.0
            label = pmc.id
        else:
            omc, d2, idx = self._nearest(x, self.o_list)
            if (omc is not None) and (d2 <= self.eps):
                self._decay(omc, self.t)
                omc.c = (omc.w * omc.c + x) / (omc.w + 1.0)
                omc.w += 1.0
                if omc.w >= self.beta * self.mu:
                    self.p_list.append(omc)
                label = omc.id
            else:
                omc = self._new_mc(x, w=1.0, t_now=self.t, to_p=False)
                label = omc.id

        if (self.t % self.cleanup_interval) == 0:
            self._cleanup()

        return int(label)

    def get_state(self):
        if not self.p_list:
            return {"k": 0, "centroids": None, "counts": []}
        C = np.stack([mc.c for mc in self.p_list], axis=0)
        W = np.array([mc.w for mc in self.p_list], dtype=float)
        return {"k": int(C.shape[0]), "centroids": C, "counts": W.tolist()}

########################################
# CluStream-Lite
########################################

class CluStreamLite(StreamClusterer):
    class _CF:
        __slots__ = ("N","LS","SS","id")
        def __init__(self, x, mc_id):
            self.N = 1.0
            self.LS = x.astype(float).copy()
            self.SS = (x * x).astype(float).copy()
            self.id = int(mc_id)

        @property
        def c(self):
            return self.LS / max(self.N, 1e-12)

        @property
        def radius(self):
            c = self.c
            var = np.maximum(self.SS / max(self.N, 1e-12) - c * c, 0.0)
            return float(np.sqrt(var.sum()))

        def decay(self, rho):
            self.N *= rho; self.LS *= rho; self.SS *= rho

        def absorb(self, x):
            self.N += 1.0; self.LS += x; self.SS += x * x

    def __init__(self, dim: int = 2, name: str = "clustream",
                 q: int = 50, rho: float = 0.999, radius_factor: float = 2.0):
        super().__init__(dim=dim, name=name)
        self.q = int(q)
        self.rho = float(rho)
        self.radius_factor = float(radius_factor)
        self._next_id = 0
        self.mcs = []

    def _nearest_idx(self, x):
        if not self.mcs:
            return None, None
        C = np.stack([mc.c for mc in self.mcs], axis=0)
        diffs = C - x[None, :]
        d2 = np.einsum('ij,ij->i', diffs, diffs)
        j = int(np.argmin(d2))
        return j, float(np.sqrt(d2[j]))

    def _merge_closest(self):
        if len(self.mcs) < 2:
            return
        C = np.stack([mc.c for mc in self.mcs], axis=0)
        D = ((C[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(D, np.inf)
        i, j = np.unravel_index(np.argmin(D), D.shape)
        if i > j: i, j = j, i
        a, b = self.mcs[i], self.mcs[j]
        merged = CluStreamLite._CF(a.c, a.id)
        merged.N = a.N + b.N; merged.LS = a.LS + b.LS; merged.SS = a.SS + b.SS
        self.mcs[i] = merged; del self.mcs[j]

    def partial_fit(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float).reshape(-1)
        if self.mcs:
            for mc in self.mcs:
                mc.decay(self.rho)

        if not self.mcs:
            mc = CluStreamLite._CF(x, self._next_id); self._next_id += 1
            self.mcs.append(mc)
            return mc.id

        idx, d = self._nearest_idx(x)
        mc = self.mcs[idx]
        if d <= self.radius_factor * max(mc.radius, 1e-6):
            mc.absorb(x); label = mc.id
        else:
            if len(self.mcs) < self.q:
                mc2 = CluStreamLite._CF(x, self._next_id); self._next_id += 1
                self.mcs.append(mc2); label = mc2.id
            else:
                self._merge_closest()
                mc2 = CluStreamLite._CF(x, self._next_id); self._next_id += 1
                self.mcs.append(mc2); label = mc2.id
        return int(label)

    def get_state(self):
        if not self.mcs:
            return {"k": 0, "centroids": None, "counts": [], "radii": []}
        C = np.stack([mc.c for mc in self.mcs], axis=0)
        N = np.array([mc.N for mc in self.mcs], dtype=float)
        R = np.array([mc.radius for mc in self.mcs], dtype=float)
        return {"k": int(C.shape[0]), "centroids": C, "counts": N.tolist(), "radii": R.tolist()}

########################################
# StreamKM++
########################################

class StreamKMpp(StreamClusterer):
    def __init__(self, dim: int = 2, name: str = "streamkmpp",
                 K: int = 8, reservoir_size: int = 2000, recluster_every: int = 1000,
                 seed: int = 2025, polish_iters: int = 2):
        super().__init__(dim=dim, name=name)
        self.K = int(K); self.m = int(reservoir_size); self.T = int(recluster_every)
        self.polish_iters = int(polish_iters)
        self._rng = np.random.default_rng(seed); self._t = 0
        self._reservoir = []; self.centroids = np.empty((0, dim), dtype=float)
        self.counts = np.empty((0,), dtype=float)

    def _kmeanspp_init(self, X: np.ndarray, K: int):
        n = X.shape[0]
        idx0 = int(self._rng.integers(0, n)); centers = [X[idx0].copy()]
        for _ in range(1, K):
            diffs = X[:, None, :] - np.stack(centers, axis=0)[None, :, :]
            d2 = np.sum(diffs * diffs, axis=2).min(axis=1)
            probs = d2 / max(d2.sum(), 1e-12); new_id = self._rng.choice(n, p=probs)
            centers.append(X[int(new_id)].copy())
        return np.stack(centers, axis=0)

    def _polish(self, X: np.ndarray):
        if X.shape[0] < self.K: 
            self.centroids = X.copy(); 
            self.counts = np.zeros((self.centroids.shape[0],), dtype=float)
            return
        self.centroids = self._kmeanspp_init(X, self.K)
        self.counts = np.zeros((self.centroids.shape[0],), dtype=float)
        for _ in range(self.polish_iters):
            diffs = X[:, None, :] - self.centroids[None, :, :]
            assign = np.sum(diffs * diffs, axis=2).argmin(axis=1)
            for k in range(self.centroids.shape[0]):
                mask = (assign == k)
                if np.any(mask):
                    self.centroids[k] = X[mask].mean(axis=0)

    def partial_fit(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float).reshape(-1); self._t += 1
        if len(self._reservoir) < self.m:
            self._reservoir.append(x.copy())
        else:
            j = int(self._rng.integers(0, self._t))
            if j < self.m: self._reservoir[j] = x.copy()

        if (self._t % self.T == 0) and (len(self._reservoir) >= max(2, self.K)):
            Xr = np.stack(self._reservoir, axis=0); self._polish(Xr)

        if self.centroids.shape[0] > 0:
            diffs = self.centroids - x
            j = int(np.argmin(np.einsum('ij,ij->i', diffs, diffs)))
            if self.counts.shape[0] == 0:
                self.counts = np.zeros((self.centroids.shape[0],), dtype=float)
            self.counts[j] += 1.0
            return j
        return 0

    def get_state(self):
        if self.centroids.shape[0] == 0: return {"k": 0, "centroids": None, "counts": []}
        return {"k": int(self.centroids.shape[0]), "centroids": self.centroids.copy(), "counts": self.counts.copy().tolist() if self.counts.size > 0 else [0.0] * int(self.centroids.shape[0])}

########################################
# FLOC: Fused-Loss Online Clustering
########################################

class FLOC(StreamClusterer):
    class _CF:
        __slots__ = ("N","LS","SS","t","id")
        def __init__(self, x, t, mc_id):
            x = x.astype(float)
            self.N = 1.0
            self.LS = x.copy()
            self.SS = (x * x).copy()
            self.t = int(t)
            self.id = int(mc_id)

        def clone(self):
            c = FLOC._CF(self.LS*0, self.t, self.id)
            c.N = float(self.N); c.LS = self.LS.copy(); c.SS = self.SS.copy()
            return c

    def __init__(self, dim: int = 2, name: str = "floc",
                 alpha: float = 100.0, beta: float = 0.2, gamma: float = 20.0,
                 lambda_new: float = 2000.0, rho: float = 0.999,
                 merge_every: int = 200, max_k: int = 100, min_weight: float = 1.2,
                 max_merges_per_cleanup: int = 20):
        super().__init__(dim=dim, name=name)
        self.alpha = float(alpha); self.beta = float(beta); self.gamma = float(gamma)
        self.lambda_new = float(lambda_new); self.rho = float(rho)
        self.merge_every = int(merge_every); self.max_k = int(max_k)
        self.min_weight = float(min_weight); self.max_merges_per_cleanup = int(max_merges_per_cleanup)
        self.t = 0; self._next_id = 0; self.mcs = []

    @staticmethod
    def _center(cf): return cf.LS / max(cf.N, 1e-12)
    @staticmethod
    def _sse(cf):
        c = FLOC._center(cf); return float(np.sum(cf.SS) - cf.N * float(np.dot(c, c)))
    @staticmethod
    def _radius2(cf): N = max(cf.N, 1e-12); return FLOC._sse(cf) / N

    def _decay_to_now(self, cf):
        if self.rho >= 1.0 or cf.t == self.t: return
        dt = self.t - cf.t
        if dt <= 0: cf.t = self.t; return
        factor = self.rho ** dt
        cf.N *= factor; cf.LS *= factor; cf.SS *= factor; cf.t = self.t

    def _score(self, x, cf):
        self._decay_to_now(cf)
        c = self._center(cf); d2 = float(np.sum((x - c) ** 2)); R2 = self._radius2(cf)
        sc = d2 + self.beta * R2 - self.gamma * np.log(cf.N + 1.0)
        return sc

    def _new_cf(self, x):
        cf = FLOC._CF(x, self.t, self._next_id); self._next_id += 1
        self.mcs.append(cf); return cf

    def _loss_cf(self, cf):
        self._decay_to_now(cf); sse = self._sse(cf); R2 = sse / max(cf.N, 1e-12)
        return sse + self.alpha + self.beta * R2

    def _loss_total(self): 
        total_loss=sum(self._loss_cf(cf) for cf in self.mcs)
        # print(total_loss)
        return total_loss

    @staticmethod
    def _merge_cf(cf_a, cf_b, t_now):
        m = cf_a.clone(); m.N = cf_a.N + cf_b.N; m.LS = cf_a.LS + cf_b.LS; m.SS = cf_a.SS + cf_b.SS
        m.t = t_now; m.id = min(cf_a.id, cf_b.id); return m

    def _cleanup_and_merge(self):
        kept = []
        for cf in self.mcs:
            self._decay_to_now(cf)
            if cf.N >= self.min_weight: kept.append(cf)
        self.mcs = kept
        if len(self.mcs) <= 1: return

        Ls = [self._loss_cf(cf) for cf in self.mcs]
        merges_done = 0
        while merges_done < self.max_merges_per_cleanup and len(self.mcs) >= 2:
            best = None; best_delta = None; best_pair = None
            for i in range(len(self.mcs)):
                for j in range(i+1, len(self.mcs)):
                    ci, cj = self.mcs[i], self.mcs[j]
                    m = self._merge_cf(ci, cj, self.t)
                    L_merged = self._loss_cf(m)
                    delta = L_merged - (Ls[i] + Ls[j])
                    if (best_delta is None) or (delta < best_delta):
                        best_delta = delta; best = m; best_pair = (i, j)
            must_merge = (len(self.mcs) > self.max_k)
            if (best_delta is not None) and (best_delta < 0 or must_merge):
                i, j = best_pair
                self.mcs[i] = best; del self.mcs[j]
                Ls[i] = self._loss_cf(best); del Ls[j]
                merges_done += 1
            else:
                break
    def linear_normalize(self,x):
        x = x - np.min(x)
        if np.sum(x) == 0:
            return np.ones_like(x) / len(x)
        return 2*x / np.sum(x)-1
    
    def min_max_normalize_neg1(self,x):
        """线性缩放到[-1, 1]区间"""
        x_min = np.min(x)
        x_max = np.max(x)
        
        if x_max == x_min:  # 所有值相同
            return np.zeros_like(x)
        
        # 公式: 2 * (x - x_min) / (x_max - x_min) - 1
        normalized = 2 * (x - x_min) / (x_max - x_min) - 1
        return normalized
    def softmax(self,x, axis=1,tau=1.0):
        x = np.asarray(x, dtype=float).reshape(-1)
        x = x / tau
        # 减去最大值提高数值稳定性（防止指数爆炸）
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def partial_fit(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float).reshape(-1); self.t += 1
        if not self.mcs:
            self._new_cf(x); return 0
        scores = [self._score(x, cf) for cf in self.mcs]
        # print(self.softmax(scores,axis=0,tau=10.0))
        # print(self.linear_normalize(scores))
        j = int(np.argmin(scores))
        if scores[j] > self.lambda_new and (len(self.mcs) < self.max_k):
            self._new_cf(x); label = len(self.mcs) - 1
        else:
            cf = self.mcs[j]; self._decay_to_now(cf)
            cf.N += 1.0; cf.LS += x; cf.SS += x * x; label = j
        if (self.t % self.merge_every) == 0:
            self._cleanup_and_merge()
        return int(label)

    def get_state(self):
        if not self.mcs: 
            return {"k": 0, "centroids": None, "loss": 0.0, "counts": [], "radii": []}
        C = np.stack([self._center(cf) for cf in self.mcs], axis=0)
        counts = [float(cf.N) for cf in self.mcs]
        radii = [float(np.sqrt(FLOC._radius2(cf))) for cf in self.mcs]
        return {"k": int(C.shape[0]), "centroids": C, "loss": float(self._loss_total()), "counts": counts, "radii": radii}

# Registry
ALGORITHM_REGISTRY = {
    "demo_random": DemoRandomClusterer,
    "demo_grid": DemoGridClusterer,
    "online_kmeans": OnlineKMeans,
    "minibatch_kmeans": MiniBatchKMeans,
    "dp_means": DPMeans,
    "denstream": DenStreamLite,
    "clustream": CluStreamLite,
    "streamkmpp": StreamKMpp,
    "floc": FLOC,
}
