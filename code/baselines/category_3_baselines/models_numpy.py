
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def clip_score(values: Sequence[float]) -> np.ndarray:
    return np.clip(np.array(values, dtype=np.float64), 1.0, 5.0)


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


class RidgeRegressor:
    def __init__(self, alpha: float = 1.0, max_iter: int = 80):
        self.alpha = alpha
        self.max_iter = max_iter
        self.weights: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
        weights = np.zeros(x_aug.shape[1], dtype=np.float64)
        weights[-1] = float(np.median(y)) if len(y) else 3.0
        penalty = np.eye(x_aug.shape[1], dtype=np.float64) * self.alpha
        penalty[-1, -1] = 0.0
        for _ in range(self.max_iter):
            residual = x_aug @ weights - y
            sample_weights = 1.0 / np.maximum(np.abs(residual), 0.05)
            weighted_x = x_aug * sample_weights[:, None]
            gram = x_aug.T @ weighted_x
            rhs = weighted_x.T @ y
            try:
                updated = np.linalg.solve(gram + penalty, rhs)
            except np.linalg.LinAlgError:
                updated = np.linalg.pinv(gram + penalty) @ rhs
            if np.max(np.abs(updated - weights)) < 1e-6:
                weights = updated
                break
            weights = updated
        self.weights = weights
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("RidgeRegressor must be fitted before prediction.")
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
        return clip_score(x_aug @ self.weights)


class AdamOptimizer:
    def __init__(self, params: Dict[str, np.ndarray]):
        self.m = {name: np.zeros_like(value) for name, value in params.items()}
        self.v = {name: np.zeros_like(value) for name, value in params.items()}
        self.t = 0

    def step(
        self,
        params: Dict[str, np.ndarray],
        grads: Dict[str, np.ndarray],
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.t += 1
        for name, grad in grads.items():
            self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * (grad * grad)
            m_hat = self.m[name] / (1.0 - beta1**self.t)
            v_hat = self.v[name] / (1.0 - beta2**self.t)
            params[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)


class MLPClassifier:
    def __init__(
        self,
        hidden_dims: Sequence[int] = (64,),
        epochs: int = 180,
        batch_size: int = 64,
        lr: float = 0.004,
        l2: float = 1e-4,
        seed: int = 13,
    ):
        self.hidden_dims = tuple(hidden_dims)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.l2 = l2
        self.seed = seed
        self.params: Dict[str, np.ndarray] = {}

    def _init_params(self, input_dim: int) -> None:
        rng = np.random.default_rng(self.seed)
        dims = [input_dim, *self.hidden_dims, 1]
        self.params = {}
        for idx in range(len(dims) - 1):
            scale = math.sqrt(2.0 / max(dims[idx] + dims[idx + 1], 1))
            self.params[f"W{idx}"] = rng.normal(0.0, scale, size=(dims[idx], dims[idx + 1]))
            self.params[f"b{idx}"] = np.zeros(dims[idx + 1], dtype=np.float64)

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        activations = [x]
        preacts: List[np.ndarray] = []
        h = x
        last = len(self.hidden_dims)
        for idx in range(last):
            z = h @ self.params[f"W{idx}"] + self.params[f"b{idx}"]
            preacts.append(z)
            h = relu(z)
            activations.append(h)
        logits = h @ self.params[f"W{last}"] + self.params[f"b{last}"]
        return logits, activations, preacts

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        self._init_params(x.shape[1])
        optimizer = AdamOptimizer(self.params)
        rng = np.random.default_rng(self.seed)
        y_target = y.reshape(-1, 1).astype(np.float64)

        for _ in range(self.epochs):
            order = rng.permutation(x.shape[0])
            for start in range(0, x.shape[0], self.batch_size):
                batch_idx = order[start : start + self.batch_size]
                xb = x[batch_idx]
                yb = y_target[batch_idx]
                logits, activations, preacts = self._forward(xb)
                delta = np.sign(logits - yb) / max(len(batch_idx), 1)
                grads: Dict[str, np.ndarray] = {}
                last = len(self.hidden_dims)
                grads[f"W{last}"] = activations[-1].T @ delta + self.l2 * self.params[f"W{last}"]
                grads[f"b{last}"] = delta.sum(axis=0)
                dh = delta @ self.params[f"W{last}"].T
                for idx in range(last - 1, -1, -1):
                    dh = dh * (preacts[idx] > 0.0)
                    grads[f"W{idx}"] = activations[idx].T @ dh + self.l2 * self.params[f"W{idx}"]
                    grads[f"b{idx}"] = dh.sum(axis=0)
                    if idx > 0:
                        dh = dh @ self.params[f"W{idx}"].T
                optimizer.step(self.params, grads, self.lr)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits, _, _ = self._forward(x)
        return logits

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits, _, _ = self._forward(x)
        return clip_score(logits.ravel())


@dataclass
class TreeNode:
    value: float
    feature: int = -1
    threshold: float = 0.0
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


class RegressionTree:
    def __init__(
        self,
        max_depth: int = 7,
        min_samples_leaf: int = 20,
        max_features: Optional[int] = None,
        seed: int = 13,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = np.random.default_rng(seed)
        self.root: Optional[TreeNode] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RegressionTree":
        self.root = self._build(x, y, np.arange(x.shape[0]), 0)
        return self

    def _build(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray, depth: int) -> TreeNode:
        node_value = float(np.median(y[indices])) if len(indices) else 3.0
        if depth >= self.max_depth or len(indices) < 2 * self.min_samples_leaf or np.var(y[indices]) < 1e-8:
            return TreeNode(value=node_value)

        n_features = x.shape[1]
        max_features = self.max_features or max(1, int(math.sqrt(n_features)))
        feature_candidates = self.rng.choice(n_features, size=min(max_features, n_features), replace=False)
        parent_loss = float(np.abs(y[indices] - node_value).sum())
        best_score = parent_loss
        best_feature = -1
        best_threshold = 0.0
        best_left: Optional[np.ndarray] = None
        best_right: Optional[np.ndarray] = None

        for feature in feature_candidates:
            feature_values = x[indices, feature]
            if np.allclose(feature_values, feature_values[0]):
                continue
            thresholds = np.unique(np.percentile(feature_values, [10, 20, 35, 50, 65, 80, 90]))
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                left_count = int(left_mask.sum())
                right_count = len(indices) - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                left_idx = indices[left_mask]
                right_idx = indices[~left_mask]
                left_value = float(np.median(y[left_idx]))
                right_value = float(np.median(y[right_idx]))
                score = float(np.abs(y[left_idx] - left_value).sum() + np.abs(y[right_idx] - right_value).sum())
                if score < best_score:
                    best_score = score
                    best_feature = int(feature)
                    best_threshold = float(threshold)
                    best_left = left_idx
                    best_right = right_idx

        if best_feature < 0 or best_left is None or best_right is None:
            return TreeNode(value=node_value)
        return TreeNode(
            value=node_value,
            feature=best_feature,
            threshold=best_threshold,
            left=self._build(x, y, best_left, depth + 1),
            right=self._build(x, y, best_right, depth + 1),
        )

    def _predict_one(self, row: np.ndarray, node: TreeNode) -> float:
        while node.feature >= 0 and node.left is not None and node.right is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("RegressionTree must be fitted before prediction.")
        return np.array([self._predict_one(row, self.root) for row in x], dtype=np.float64)


class LSTMSequenceClassifier:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 28,
        epochs: int = 22,
        batch_size: int = 64,
        lr: float = 0.003,
        l2: float = 1e-5,
        seed: int = 13,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.l2 = l2
        self.seed = seed
        self.params: Dict[str, np.ndarray] = {}

    def _init_params(self) -> None:
        rng = np.random.default_rng(self.seed)
        gate_dim = self.input_dim + self.hidden_dim
        scale = math.sqrt(1.0 / gate_dim)
        self.params = {}
        for name in ["i", "f", "o", "g"]:
            self.params[f"W{name}"] = rng.normal(0.0, scale, size=(gate_dim, self.hidden_dim))
            self.params[f"b{name}"] = np.zeros(self.hidden_dim, dtype=np.float64)
        self.params["Wy"] = rng.normal(0.0, math.sqrt(1.0 / self.hidden_dim), size=(self.hidden_dim, 1))
        self.params["by"] = np.zeros(1, dtype=np.float64)

    def _forward(self, x: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]], np.ndarray]:
        batch = x.shape[0]
        h = np.zeros((batch, self.hidden_dim), dtype=np.float64)
        c = np.zeros_like(h)
        caches: List[Dict[str, np.ndarray]] = []
        for t in range(x.shape[1]):
            xt = x[:, t, :]
            m = mask[:, t : t + 1]
            h_prev = h
            c_prev = c
            z = np.concatenate([xt, h_prev], axis=1)
            i = sigmoid(z @ self.params["Wi"] + self.params["bi"])
            f = sigmoid(z @ self.params["Wf"] + self.params["bf"])
            o = sigmoid(z @ self.params["Wo"] + self.params["bo"])
            g = np.tanh(z @ self.params["Wg"] + self.params["bg"])
            c_candidate = f * c_prev + i * g
            h_candidate = o * np.tanh(c_candidate)
            c = m * c_candidate + (1.0 - m) * c_prev
            h = m * h_candidate + (1.0 - m) * h_prev
            caches.append(
                {
                    "z": z,
                    "i": i,
                    "f": f,
                    "o": o,
                    "g": g,
                    "c_prev": c_prev,
                    "c_candidate": c_candidate,
                    "h_prev": h_prev,
                    "m": m,
                }
            )
        logits = h @ self.params["Wy"] + self.params["by"]
        return logits, caches, h

    def fit(self, x: np.ndarray, mask: np.ndarray, y: np.ndarray) -> "LSTMSequenceClassifier":
        self._init_params()
        optimizer = AdamOptimizer(self.params)
        rng = np.random.default_rng(self.seed)
        y_target = y.reshape(-1, 1).astype(np.float64)
        for _ in range(self.epochs):
            order = rng.permutation(x.shape[0])
            for start in range(0, x.shape[0], self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = x[idx]
                mb = mask[idx]
                yb = y_target[idx]
                logits, caches, final_h = self._forward(xb, mb)
                dlogits = np.sign(logits - yb) / max(len(idx), 1)
                grads = {name: np.zeros_like(value) for name, value in self.params.items()}
                grads["Wy"] = final_h.T @ dlogits + self.l2 * self.params["Wy"]
                grads["by"] = dlogits.sum(axis=0)
                dh = dlogits @ self.params["Wy"].T
                dc = np.zeros_like(dh)
                for cache in reversed(caches):
                    z = cache["z"]
                    i = cache["i"]
                    f = cache["f"]
                    o = cache["o"]
                    g = cache["g"]
                    c_prev = cache["c_prev"]
                    c_candidate = cache["c_candidate"]
                    m = cache["m"]

                    dh_candidate = dh * m
                    dh_prev_skip = dh * (1.0 - m)
                    tanh_c = np.tanh(c_candidate)
                    do = dh_candidate * tanh_c
                    dc_candidate = dc * m + dh_candidate * o * (1.0 - tanh_c * tanh_c)
                    dc_prev = dc * (1.0 - m) + dc_candidate * f
                    df = dc_candidate * c_prev
                    di = dc_candidate * g
                    dg = dc_candidate * i

                    dpi = di * i * (1.0 - i)
                    dpf = df * f * (1.0 - f)
                    dpo = do * o * (1.0 - o)
                    dpg = dg * (1.0 - g * g)

                    dz = np.zeros_like(z)
                    for gate, dpre in [("i", dpi), ("f", dpf), ("o", dpo), ("g", dpg)]:
                        grads[f"W{gate}"] += z.T @ dpre + self.l2 * self.params[f"W{gate}"]
                        grads[f"b{gate}"] += dpre.sum(axis=0)
                        dz += dpre @ self.params[f"W{gate}"].T
                    dh = dh_prev_skip + dz[:, self.input_dim :]
                    dc = dc_prev

                for name in grads:
                    np.clip(grads[name], -5.0, 5.0, out=grads[name])
                optimizer.step(self.params, grads, self.lr)
        return self

    def predict_proba(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        logits, _, _ = self._forward(x, mask)
        return logits

    def predict(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        logits, _, _ = self._forward(x, mask)
        return clip_score(logits.ravel())


class TransformerSequenceClassifier:
    def __init__(
        self,
        input_dim: int,
        max_len: int,
        hidden_dim: int = 28,
        epochs: int = 18,
        batch_size: int = 48,
        lr: float = 0.0025,
        l2: float = 1e-5,
        seed: int = 13,
    ):
        self.input_dim = input_dim
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.l2 = l2
        self.seed = seed
        self.params: Dict[str, np.ndarray] = {}
        self.positional = self._positional_encoding(max_len, hidden_dim)

    def _positional_encoding(self, max_len: int, hidden_dim: int) -> np.ndarray:
        positions = np.arange(max_len)[:, None]
        dims = np.arange(hidden_dim)[None, :]
        div = np.exp(-(dims // 2) * math.log(10000.0) / max(hidden_dim, 1))
        angles = positions * div
        encoded = np.zeros((max_len, hidden_dim), dtype=np.float64)
        encoded[:, 0::2] = np.sin(angles[:, 0::2])
        encoded[:, 1::2] = np.cos(angles[:, 1::2])
        return encoded

    def _init_params(self) -> None:
        rng = np.random.default_rng(self.seed)
        scale_in = math.sqrt(2.0 / max(self.input_dim + self.hidden_dim, 1))
        scale_h = math.sqrt(1.0 / max(self.hidden_dim, 1))
        self.params = {
            "Win": rng.normal(0.0, scale_in, size=(self.input_dim, self.hidden_dim)),
            "Wq": rng.normal(0.0, scale_h, size=(self.hidden_dim, self.hidden_dim)),
            "Wk": rng.normal(0.0, scale_h, size=(self.hidden_dim, self.hidden_dim)),
            "Wv": rng.normal(0.0, scale_h, size=(self.hidden_dim, self.hidden_dim)),
            "Wy": rng.normal(0.0, scale_h, size=(self.hidden_dim, 1)),
            "by": np.zeros(1, dtype=np.float64),
        }

    def _forward(self, x: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        batch, seq_len, _ = x.shape
        e = x @ self.params["Win"] + self.positional[:seq_len][None, :, :]
        q = e @ self.params["Wq"]
        k = e @ self.params["Wk"]
        v = e @ self.params["Wv"]
        scale = 1.0 / math.sqrt(self.hidden_dim)
        scores = (q @ np.swapaxes(k, 1, 2)) * scale
        key_mask = mask[:, None, :]
        scores = np.where(key_mask > 0.0, scores, -1e9)
        attn = np.exp(scores - scores.max(axis=2, keepdims=True))
        attn = attn / attn.sum(axis=2, keepdims=True)
        encoded = attn @ v
        lengths = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
        pooled = (encoded * mask[:, :, None]).sum(axis=1) / lengths
        logits = pooled @ self.params["Wy"] + self.params["by"]
        cache = {
            "x": x,
            "mask": mask,
            "e": e,
            "q": q,
            "k": k,
            "v": v,
            "attn": attn,
            "encoded": encoded,
            "lengths": lengths,
            "pooled": pooled,
        }
        return logits, cache

    def fit(self, x: np.ndarray, mask: np.ndarray, y: np.ndarray) -> "TransformerSequenceClassifier":
        self._init_params()
        optimizer = AdamOptimizer(self.params)
        rng = np.random.default_rng(self.seed)
        y_target = y.reshape(-1, 1).astype(np.float64)
        for _ in range(self.epochs):
            order = rng.permutation(x.shape[0])
            for start in range(0, x.shape[0], self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = x[idx]
                mb = mask[idx]
                yb = y_target[idx]
                logits, cache = self._forward(xb, mb)
                dlogits = np.sign(logits - yb) / max(len(idx), 1)
                grads = {name: np.zeros_like(value) for name, value in self.params.items()}
                pooled = cache["pooled"]
                grads["Wy"] = pooled.T @ dlogits + self.l2 * self.params["Wy"]
                grads["by"] = dlogits.sum(axis=0)
                dpooled = dlogits @ self.params["Wy"].T

                dencoded = dpooled[:, None, :] * cache["mask"][:, :, None] / cache["lengths"][:, None, :]
                attn = cache["attn"]
                v = cache["v"]
                q = cache["q"]
                k = cache["k"]
                e = cache["e"]
                dattn = dencoded @ np.swapaxes(v, 1, 2)
                dv = np.swapaxes(attn, 1, 2) @ dencoded
                dscores = attn * (dattn - (dattn * attn).sum(axis=2, keepdims=True))
                scale = 1.0 / math.sqrt(self.hidden_dim)
                dq = (dscores @ k) * scale
                dk = (np.swapaxes(dscores, 1, 2) @ q) * scale

                de = dq @ self.params["Wq"].T + dk @ self.params["Wk"].T + dv @ self.params["Wv"].T
                grads["Wq"] = e.reshape(-1, self.hidden_dim).T @ dq.reshape(-1, self.hidden_dim) + self.l2 * self.params["Wq"]
                grads["Wk"] = e.reshape(-1, self.hidden_dim).T @ dk.reshape(-1, self.hidden_dim) + self.l2 * self.params["Wk"]
                grads["Wv"] = e.reshape(-1, self.hidden_dim).T @ dv.reshape(-1, self.hidden_dim) + self.l2 * self.params["Wv"]
                grads["Win"] = xb.reshape(-1, self.input_dim).T @ de.reshape(-1, self.hidden_dim) + self.l2 * self.params["Win"]
                for name in grads:
                    np.clip(grads[name], -5.0, 5.0, out=grads[name])
                optimizer.step(self.params, grads, self.lr)
        return self

    def predict_proba(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        logits, _ = self._forward(x, mask)
        return logits

    def predict(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        logits, _ = self._forward(x, mask)
        return clip_score(logits.ravel())
