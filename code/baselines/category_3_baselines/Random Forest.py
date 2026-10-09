
from pathlib import Path
from typing import Any, Dict

import numpy as np

from baseline_common import (
    build_arg_parser,
    default_output_path,
    labels_for_examples,
    load_examples,
    load_usefulness_scores,
    prepare_tabular_features,
    print_run_summary,
    save_predictions_excel,
)


BASELINE_NAME = "Random Forest"


def mae_leaf_loss(y: np.ndarray, score: float) -> float:
    return float(np.abs(y - float(score)).sum())


def best_leaf_value(y: np.ndarray) -> float:
    return float(np.median(y)) if len(y) else 3.0


class MaeTreeNode:
    def __init__(self, value: float, feature: int = -1, threshold: float = 0.0):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = None
        self.right = None


class MaeRegressionTree:
    def __init__(self, max_depth: int = 2, min_samples_leaf: int = 10, max_features: int = 0, seed: int = 13):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = np.random.default_rng(seed)
        self.root = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MaeRegressionTree":
        self.root = self._build(x, y, np.arange(x.shape[0]), 0)
        return self

    def _build(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray, depth: int) -> MaeTreeNode:
        node_y = y[indices]
        node_value = best_leaf_value(node_y)
        node = MaeTreeNode(node_value)
        if depth >= self.max_depth or len(indices) < 2 * self.min_samples_leaf or np.var(node_y) < 1e-8:
            return node

        n_features = x.shape[1]
        max_features = self.max_features or max(1, int(np.sqrt(n_features)))
        feature_candidates = self.rng.choice(n_features, size=min(max_features, n_features), replace=False)
        best_loss = mae_leaf_loss(node_y, node_value)
        best_feature = -1
        best_threshold = 0.0
        best_left = None
        best_right = None

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
                left_value = best_leaf_value(y[left_idx])
                right_value = best_leaf_value(y[right_idx])
                loss = mae_leaf_loss(y[left_idx], left_value) + mae_leaf_loss(y[right_idx], right_value)
                if loss < best_loss:
                    best_loss = loss
                    best_feature = int(feature)
                    best_threshold = float(threshold)
                    best_left = left_idx
                    best_right = right_idx

        if best_feature < 0 or best_left is None or best_right is None:
            return node
        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self._build(x, y, best_left, depth + 1)
        node.right = self._build(x, y, best_right, depth + 1)
        return node

    def predict_one(self, row: np.ndarray) -> float:
        node = self.root
        while node.feature >= 0 and node.left is not None and node.right is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return float(node.value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(row) for row in x], dtype=np.float64)


class MaeRandomForest:
    def __init__(self, n_estimators: int = 10, max_depth: int = 2, min_samples_leaf: int = 10, max_features: int = 0, seed: int = 13):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.trees = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MaeRandomForest":
        rng = np.random.default_rng(self.seed)
        max_features = self.max_features or max(1, int(np.sqrt(x.shape[1])))
        self.trees = []
        for tree_idx in range(self.n_estimators):
            sample_idx = rng.integers(0, x.shape[0], size=x.shape[0])
            tree = MaeRegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                seed=self.seed + tree_idx + 1,
            ).fit(x[sample_idx], y[sample_idx])
            self.trees.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = np.vstack([tree.predict(x) for tree in self.trees])
        return np.clip(predictions.mean(axis=0), 1.0, 5.0)


def run(
    train_path: str,
    test_path: str,
    output_dir: str,
    output_path: str = "",
    seed: int = 13,
    usefulness_jsonl: str = "",
) -> Dict[str, Any]:
    train_examples = load_examples(train_path)
    test_examples = load_examples(test_path)
    usefulness_by_id = load_usefulness_scores(usefulness_jsonl)
    y_train = labels_for_examples(train_examples)
    y_test = labels_for_examples(test_examples)
    x_train, x_test, _, _ = prepare_tabular_features(train_examples, test_examples, usefulness_by_id=usefulness_by_id)
    model = MaeRandomForest(n_estimators=10, max_depth=4, min_samples_leaf=10, seed=seed).fit(x_train, y_train)
    predictions = model.predict(x_test)
    out = Path(output_path) if output_path else default_output_path(output_dir, BASELINE_NAME)
    save_predictions_excel(predictions, y_test, out, BASELINE_NAME)
    print_run_summary(BASELINE_NAME, train_examples, test_examples, out)
    return {"name": BASELINE_NAME, "predictions": predictions, "truths": y_test, "output_path": str(out)}


def main() -> None:
    parser = build_arg_parser(BASELINE_NAME)
    args = parser.parse_args()
    run(args.train, args.test, args.output_dir, args.output, args.seed, args.usefulness_jsonl)


if __name__ == "__main__":
    main()
