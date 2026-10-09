
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
from models_numpy import RegressionTree, clip_score


BASELINE_NAME = "GBDT"


def mae_negative_gradient(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sign(y - scores)


def fit_predict_gbdt(x_train, y_train, x_test, seed: int):
    learning_rate = 0.035
    n_estimators = 10
    pred_train = np.full(x_train.shape[0], float(np.median(y_train)), dtype=np.float64)
    pred_test = np.full(x_test.shape[0], float(np.median(y_train)), dtype=np.float64)
    for tree_idx in range(n_estimators):
        residual = mae_negative_gradient(pred_train, y_train)
        tree = RegressionTree(
            max_depth=5,
            min_samples_leaf=20,
            max_features=x_train.shape[1],
            seed=seed + tree_idx,
        ).fit(x_train, residual)
        pred_train += learning_rate * tree.predict(x_train)
        pred_test += learning_rate * tree.predict(x_test)
    return clip_score(pred_test)


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
    predictions = fit_predict_gbdt(x_train, y_train, x_test, seed)
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
