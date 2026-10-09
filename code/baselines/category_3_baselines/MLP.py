
from pathlib import Path
from typing import Any, Dict

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
from models_numpy import MLPClassifier


BASELINE_NAME = "MLP"


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
    model = MLPClassifier(hidden_dims=(64, 32), epochs=70, batch_size=64, lr=0.004, seed=seed).fit(x_train, y_train)
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
