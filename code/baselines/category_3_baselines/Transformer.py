
from pathlib import Path
from typing import Any, Dict

from baseline_common import (
    build_arg_parser,
    default_output_path,
    labels_for_examples,
    load_examples,
    load_usefulness_scores,
    print_run_summary,
    save_predictions_excel,
    sequence_tensor,
)
from models_numpy import TransformerSequenceClassifier


BASELINE_NAME = "Transformer"


def run(
    train_path: str,
    test_path: str,
    output_dir: str,
    output_path: str = "",
    seed: int = 13,
    usefulness_jsonl: str = "",
) -> Dict[str, Any]:
    max_len = 64
    train_examples = load_examples(train_path)
    test_examples = load_examples(test_path)
    usefulness_by_id = load_usefulness_scores(usefulness_jsonl)
    y_train = labels_for_examples(train_examples)
    y_test = labels_for_examples(test_examples)
    x_train, mask_train = sequence_tensor(train_examples, max_len=max_len, usefulness_by_id=usefulness_by_id)
    x_test, mask_test = sequence_tensor(test_examples, max_len=max_len, usefulness_by_id=usefulness_by_id)
    model = TransformerSequenceClassifier(
        input_dim=x_train.shape[2],
        max_len=max_len,
        hidden_dim=16,
        epochs=10,
        batch_size=48,
        lr=0.0025,
        seed=seed,
    ).fit(x_train, mask_train, y_train)
    predictions = model.predict(x_test, mask_test)
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
