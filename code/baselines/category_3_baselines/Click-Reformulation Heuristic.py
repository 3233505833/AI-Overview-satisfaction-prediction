
import math
from pathlib import Path
from typing import Any, Dict

from baseline_common import (
    build_arg_parser,
    behavior_feature_dict,
    clip_predictions,
    default_output_path,
    labels_for_examples,
    load_examples,
    load_usefulness_scores,
    print_run_summary,
    save_predictions_excel,
)


BASELINE_NAME = "Click-Reformulation Heuristic"


def click_reformulation_rule_score(example, usefulness_by_id) -> float:
    features = behavior_feature_dict(example, usefulness_by_id)
    score = 4.0

    if features["trad_click_count"] > 0:
        score -= 0.9
        score -= min(math.log1p(features["trad_content_dwell_sum"]) / 3.0, 1.2)
    if features["has_next_query"] > 0:
        score -= 1.0
        if features["reformulation_jaccard"] < 0.35:
            score -= 0.7

    if features["trad_examination_count"] >= 4:
        score -= 0.4
    if features["aio_examination_dwell_sum"] >= 10.0 and features["trad_click_count"] == 0:
        score += 0.5
    if features["has_next_query"] <= 0 and features["trad_click_count"] == 0:
        score += 0.5
    score += (features["aio_sentence_usefulness_mean"] - 3.0) * 0.45
    return float(clip_predictions([score])[0])


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
    y_test = labels_for_examples(test_examples)
    predictions = [click_reformulation_rule_score(example, usefulness_by_id) for example in test_examples]
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
