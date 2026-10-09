import json
import re
from os.path import dirname, join

format_penalty = -1.0
outcome_weight = 0.7
p_weight = 0.3
max_distance = 4


def extract_solution(solution_str):
    match = re.search(
        r'<answer>\s*(\[\s*(?:"[+\-0]"\s*(?:,\s*"[+\-0]"\s*)*)?\])\s*###\s*([1-5])\s*</answer>',
        solution_str,
        flags=re.S,
    )
    if match is None:
        return None
    return {
        "intermediate_labels": json.loads(match.group(1)),
        "final_score": int(match.group(2)),
    }


def compute_score(solution_str, ground_truth, extra_info=None):
    with open(join(dirname(__file__), "reference_signal_labels.json"), encoding="utf-8") as file:
        labels_by_query_id = json.load(file)
    query_id = str(extra_info["query_id"])
    intermediate_labels = [str(value) for value in labels_by_query_id[query_id]]

    parsed = extract_solution(solution_str)
    if parsed is None:
        return format_penalty

    try:
        y_pred = parsed["final_score"]
        y_true = int(ground_truth)
    except (TypeError, ValueError):
        return format_penalty
    if y_true < 1 or y_true > 5:
        return format_penalty

    predicted_labels = [str(value) for value in parsed["intermediate_labels"]]
    if len(predicted_labels) != len(intermediate_labels):
        return format_penalty

    distance = abs(y_pred - y_true)
    outcome_reward = max(0.0, 1.0 - distance / max_distance)

    selected = [i for i, polarity in enumerate(intermediate_labels) if polarity != "0"]
    process_reward = (
        sum(predicted_labels[i] == intermediate_labels[i] for i in selected) / len(selected)
        if selected else 0.0
    )
    return outcome_weight * outcome_reward + p_weight * process_reward
