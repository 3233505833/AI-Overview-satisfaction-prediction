import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from model_f import f_ref, f_ref_j



def second(candidates, y):
    retained = [trace[:-1] for trace in candidates if trace[-1] == y]
    voted = []
    for i, column in enumerate(zip(*retained), start=1):
        polarity, count = Counter(column).most_common(1)[0]
        if count > len(retained) / 2:
            voted.append({"signal_index": i, "polarity": polarity})
    return voted


def third(input_path, voted_path, num_signals, f_ref, f_ref_j):
    voted = json.loads(Path(voted_path).read_text(encoding="utf-8"))
    result = {}
    with TemporaryDirectory() as folder:
        full_path = Path(folder) / "full_signals.json"
        for query_id, judgments in voted.items():
            labels = ["0"] * num_signals[query_id]
            if judgments:
                full_path.write_text(json.dumps({query_id: judgments}), encoding="utf-8")
                baseline = f_ref(str(input_path), str(full_path))
                for judgment in judgments:
                    j = judgment["signal_index"]
                    removed_score = f_ref_j(str(input_path), str(full_path), j)
                    delta = abs(removed_score - baseline)
                    if delta > 0 and judgment["polarity"] != "0":
                        labels[judgment["signal_index"] - 1] = judgment["polarity"]
            result[query_id] = labels
    return result


if __name__ == "__main__":
    folder = Path(__file__).parent
    input_path = folder / "input.jsonl"
    candidate_path = folder / "candidate_signal_traces.json"
    voted_path = folder / "voted_signals.json"#output1
    output_path = folder / "reference_signal_labels.json"#output2

    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    ground_truth = {row["extra_info"]["query_id"]: row["reward_model"]["ground_truth"] for row in rows}
    voted = {query_id: second(traces, ground_truth[query_id]) for query_id, traces in candidates.items()}
    voted_path.write_text(json.dumps(voted, ensure_ascii=False, indent=2), encoding="utf-8")

    num_signals = {query_id: len(traces[0]) - 1 for query_id, traces in candidates.items()}
    result = third(input_path, voted_path, num_signals, f_ref, f_ref_j)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")