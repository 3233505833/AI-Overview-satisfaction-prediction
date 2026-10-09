
import argparse
import importlib.util
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from baseline_common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_USEFULNESS_PATH,
    save_combined_workbook,
)


def load_baseline(file_name: str):
    module_name = file_name.removesuffix(".py").replace(" ", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, CURRENT_DIR / file_name)
    if spec is None or spec.loader is None:
        raise ImportError(file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


click_reformulation = load_baseline("Click-Reformulation Heuristic.py")
dwell_time = load_baseline("Dwell-time Heuristic.py")
ridge = load_baseline("Ridge.py")
lstm = load_baseline("LSTM.py")
random_forest = load_baseline("Random Forest.py")
mlp = load_baseline("MLP.py")
transformer = load_baseline("Transformer.py")
xgboost = load_baseline("GBDT.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all AI Overview satisfaction baselines.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--test", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--usefulness-jsonl", default=str(DEFAULT_USEFULNESS_PATH))
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    results = []
    results.append(dwell_time.run(args.train, args.test, args.output_dir, seed=args.seed, usefulness_jsonl=args.usefulness_jsonl))
    results.append(click_reformulation.run(args.train, args.test, args.output_dir, seed=args.seed, usefulness_jsonl=args.usefulness_jsonl))
    results.append(ridge.run(args.train, args.test, args.output_dir, seed=args.seed, usefulness_jsonl=args.usefulness_jsonl))
    results.append(random_forest.run(args.train, args.test, args.output_dir, seed=args.seed, usefulness_jsonl=args.usefulness_jsonl))
    results.append(mlp.run(args.train, args.test, args.output_dir, seed=args.seed, usefulness_jsonl=args.usefulness_jsonl))
    results.append(xgboost.run(args.train, args.test, args.output_dir, seed=args.seed, usefulness_jsonl=args.usefulness_jsonl))
    results.append(lstm.run(args.train, args.test, args.output_dir, seed=args.seed, usefulness_jsonl=args.usefulness_jsonl))
    results.append(transformer.run(args.train, args.test, args.output_dir, seed=args.seed, usefulness_jsonl=args.usefulness_jsonl))

    combined = save_combined_workbook(results, Path(args.output_dir) / "all_baseline_predictions.xlsx")
    print(f"combined_output={combined}")


if __name__ == "__main__":
    main()
