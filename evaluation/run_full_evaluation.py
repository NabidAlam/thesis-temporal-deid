#!/usr/bin/env python3
"""
Run full evaluation for both SAMURAI and TSP-SAM on all DAVIS-2017 sequences.
This script discovers sequences automatically from input/davis2017/JPEGImages/480p
and writes consolidated CSV/JSON under evaluation/results/baseline_results/{method}/davis/.
"""

from pathlib import Path
from datetime import datetime
from evaluation.davis_baseline_eval import VideoEvaluator


def run_method(method: str) -> None:
    evaluator = VideoEvaluator(dataset_root="input", output_root="output", dataset="davis")
    images_dir = evaluator.images_dir
    sequences = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
    print(f"Discovered {len(sequences)} sequences for DAVIS evaluation")

    results = evaluator.evaluate_method(method, sequences)
    if not results:
        print(f"No results to save for method: {method}")
        return

    save_path = Path("evaluation/results/baseline_results")
    method_folder = save_path / method / "davis"
    method_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = method_folder / f"baseline_results_{method}_davis_{timestamp}"
    evaluator.save_results(results, final_path)


def main():
    for method in ["samurai", "tsp-sam"]:
        print(f"\n==== Running evaluation for {method.upper()} on DAVIS ====")
        run_method(method)


if __name__ == "__main__":
    main()


