# experiment_1_intermediate_recovery.py

"""
Experiment 1: Intermediate Concept Recovery
"""

import json
from pathlib import Path

from upstream_data import get_evaluation_dir
from config import (
    MODEL_NAME,
    LENS_REPO,
    LENS_REVISION,
    LENS_FILE,
    LOCAL_LENS_FILES,
    EVAL_NAME, 
    EVAL_FILE,
)

from utils import (
    load_model,
    load_lens,
    load_local_lens,
    load_data,
    load_all_lenses,
    get_device,
    run_jlens,
    token_rank,
    get_single_token_id,
    save_slice,
    get_readout_position,
)

import torch
import jlens
from jlens import JacobianLens

# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = get_evaluation_dir()
RESULT_DIR = ROOT / "results" / "experiment_1"

# Start small while debugging.
# Set to None later to run the complete dataset.
N_EXAMPLES = None #5
K_VALUES = [1, 5, 10, 20, 50, 80, 100]


# ============================================================
# Evaluate one example
# ============================================================
def evaluate_example(model, tokenizer, lens, item):

    prompt = item["prompt"]
    target = item["target"]

    readout_position = get_readout_position(tokenizer, prompt, target)
    print("readout position")
    print(readout_position)

    # Run J-Lens ONCE for this prompt.
    lens_logits, _, _ = run_jlens(
        lens,
        model,
        prompt,
        positions=[readout_position]
    )
    
    results = []

    for intermediate in item["intermediates"]:

        token_id, token_text = get_single_token_id(
            tokenizer,
            intermediate,
        )

        layer_ranks = {}

        for layer, logits in lens_logits.items():

            rank = token_rank(
                logits[0],
                token_id,
            )

            layer_ranks[layer] = rank

        best_layer = min(
            layer_ranks,
            key=layer_ranks.get,
        )

        best_rank = layer_ranks[best_layer]

        results.append(
            {
                "intermediate": intermediate,
                "token_text": token_text,
                "token_id": token_id,
                "best_rank": best_rank,
                "best_layer": best_layer,
                "layer_ranks": layer_ranks,
            }
        )

    return results

# ============================================================
# Evaluate one lens
# ============================================================

def evaluate_lens(model, tokenizer, lens, items, lens_name):

    print()
    print("=" * 70)
    print(f"Evaluating lens: {lens_name}")
    print("=" * 70)

    all_results = []

    for i, item in enumerate(items):

        print()
        print(f"[{i + 1}/{len(items)}] {item['name']}")
        print(f"Prompt: {item['prompt']}")
        print(f"Target: {item['target']}")
        print(f"Intermediate(s): {item['intermediates']}")

        try:

            intermediate_results = evaluate_example(
                model,
                tokenizer,
                lens,
                item,
            )

        except ValueError as error:

            print(f"SKIPPED: {error}")

            all_results.append(
                {
                    "name": item["name"],
                    "prompt": item["prompt"],
                    "target": item.get("target"),
                    "status": "skipped",
                    "reason": str(error),
                }
            )

            continue

        for result in intermediate_results:

            print(
                f"  {result['intermediate']!r}"
                f" -> token {result['token_text']!r}"
                f" (id={result['token_id']})"
            )

            print(
                f"     best rank = {result['best_rank']}"
                f" at layer {result['best_layer']}"
            )

            print("     layer ranks:")

            for layer, rank in result["layer_ranks"].items():
                print(
                    f"       L{layer:02d}: {rank}"
                )

        all_results.append(
            {
                "name": item["name"],
                "prompt": item["prompt"],
                "target": item.get("target"),
                "status": "ok",
                "intermediates": intermediate_results,
            }
        )

    return all_results


# ============================================================
# pass@k
# ============================================================
def compute_pass_at_k(results):
    """
    Compute pass@k over all successfully evaluated intermediates.

    Each intermediate counts as recovered if:
        min rank over layers <= k
    """

    best_ranks = []

    for item in results:

        if item["status"] != "ok":
            continue

        for intermediate in item["intermediates"]:
            best_ranks.append(
                intermediate["best_rank"]
            )

    metrics = {}

    for k in K_VALUES:

        if len(best_ranks) == 0:
            metrics[k] = float("nan")
        else:
            metrics[k] = sum(
                rank <= k
                for rank in best_ranks
            ) / len(best_ranks)

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device = get_device()
    print(f"\nUsing device: {device}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    print("\nLoading model...")

    model, tokenizer = load_model(
        MODEL_NAME,
        device=device,
    )
    print("Model loaded.")

    # --------------------------------------------------------
    # Load lenses
    # --------------------------------------------------------
    print("\nLoading lenses...")
    lenses = load_all_lenses(
        root=ROOT,
        lens_repo=LENS_REPO,
        lens_file=LENS_FILE,
        lens_revision=LENS_REVISION,
        local_lens_files=LOCAL_LENS_FILES,
    )
    print(
        f"Loaded {len(lenses)} lenses: "
        f"{list(lenses.keys())}"
    )

    # --------------------------------------------------------
    # Load evaluation data
    # --------------------------------------------------------
    data_file = DATA_DIR / EVAL_FILE

    items = load_data(
        data_file,
        n_examples=N_EXAMPLES,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    experiment_results = {}

    for lens_name, lens in lenses.items():
        results = evaluate_lens(
            model,
            tokenizer,
            lens,
            items,
            lens_name,
        )

        metrics = compute_pass_at_k(
            results
        )

        experiment_results[lens_name] = {
            "results": results,
            "pass_at_k": metrics,
        }

        print(f"\npass@k for {lens_name}")

        for k, value in metrics.items():
            print(
                f"  pass@{k:<3} = {value:.3f}"
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_file = (
        RESULT_DIR
        / f"{EVAL_NAME}_debug_results.json"
    )

    with open(output_file, "w") as f:
        json.dump(
            experiment_results,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Experiment complete")
    print("=" * 70)

    print(
        f"\nEvaluation: {EVAL_NAME}"
    )

    print(
        f"Results saved to:\n{output_file}"
    )

    print("\nSummary:")

    for lens_name, result in experiment_results.items():

        print(f"\n{lens_name}")

        for k, value in result["pass_at_k"].items():

            print(
                f"  pass@{k:<3} = {value:.3f}"
            )

if __name__ == "__main__":
    main()