"""
Experiment 2: Vocabulary-linked J-Lens direction alignment.

For token v and layer l, the J-Lens direction in the source residual space is

    d_{v,l} = J_l^T w_v

where w_v is the model's LM-head / unembedding vector for token v.

We compare the same token direction across fitted lenses with cosine similarity.
The important baseline is Anthropic-1000 vs WikiText-100; this tells us how much
variation to expect from fitting on fewer samples. We then compare that with
WikiText-100 vs Deception-100.
"""

import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from config import (
    MODEL_NAME,
    LENS_REPO,
    LENS_REVISION,
    LENS_FILE,
    LOCAL_LENS_FILES,
    DIRECTION_TOKEN_GROUPS,
)
from utils import (
    get_device,
    get_single_token_id,
    load_all_lenses,
    load_model,
)

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "experiment_2"


def get_token_specs(tokenizer):
    """Resolve configured words to one-token vocabulary entries."""
    specs = []
    seen_ids = set()

    for group, words in DIRECTION_TOKEN_GROUPS.items():
        for word in words:
            try:
                token_id, token_text = get_single_token_id(tokenizer, word)
            except ValueError:
                print(f"Skipping multi-token word: {word!r}")
                continue

            # A word can occur in more than one conceptual group. Keep one row per
            # (group, token) but avoid accidental duplicates inside a group.
            key = (group, token_id)
            if key in seen_ids:
                continue
            seen_ids.add(key)

            specs.append({
                "group": group,
                "word": word,
                "token_text": token_text,
                "token_id": token_id,
            })

    return specs


def direction_for_token(lens, layer, unembedding_vector):
    """
    Return d = J_l^T w_v.

    With row-vector notation this is equivalently w_v @ J_l.
    """
    J = lens.jacobians[layer].float()
    w = unembedding_vector.float().cpu()
    return w @ J


def cosine(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    print("Loading model...")
    model, tokenizer = load_model(MODEL_NAME, device=device)

    print("Loading lenses...")
    lenses = load_all_lenses(
        root=ROOT,
        lens_repo=LENS_REPO,
        lens_file=LENS_FILE,
        lens_revision=LENS_REVISION,
        local_lens_files=LOCAL_LENS_FILES,
    )

    # HFLensModel exposes the actual HF LM head used by model.unembed().
    # We only need its weight rows, so move the selected vectors to CPU.
    unembedding = model._lm_head.weight.detach().float().cpu()

    token_specs = get_token_specs(tokenizer)
    print(f"Resolved {len(token_specs)} single-token analysis words.")

    lens_pairs = [
        ("anthropic_1000", "wiki_100"),
        ("anthropic_1000", "deception_100"),
        ("wiki_100", "deception_100"),
    ]

    # Only compare layers present in every lens.
    common_layers = sorted(
        set.intersection(*(set(l.source_layers) for l in lenses.values()))
    )

    rows = []

    for layer in common_layers:
        print(f"Layer {layer}")

        for spec in token_specs:
            w = unembedding[spec["token_id"]]

            directions = {
                lens_name: direction_for_token(lens, layer, w)
                for lens_name, lens in lenses.items()
            }

            for lens_a, lens_b in lens_pairs:
                rows.append({
                    "layer": layer,
                    "group": spec["group"],
                    "word": spec["word"],
                    "token_text": spec["token_text"],
                    "token_id": spec["token_id"],
                    "lens_a": lens_a,
                    "lens_b": lens_b,
                    "cosine_similarity": cosine(
                        directions[lens_a], directions[lens_b]
                    ),
                })

    # Per-layer summaries, both overall and by conceptual group.
    summaries = []
    groups = sorted(set(r["group"] for r in rows))

    for layer in common_layers:
        for lens_a, lens_b in lens_pairs:
            pair_rows = [
                r for r in rows
                if r["layer"] == layer
                and r["lens_a"] == lens_a
                and r["lens_b"] == lens_b
            ]

            for group in ["all"] + groups:
                selected = pair_rows if group == "all" else [
                    r for r in pair_rows if r["group"] == group
                ]
                if not selected:
                    continue

                values = [r["cosine_similarity"] for r in selected]
                summaries.append({
                    "layer": layer,
                    "group": group,
                    "lens_a": lens_a,
                    "lens_b": lens_b,
                    "n_tokens": len(values),
                    "mean_cosine": sum(values) / len(values),
                    "min_cosine": min(values),
                    "max_cosine": max(values),
                })

    # Save machine-readable outputs.
    with open(RESULT_DIR / "direction_alignment_detailed.json", "w") as f:
        json.dump(rows, f, indent=2)

    with open(RESULT_DIR / "direction_alignment_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    with open(RESULT_DIR / "direction_alignment_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    print("\nOverall layer-wise mean cosine similarity")
    for s in summaries:
        if s["group"] != "all":
            continue
        print(
            f"L{s['layer']:02d}  {s['lens_a']:>14} vs {s['lens_b']:<14} "
            f"mean cos={s['mean_cosine']:.4f}"
        )

    print(f"\nSaved results to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
