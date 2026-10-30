"""Regenerate the three README figures from committed result files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LABELS = {
    "wikitext_1000": "WikiText-1000",
    "wiki_100": "WikiText-100",
    "deception_100": "Deception-100",
}


def plot_experiment_1() -> None:
    path = RESULTS / "experiment_1" / "multihop_debug_results.json"
    data = json.loads(path.read_text())
    fig, ax = plt.subplots(figsize=(10, 6.4))
    for key in ["wikitext_1000", "wiki_100", "deception_100"]:
        metrics = data[key]["pass_at_k"]
        xs = sorted(int(k) for k in metrics)
        ys = [metrics[str(k)] if str(k) in metrics else metrics[k] for k in xs]
        ax.plot(xs, ys, marker="o", label=LABELS[key])
    ax.set_title("Experiment 1: Multihop intermediate-concept recovery")
    ax.set_xlabel("Rank threshold k")
    ax.set_ylabel("Intermediate recovery")
    ax.set_ylim(0, 0.9)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "experiment1.png", dpi=160)
    plt.close(fig)


def plot_experiment_2() -> None:
    path = RESULTS / "experiment_2" / "direction_alignment_summary.json"
    rows = json.loads(path.read_text())
    pairs = [
        ("wikitext_1000", "wiki_100"),
        ("wikitext_1000", "deception_100"),
        ("wiki_100", "deception_100"),
    ]
    fig, ax = plt.subplots(figsize=(10, 6.4))
    for a, b in pairs:
        selected = [
            r for r in rows
            if r["group"] == "all" and r["lens_a"] == a and r["lens_b"] == b
        ]
        selected.sort(key=lambda r: r["layer"])
        ax.plot(
            [r["layer"] for r in selected],
            [r["mean_cosine"] for r in selected],
            marker="o",
            label=f"{LABELS[a]} ↔ {LABELS[b]}",
        )
    ax.set_title("Experiment 2: J-Lens direction alignment across layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "experiment2.png", dpi=160)
    plt.close(fig)


def plot_experiment_2b() -> None:
    path = RESULTS / "experiment_2b" / "sparse_concepts.json"
    items = json.loads(path.read_text())
    keys = ["wikitext_1000", "wiki_100", "deception_100"]
    example_hits = {k: 0 for k in keys}
    layer_hits = {k: 0 for k in keys}
    layer_cases = {k: 0 for k in keys}

    for item in items:
        hit_in_example = {k: False for k in keys}
        for layer_data in item["layers"].values():
            for key in keys:
                if key not in layer_data:
                    continue
                recovered = layer_data[key].get("intermediate_recovery", {})
                hit = any(v.get("selected", False) for v in recovered.values())
                layer_cases[key] += 1
                if hit:
                    layer_hits[key] += 1
                    hit_in_example[key] = True
        for key in keys:
            example_hits[key] += int(hit_in_example[key])

    fig, ax = plt.subplots(figsize=(9, 6.4))
    values = [example_hits[k] for k in keys]
    bars = ax.bar([LABELS[k] for k in keys], values)
    ax.set_title("Experiment 2b: Sparse J-space concept recovery")
    ax.set_ylabel("Examples with intermediate recovered")
    ax.set_ylim(0, max(10, len(items)))
    ax.grid(axis="y", alpha=0.25)
    for bar, key in zip(bars, keys):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{example_hits[key]}/{len(items)}\n"
            f"({layer_hits[key]}/{layer_cases[key]} layer-cases)",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(RESULTS / "experiment2b.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    plot_experiment_1()
    plot_experiment_2()
    plot_experiment_2b()
    print("Regenerated results/experiment1.png, experiment2.png, and experiment2b.png")
