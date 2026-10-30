"""Fit the two local 100-example J-Lenses used by the experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jlens
from jlens.examples import load_wikitext_prompts

import config
from utils import get_device, load_model

ROOT = Path(__file__).resolve().parent
DECEPTION_DATA = ROOT / "data" / "deception_prompts_100_jlens.json"
OUTPUT_DIR = ROOT / "Fitted-Jlens"


def load_deception_prompts(n_prompts: int) -> list[str]:
    with open(DECEPTION_DATA, "r") as f:
        prompts = json.load(f)
    if not isinstance(prompts, list) or not all(isinstance(x, str) for x in prompts):
        raise ValueError(f"Expected a JSON list of strings in {DECEPTION_DATA}")
    if len(prompts) < n_prompts:
        raise ValueError(
            f"Requested {n_prompts} deception prompts, but only {len(prompts)} are available."
        )
    return prompts[:n_prompts]


def fit_one(model, prompts: list[str], output_path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = output_path.with_name(output_path.stem + "_checkpoint.pt")

    print(f"\nFitting {output_path.name} on {len(prompts)} prompts...")
    start = time.perf_counter()
    lens = jlens.fit(
        model,
        prompts=prompts,
        dim_batch=4,
        max_seq_len=128,
        checkpoint_path=str(checkpoint),
    )
    lens.save(str(output_path))
    elapsed = time.perf_counter() - start
    print(f"Saved: {output_path}")
    print(f"Elapsed: {elapsed / 60:.2f} minutes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        choices=["wikitext", "deception", "both"],
        default="both",
        help="Which local lens(es) to fit.",
    )
    parser.add_argument("--n-prompts", type=int, default=100)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    model, _ = load_model(config.MODEL_NAME, device=device)

    if args.corpus in {"wikitext", "both"}:
        prompts = load_wikitext_prompts(n_prompts=args.n_prompts)
        fit_one(model, prompts, ROOT / config.LOCAL_LENS_FILES["wiki_100"])

    if args.corpus in {"deception", "both"}:
        prompts = load_deception_prompts(args.n_prompts)
        fit_one(model, prompts, ROOT / config.LOCAL_LENS_FILES["deception_100"])


if __name__ == "__main__":
    main()
