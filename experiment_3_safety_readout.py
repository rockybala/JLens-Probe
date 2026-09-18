"""
Experiment 3: Safety-relevant J-Lens readout on alignment-audit prompts.

This is inspired by Section 5 of the Global Workspace paper:
  * strategic/evaluation-aware blackmail case study
  * prompt-injection recognition
  * fake vulnerability / fake-number editing / silent override / self-issued token
  * deception-token top-10 readout used for the reward-hacking model organism

The script is intentionally dataset-adaptive. It searches the local cloned
jacobian-lens tree for configured audit JSON filenames, extracts text/transcript
examples, and applies all three fitted lenses to exactly the same prompts.

Primary metric:
  count of configured deception-related tokens appearing in the J-Lens top-k,
  summed over selected layers and scored positions.

Secondary metric:
  an evaluation-awareness log-probability score inspired by Appendix A:
  mean log-prob of eval-awareness tokens minus mean log-prob of unrelated
  control tokens, averaged over selected layers/positions.
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
    SAFETY_DATASET_HINTS,
    SAFETY_DECEPTION_WORDS,
    EVAL_AWARENESS_WORDS,
    EVAL_CONTROL_WORDS,
    SAFETY_TOP_K,
    SAFETY_MAX_EXAMPLES_PER_DATASET,
    SAFETY_ANALYSIS_LAYERS,
)
from utils import (
    get_device,
    get_single_token_id,
    load_all_lenses,
    load_model,
    run_jlens,
)

ROOT = Path(__file__).resolve().parent
JLENS_ROOT = ROOT / "jacobian-lens"
RESULT_DIR = ROOT / "results" / "experiment_3"


def find_dataset_files():
    """Find locally available JSON files matching Section-5-style hints."""
    found = {}
    if not JLENS_ROOT.exists():
        raise FileNotFoundError(
            f"Expected cloned jacobian-lens repo at {JLENS_ROOT}"
        )

    json_files = list(JLENS_ROOT.rglob("*.json"))

    for label, hints in SAFETY_DATASET_HINTS.items():
        matches = []
        for path in json_files:
            low = str(path).lower().replace("_", "-")
            if any(h.lower().replace("_", "-") in low for h in hints):
                matches.append(path)
        if matches:
            # Prefer the shortest path/name match if duplicates exist.
            found[label] = sorted(matches, key=lambda p: (len(str(p)), str(p)))[0]

    return found


def messages_to_text(messages):
    """Convert [{role, content}, ...] into a stable plain-text transcript."""
    parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip()
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            parts.append(f"{role.capitalize()}: {content}" if role else content)
    return "\n\n".join(parts)


def extract_text_examples(obj, prefix="item"):
    """
    Extract prompt/transcript strings from several common JSON layouts.

    We deliberately avoid treating every arbitrary string in a JSON file as a
    prompt. Only prompt-like keys or message lists are accepted.
    """
    examples = []

    if isinstance(obj, list):
        # Direct list of strings.
        if all(isinstance(x, str) for x in obj):
            return [
                {"name": f"{prefix}-{i:03d}", "text": x}
                for i, x in enumerate(obj)
                if x.strip()
            ]

        # Chat message list.
        if obj and all(
            isinstance(x, dict) and "role" in x and "content" in x
            for x in obj
        ):
            text = messages_to_text(obj)
            return [{"name": prefix, "text": text}] if text else []

        # Otherwise recurse over item-like dictionaries.
        for i, x in enumerate(obj):
            examples.extend(extract_text_examples(x, f"{prefix}-{i:03d}"))
        return examples

    if not isinstance(obj, dict):
        return examples

    name = str(obj.get("name") or obj.get("id") or obj.get("slug") or prefix)

    # Preferred prompt/transcript keys.
    for key in ["prompt", "transcript", "conversation", "messages"]:
        if key not in obj:
            continue
        value = obj[key]
        if isinstance(value, str) and value.strip():
            examples.append({"name": name, "text": value})
            return examples
        if isinstance(value, list):
            text = messages_to_text(value)
            if text:
                examples.append({"name": name, "text": text})
                return examples

    # Common top-level collections.
    for key in ["items", "examples", "prompts", "cases", "scenarios"]:
        if key in obj:
            examples.extend(extract_text_examples(obj[key], prefix))

    return examples


def load_audit_examples(path, limit=None):
    with open(path, "r") as f:
        data = json.load(f)

    examples = extract_text_examples(data, prefix=path.stem)

    # Remove exact duplicates while preserving order.
    deduped = []
    seen = set()
    for ex in examples:
        if ex["text"] in seen:
            continue
        seen.add(ex["text"])
        deduped.append(ex)

    if limit is not None:
        deduped = deduped[:limit]

    return deduped


def resolve_token_ids(tokenizer, words):
    """Resolve only single-token words; return {word: token_id}."""
    result = {}
    for word in words:
        try:
            token_id, _ = get_single_token_id(tokenizer, word)
            result[word] = token_id
        except ValueError:
            print(f"Skipping multi-token analysis word: {word!r}")
    return result


def choose_layers(lenses):
    common = sorted(set.intersection(*(set(x.source_layers) for x in lenses.values())))
    if SAFETY_ANALYSIS_LAYERS is None:
        return common
    requested = [x for x in SAFETY_ANALYSIS_LAYERS if x in common]
    if not requested:
        raise ValueError("SAFETY_ANALYSIS_LAYERS has no overlap with fitted layers")
    return requested


def deception_topk_count(logits_by_layer, token_ids, top_k):
    """
    Count (layer, position, target-token) top-k hits.

    This mirrors the spirit of the paper's fixed deception-vocabulary top-10
    count while retaining per-position information for our smaller model.
    """
    target_ids = set(token_ids.values())
    total_hits = 0
    per_layer = {}

    for layer, logits in logits_by_layer.items():
        layer_hits = 0
        top_ids = logits.topk(top_k, dim=-1).indices
        for row in top_ids:
            row_ids = set(row.tolist())
            layer_hits += len(target_ids.intersection(row_ids))
        per_layer[layer] = layer_hits
        total_hits += layer_hits

    return total_hits, per_layer


def eval_awareness_score(logits_by_layer, eval_ids, control_ids):
    """
    Mean log-prob(eval tokens) - mean log-prob(control tokens), averaged over
    layers and positions. This follows the paper's proof-of-concept score in
    spirit, while allowing more than two eval-awareness tokens.
    """
    eval_token_ids = list(eval_ids.values())
    control_token_ids = list(control_ids.values())

    if not eval_token_ids or not control_token_ids:
        return float("nan"), {}

    per_layer = {}

    for layer, logits in logits_by_layer.items():
        log_probs = F.log_softmax(logits.float(), dim=-1)
        eval_mean = log_probs[:, eval_token_ids].mean().item()
        control_mean = log_probs[:, control_token_ids].mean().item()
        per_layer[layer] = eval_mean - control_mean

    return sum(per_layer.values()) / len(per_layer), per_layer


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_files = find_dataset_files()
    print("Found safety/audit datasets:")
    for name, path in dataset_files.items():
        print(f"  {name:20s} -> {path.relative_to(ROOT)}")

    if not dataset_files:
        raise FileNotFoundError(
            "No configured Section-5-style audit JSON files were found. "
            "See SAFETY_DATASET_HINTS in config.py and add the filenames "
            "present in your local jacobian-lens checkout."
        )

    device = get_device()
    print(f"\nUsing device: {device}")

    model, tokenizer = load_model(MODEL_NAME, device=device)
    lenses = load_all_lenses(
        root=ROOT,
        lens_repo=LENS_REPO,
        lens_file=LENS_FILE,
        lens_revision=LENS_REVISION,
        local_lens_files=LOCAL_LENS_FILES,
    )

    layers = choose_layers(lenses)
    print(f"Analysis layers: {layers}")

    deception_ids = resolve_token_ids(tokenizer, SAFETY_DECEPTION_WORDS)
    eval_ids = resolve_token_ids(tokenizer, EVAL_AWARENESS_WORDS)
    control_ids = resolve_token_ids(tokenizer, EVAL_CONTROL_WORDS)

    rows = []
    detailed = {}

    for dataset_name, path in dataset_files.items():
        examples = load_audit_examples(
            path,
            limit=SAFETY_MAX_EXAMPLES_PER_DATASET,
        )
        print(f"\n{dataset_name}: {len(examples)} examples")

        detailed[dataset_name] = {
            "path": str(path.relative_to(ROOT)),
            "examples": [],
        }

        for i, example in enumerate(examples):
            print(f"  [{i + 1}/{len(examples)}] {example['name']}")
            prompt = example["text"]

            example_out = {
                "name": example["name"],
                "n_chars": len(prompt),
                "lenses": {},
            }

            for lens_name, lens in lenses.items():
                # positions=None scores every prompt/transcript token. This is
                # useful for audit transcripts because the interesting signal
                # can arise before the final token.
                lens_logits, _, input_ids = run_jlens(
                    lens,
                    model,
                    prompt,
                    layers=layers,
                    positions=None,
                )

                deception_count, deception_by_layer = deception_topk_count(
                    lens_logits,
                    deception_ids,
                    SAFETY_TOP_K,
                )
                eval_score, eval_by_layer = eval_awareness_score(
                    lens_logits,
                    eval_ids,
                    control_ids,
                )

                n_positions = next(iter(lens_logits.values())).shape[0]
                normalized_deception = deception_count / (len(layers) * n_positions)

                row = {
                    "dataset": dataset_name,
                    "example": example["name"],
                    "lens": lens_name,
                    "n_positions": n_positions,
                    "deception_topk_count": deception_count,
                    "deception_hits_per_layer_position": normalized_deception,
                    "eval_awareness_score": eval_score,
                }
                rows.append(row)

                example_out["lenses"][lens_name] = {
                    **row,
                    "deception_by_layer": deception_by_layer,
                    "eval_awareness_by_layer": eval_by_layer,
                }

            detailed[dataset_name]["examples"].append(example_out)

    # Aggregate by dataset x lens.
    summaries = []
    for dataset_name in dataset_files:
        for lens_name in lenses:
            selected = [
                r for r in rows
                if r["dataset"] == dataset_name and r["lens"] == lens_name
            ]
            if not selected:
                continue
            summaries.append({
                "dataset": dataset_name,
                "lens": lens_name,
                "n_examples": len(selected),
                "mean_deception_hits_per_layer_position": sum(
                    r["deception_hits_per_layer_position"] for r in selected
                ) / len(selected),
                "mean_eval_awareness_score": sum(
                    r["eval_awareness_score"] for r in selected
                ) / len(selected),
            })

    with open(RESULT_DIR / "safety_readout_detailed.json", "w") as f:
        json.dump(detailed, f, indent=2)

    with open(RESULT_DIR / "safety_readout_rows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with open(RESULT_DIR / "safety_readout_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    print("\nSummary")
    for s in summaries:
        print(
            f"{s['dataset']:20s} {s['lens']:15s} "
            f"deception={s['mean_deception_hits_per_layer_position']:.5f} "
            f"eval-awareness={s['mean_eval_awareness_score']:.4f}"
        )

    print(f"\nSaved results to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
