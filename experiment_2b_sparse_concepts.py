"""
Experiment 2b: Sparse J-Lens concept recovery.

Question
--------
For the SAME model residual activation h_l, do the three fitted J-Lenses
identify the same sparse set of concepts?

We use the multihop examples from Experiment 1.  At the readout position,
for each layer and lens we build J-Lens concept directions

    d_v,l = J_l^T w_v

and greedily reconstruct h_l with a non-negative sparse combination of K
concept directions.  We then ask only one simple question: which token concepts
were selected, and does the known intermediate concept appear among them?

Practical note
--------------
The paper does not release its exact gradient-pursuit implementation. This
script therefore implements a simple full-vocabulary non-negative greedy
pursuit. At every step it scores ALL token-indexed J-Lens vectors against the
current residual, selects one concept, refits non-negative coefficients for the
selected vectors, and repeats. There is no vocabulary pre-filter.

This follows the paper's sparse-decomposition objective, but should be described
as an approximate implementation rather than an exact reproduction of
Anthropic's unpublished gradient-pursuit code.
"""

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
    EVAL_FILE,
)
from utils import (
    get_device,
    get_readout_position,
    get_single_token_id,
    load_all_lenses,
    load_data,
    load_model,
)
from jlens.hooks import ActivationRecorder


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "jacobian-lens" / "data" / "evaluations"
RESULT_DIR = ROOT / "results" / "experiment_2b"

# Keep this deliberately small/clean for the first run.
# Set to None after the pilot if you want the complete Experiment-1 dataset.
N_EXAMPLES = 10

# The paper uses K=25 for sparse decompositions of intermediate probes.
K = 25

# Focus on the late-middle region where Experiment 1 recovered concepts well.
# Set to None to use every layer shared by the three lenses.
LAYERS = list(range(18, 29))

# Small projected-gradient refit of the non-negative coefficients after each
# newly selected concept.
REFIT_STEPS = 40


def capture_residuals(model, prompt, layers, position):
    """Capture the model's residual activation at one position for each layer."""
    input_ids = model.encode(prompt)

    with torch.no_grad():
        with ActivationRecorder(model.layers, at=layers) as recorder:
            model.forward(input_ids)

    residuals = {}
    for layer in layers:
        # [batch, sequence, d_model] -> [d_model]
        residuals[layer] = (
            recorder.activations[layer][0, position]
            .detach()
            .float()
            .cpu()
        )

    return residuals


def concept_direction(lens, layer, unembedding_vector):
    """Return d_v,l = J_l^T w_v using row-vector notation w_v @ J_l."""
    J = lens.jacobians[layer].float().cpu()
    return unembedding_vector.float().cpu() @ J


def refit_nonnegative_coefficients(atoms, target, steps=40):
    """
    Fit non-negative coefficients for selected unit-norm atoms.

    atoms:  [n_selected, d_model]
    target: [d_model]
    """
    # Least-squares initialization followed by projection to alpha >= 0.
    # n_selected <= 25, so this is cheap.
    try:
        alpha = torch.linalg.lstsq(atoms.T, target).solution
    except RuntimeError:
        alpha = torch.zeros(atoms.shape[0], dtype=torch.float32)
    alpha = alpha.clamp_min(0)

    # Projected gradient descent refines the constrained solution.
    gram = atoms @ atoms.T
    rhs = atoms @ target
    # Lipschitz constant of the quadratic gradient.
    L = torch.linalg.eigvalsh(gram).max().item()
    lr = 1.0 / max(L, 1e-6)

    for _ in range(steps):
        grad = gram @ alpha - rhs
        alpha = (alpha - lr * grad).clamp_min(0)

    return alpha


def sparse_concepts(lens, layer, h, unembedding, tokenizer, k=25):
    """
    Full-vocabulary greedy non-negative pursuit over J-Lens vectors.

    For token v, the J-Lens vector is

        d_v = J_l^T w_v

    (equivalently, row v of W_U J_l). For the current residual r,

        <d_v, r> = w_v^T J_l r.

    Therefore we can score every vocabulary concept without materializing the
    full [vocab_size, d_model] J-Lens dictionary. Only the selected token's
    direction is constructed explicitly.
    """
    J = lens.jacobians[layer].float().cpu()
    W = unembedding.float().cpu()
    target = h.float().cpu()
    residual = target.clone()

    selected_ids = []
    selected_atoms = []
    alpha = None

    for _ in range(k):
        # Full-vocabulary search.
        # scores[v] = <J_l^T w_v, residual> = w_v^T (J_l residual)
        transported_residual = J @ residual
        scores = W @ transported_residual

        if selected_ids:
            scores[selected_ids] = -torch.inf

        best_id = int(torch.argmax(scores).item())
        best_score = float(scores[best_id].item())
        if not torch.isfinite(scores[best_id]) or best_score <= 0:
            break

        # Materialize only the selected J-Lens vector.
        # Row-vector form of J_l^T w_v is w_v @ J_l.
        atom = W[best_id] @ J

        selected_ids.append(best_id)
        selected_atoms.append(atom)

        atoms = torch.stack(selected_atoms, dim=0)  # [n_selected, d_model]
        alpha = refit_nonnegative_coefficients(
            atoms, target, steps=REFIT_STEPS
        )

        reconstruction = alpha @ atoms
        residual = target - reconstruction

    concepts = []
    for i, token_id in enumerate(selected_ids):
        concepts.append({
            "token_id": int(token_id),
            "token": tokenizer.decode([token_id]),
            "coefficient": float(alpha[i].item()) if alpha is not None else 0.0,
        })

    return concepts

def intermediate_is_selected(tokenizer, intermediate, concepts):
    """Check whether Experiment-1's known single-token intermediate was selected."""
    token_id, token_text = get_single_token_id(tokenizer, intermediate)
    selected_ids = {c["token_id"] for c in concepts}
    return token_id in selected_ids, token_id, token_text


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

    unembedding = model._lm_head.weight.detach().float().cpu()

    items = load_data(DATA_DIR / EVAL_FILE, n_examples=N_EXAMPLES)

    common_layers = sorted(
        set.intersection(*(set(lens.source_layers) for lens in lenses.values()))
    )
    layers = common_layers if LAYERS is None else [
        layer for layer in LAYERS if layer in common_layers
    ]

    print(f"Using layers: {layers}")
    print(f"Sparse concepts per activation: K={K}")
    print("Vocabulary search: FULL vocabulary at every pursuit step")

    results = []

    for item_index, item in enumerate(items):
        print("\n" + "=" * 80)
        print(f"[{item_index + 1}/{len(items)}] {item['name']}")
        print(f"Prompt: {item['prompt']}")
        print(f"Known intermediate(s): {item['intermediates']}")

        position = get_readout_position(
            tokenizer, item["prompt"], item["target"]
        )

        # Capture h_l ONCE. Every lens sees exactly the same model activation.
        residuals = capture_residuals(
            model, item["prompt"], layers, position
        )

        item_result = {
            "name": item["name"],
            "prompt": item["prompt"],
            "target": item["target"],
            "intermediates": item["intermediates"],
            "readout_position": position,
            "layers": {},
        }

        for layer in layers:
            print(f"\n  Layer {layer}")
            h = residuals[layer]
            item_result["layers"][str(layer)] = {}

            for lens_name, lens in lenses.items():
                concepts = sparse_concepts(
                    lens, layer, h, unembedding, tokenizer, k=K
                )

                recovered = {}
                for intermediate in item["intermediates"]:
                    try:
                        found, token_id, token_text = intermediate_is_selected(
                            tokenizer, intermediate, concepts
                        )
                        recovered[intermediate] = {
                            "selected": found,
                            "token_id": token_id,
                            "token_text": token_text,
                        }
                    except ValueError as error:
                        recovered[intermediate] = {
                            "selected": False,
                            "reason": str(error),
                        }

                item_result["layers"][str(layer)][lens_name] = {
                    "concepts": concepts,
                    "intermediate_recovery": recovered,
                }

                concept_text = ", ".join(
                    repr(c["token"]) for c in concepts
                )
                recovered_text = ", ".join(
                    f"{name}={'YES' if info['selected'] else 'no'}"
                    for name, info in recovered.items()
                )

                print(f"    {lens_name:>14}: {concept_text}")
                print(f"                    intermediate: {recovered_text}")

        results.append(item_result)

    output_file = RESULT_DIR / "sparse_concepts.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # One compact final summary: for each known intermediate, did it appear at
    # ANY tested layer for each lens?
    print("\n" + "=" * 80)
    print("KNOWN INTERMEDIATE RECOVERY IN SPARSE CONCEPT SETS")
    print("=" * 80)

    for item in results:
        print(f"\n{item['name']}")
        for intermediate in item["intermediates"]:
            pieces = []
            for lens_name in lenses:
                found_layers = []
                for layer, layer_data in item["layers"].items():
                    info = layer_data[lens_name]["intermediate_recovery"].get(
                        intermediate, {}
                    )
                    if info.get("selected", False):
                        found_layers.append(int(layer))

                if found_layers:
                    pieces.append(
                        f"{lens_name}=YES (L{min(found_layers)}-L{max(found_layers)})"
                    )
                else:
                    pieces.append(f"{lens_name}=no")

            print(f"  {intermediate!r}: " + " | ".join(pieces))

    print(f"\nSaved detailed concepts to: {output_file}")


if __name__ == "__main__":
    main()
