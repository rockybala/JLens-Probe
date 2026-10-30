from pathlib import Path

import jlens
import torch
import transformers
import json
from jlens.vis import build_page, compute_slice

# ------------------------------------------------------------------
# Model / lens loading
# ------------------------------------------------------------------
def load_model(model_name, dtype=torch.bfloat16, device=None):
    """
    Load a Hugging Face causal language model and wrap it for J-Lens.
    """

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
    )

    hf_model = hf_model.to(device)
    hf_model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    model = jlens.from_hf(hf_model, tokenizer)

    return model, tokenizer

# Load a pretrained J-Lens checkpoint
def load_lens(lens_repo, lens_file, revision=None):
    """
    Load a pre-fitted Jacobian Lens from Hugging Face or a local path.
    """
    lens = jlens.JacobianLens.from_pretrained(
        lens_repo,
        filename=lens_file,
        revision=revision,
    )
    return lens

##loading locally fitted lenses
def load_local_lens(lens_path):
    """
    Load a locally fitted Jacobian Lens.
    """
    lens_path = Path(lens_path)

    if not lens_path.exists():
        raise FileNotFoundError(f"Lens not found: {lens_path}")

    return jlens.JacobianLens.load(str(lens_path))

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

##rank of the target token among all the logits
def token_rank(logits, token_id):
    """
    Return the 1-indexed rank of a vocabulary token.
    Rank 1 means the token has the highest logit.
    """
    target_logit = logits[token_id]

    ##sum of all logits with magnitude higher than target, so target rank is sum + 1
    return (
        int((logits > target_logit).sum().item())
        + 1
    )

#Load data for evaluating the lens
def load_data(data_file, n_examples=None):
    """
    Load evaluation items from a J-Lens evaluation JSON file.
    Parameters
    ----------
    data_file : str or Path
        Path to the evaluation JSON file.

    n_examples : int or None
        Number of examples to load.
        If None, load the complete dataset.
    """

    with open(data_file, "r") as f:
        data = json.load(f)

    items = data["items"]

    if n_examples is not None:
        items = items[:n_examples]

    print(f"Loaded {len(items)} examples from {data_file.name}")

    return items

#load all lenses
def load_all_lenses(
    root,
    lens_repo,
    lens_file,
    lens_revision,
    local_lens_files,
):
    """
    Load the pretrained WikiText-1000 lens and locally fitted lenses.
    """
    lenses = {}

    # Pretrained WikiText N=1000 lens
    lenses["wikitext_1000"] = load_lens(
        lens_repo,
        lens_file,
        revision=lens_revision,
    )

    # Locally fitted lenses
    for lens_name, lens_path in local_lens_files.items():
        lenses[lens_name] = load_local_lens(
            root / lens_path
        )

    return lenses

def get_single_token_id(tokenizer, word):
    """
    Find a single-token vocabulary representation of a word.

    Try both:
        "word"
        " word"

    Returns
    -------
    token_id : int
    token_text : str
    """
    '''
    candidates = [
        word,
        " " + word,
    ]
    '''
    candidates = [
            " " + word,
            word,
    ]

    for candidate in candidates:

        token_ids = tokenizer.encode(
            candidate,
            add_special_tokens=False,
        )

        if len(token_ids) == 1:
            return token_ids[0], candidate

    raise ValueError(
        f"Intermediate {word!r} is not a single token "
        f"under either tested representation."
    )

# ------------------------------------------------------------------
# Token inspection
# ------------------------------------------------------------------
def show_tokens(tokenizer, prompt):
    """
    Print token indices, token IDs, and decoded token strings.
    """

    token_ids = tokenizer.encode(prompt)

    for i, token_id in enumerate(token_ids):
        token = tokenizer.decode([token_id])
        print(f"{i:>3}  {token_id:>8}  {repr(token)}")

#get last token position 
def get_readout_position(tokenizer, prompt, target):
    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    full_ids = tokenizer.encode(
        prompt + target,
        add_special_tokens=False,
    )

    # Find longest common token prefix
    n_common = 0
    for p_id, f_id in zip(prompt_ids, full_ids):
        if p_id != f_id:
            break
        n_common += 1

    # Last token before the target begins
    readout_index = n_common - 1

    # Convert absolute index to index within prompt,
    # e.g. -1, -2, ...
    position = readout_index - len(prompt_ids)

    return position

# ------------------------------------------------------------------
# J-Lens / Logit Lens
# ------------------------------------------------------------------
def run_jlens(
    lens,
    model,
    prompt,
    layers=None,
    positions=None,
):
    """
    Apply the Jacobian Lens.
    """
    jlens_logits, model_logits, info = lens.apply(
        model,
        prompt,
        layers=layers,
        positions=positions,
    )

    return jlens_logits, model_logits, info


def run_logit_lens(
    lens,
    model,
    prompt,
    layers=None,
    positions=None,
):
    """
    Apply the ordinary Logit Lens using the same interface,
    but without Jacobian transport.
    """

    logit_lens_logits, model_logits, info = lens.apply(
        model,
        prompt,
        layers=layers,
        positions=positions,
        use_jacobian=False,
    )

    return logit_lens_logits, model_logits, info


# ------------------------------------------------------------------
# Decode lens outputs
# ------------------------------------------------------------------

def top_tokens(logits, tokenizer, k=5):
    """
    Return the top-k decoded tokens for a vocabulary-logit vector.
    """

    top_ids = logits.topk(k).indices.tolist()

    return [
        {
            "token_id": token_id,
            "token": tokenizer.decode([token_id]),
            "logit": logits[token_id].item(),
        }
        for token_id in top_ids
    ]


def print_top_tokens(logits, tokenizer, k=5):
    """
    Print top-k tokens and their logits.
    """

    for item in top_tokens(logits, tokenizer, k):
        print(
            f"{repr(item['token']):>20} "
            f"id={item['token_id']:<8} "
            f"logit={item['logit']:.3f}"
        )


def compare_lenses(
    lens,
    model,
    tokenizer,
    prompt,
    layers,
    position=-2,
    k=5,
):
    """
    Compare J-Lens and ordinary Logit Lens across layers.
    """

    jlens_logits, model_logits, _ = run_jlens(
        lens,
        model,
        prompt,
        layers=layers,
        positions=[position],
    )

    logit_logits, _, _ = run_logit_lens(
        lens,
        model,
        prompt,
        layers=layers,
        positions=[position],
    )

    print(f"\nPrompt: {prompt}")
    print(f"Position: {position}\n")

    for layer in layers:

        jlens_top = [
            repr(x["token"])
            for x in top_tokens(
                jlens_logits[layer][0],
                tokenizer,
                k,
            )
        ]

        logit_top = [
            repr(x["token"])
            for x in top_tokens(
                logit_logits[layer][0],
                tokenizer,
                k,
            )
        ]

        print(f"Layer {layer:>3}")
        print(f"  Logit Lens: {logit_top}")
        print(f"  J-Lens:     {jlens_top}")

    model_top = [
        repr(x["token"])
        for x in top_tokens(
            model_logits[0],
            tokenizer,
            k,
        )
    ]

    print(f"\nModel: {model_top}")

    return jlens_logits, logit_logits, model_logits


# ------------------------------------------------------------------
# Interactive slice
# ------------------------------------------------------------------

def make_slice(
    model,
    lens,
    prompt,
    layer_stride=2,
    mask_display=True,
):
    """
    Compute position × layer J-Lens data used by the
    interactive visualization.
    """

    return compute_slice(
        model,
        lens,
        prompt,
        layer_stride=layer_stride,
        mask_display=mask_display,
    )


def save_slice(
    model,
    lens,
    prompt,
    output_dir,
    title="J-Lens analysis",
    description="",
    layer_stride=2,
    mask_display=True,
):
    """
    Generate an interactive J-Lens HTML page and save it to disk.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slice_data = make_slice(
        model,
        lens,
        prompt,
        layer_stride=layer_stride,
        mask_display=mask_display,
    )

    page, _, _ = build_page(
        slice_data,
        prompt,
        title=title,
        description=description,
        mode="fetch",
        out_dir=output_dir,
    )

    output_file = output_dir / "index.html"
    output_file.write_text(page)

    print(f"Saved interactive slice to: {output_file}")

    return output_file