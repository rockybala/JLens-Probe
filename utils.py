from pathlib import Path

import jlens
import torch
import transformers

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

    if device is not None:
        hf_model = hf_model.to(device)

    hf_model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    model = jlens.from_hf(hf_model, tokenizer)

    return model, tokenizer


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