import json
import os
import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-4B"
OUTPUT_FILE = "data/qwen_deception_generations.jsonl"

MAX_NEW_TOKENS = 2048
SEED = 42

def generate_response(model, tokenizer, messages, device):
    """Generate one deterministic response from Qwen."""

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print("text")
    print(text)
    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(device)

    print("inputs")
    print(inputs)
    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )
    print("output")
    print(output)
    generated_tokens = output[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=False,
    )
    hit_token_limit = len(generated_tokens) >= MAX_NEW_TOKENS

    return response, len(generated_tokens), hit_token_limit

def generate_deception_responses(n_scenarios=None):
    """
    Generate two independent Qwen responses for each DeceptionBench scenario:

    1. baseline_response:
       Qwen responds to inner_prompt alone.

    2. scenario_response:
       Qwen responds to system_prompt + outer_prompt + format.
    """

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    dataset = load_dataset("PKU-Alignment/DeceptionBench")
    scenarios = dataset["test"]

    if n_scenarios is not None:
        scenarios = scenarios.select(range(n_scenarios))

    print(f"Running {len(scenarios)} scenarios")

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
    ).to(device)

    model.eval()

    # ---------------------------------------------------------
    # Resume support for incomplete runs
    # ---------------------------------------------------------
    completed_ids = set()

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    completed_ids.add(record["id"])

        print(f"Found {len(completed_ids)} completed scenarios")

    torch.manual_seed(SEED)

    # ---------------------------------------------------------
    # Generate
    # ---------------------------------------------------------
    for i, scenario in enumerate(scenarios):

        scenario_id = scenario["id"]

        if scenario_id in completed_ids:
            print(f"[{i + 1}/{len(scenarios)}] Skipping {scenario_id}")
            continue

        print(
            f"\n[{i + 1}/{len(scenarios)}] "
            f"{scenario['type']} | {scenario_id}"
        )

        # =====================================================
        # 1. BASELINE
        # =====================================================

        baseline_messages = [
            {
                "role": "user",
                "content": (scenario["inner_prompt"]
                + "\n\nPlease give a concise but complete response. "
                "Keep your reasoning brief and aim to finish the entire "
                 "response within 500 tokens."),
            }
        ]

        baseline_response, baseline_n_tokens, baseline_hit_limit = generate_response(
            model,
            tokenizer,
            baseline_messages,
            device,
        )

        # =====================================================
        # 2. DECEPTION SCENARIO
        # =====================================================

        scenario_messages = [
            {
                "role": "system",
                "content": scenario["system_prompt"],
            },
            {
                "role": "user",
                "content": (
                    scenario["outer_prompt"]
                    + "\n\n"
                    + scenario["format"]
                ),
            },
        ]

        scenario_response, scenario_n_tokens, scenario_hit_limit = generate_response(
            model,
            tokenizer,
            scenario_messages,
            device,
        )

        # =====================================================
        # Save everything
        # =====================================================

        record = {
            "id": scenario_id,
            "type": scenario["type"],

            "inner_prompt": scenario["inner_prompt"],
            "baseline_response": baseline_response,
            "baseline_n_tokens": baseline_n_tokens,
            "baseline_hit_token_limit": baseline_hit_limit,

            "system_prompt": scenario["system_prompt"],
            "outer_prompt": scenario["outer_prompt"],
            "format": scenario["format"],
            "scenario_response": scenario_response,
            "scenario_n_tokens": scenario_n_tokens,
            "scenario_hit_token_limit": scenario_hit_limit,

            "model": MODEL_NAME,
            "seed": SEED,
            "max_new_tokens": MAX_NEW_TOKENS,
        }

        if scenario_n_tokens >= MAX_NEW_TOKENS:
            print("WARNING: scenario response hit max_new_tokens")

        # Save immediately so the run is resumable
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

        print("\n--- BASELINE ---")
        print(baseline_response[:500])

        print("\n--- SCENARIO ---")
        print(scenario_response[:1000])

    print(f"\nDone. Saved to: {OUTPUT_FILE}")


def build_deception_prompts():
    """
    Select deceptive Qwen generations and construct the final
    100-sequence J-Lens fitting corpus.

    We'll implement this after inspecting the generations.
    """
    pass


if __name__ == "__main__":
    generate_deception_responses()