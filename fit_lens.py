import config

from utils import load_model
import time
import jlens
from jlens.examples import load_wikitext_prompts

def main():

    model, tokenizer = load_model(
        config.MODEL_NAME,
        device=None,
    )

    prompts = load_wikitext_prompts(n_prompts=2)

    print(f"\nLoaded {len(prompts)} prompts")
    print("Example:")
    print(repr(prompts[0][:300]))

    # ---------------------------
    # Fit J-Lens
    # ---------------------------

    print("\nStarting J-Lens fitting...")

    start = time.perf_counter()

    lens = jlens.fit(
        model,
        prompts=prompts,
        dim_batch=4,
        max_seq_len=32,
        checkpoint_path="results/jlens_n2_checkpoint.pt",
    )

    elapsed = time.perf_counter() - start

    # ---------------------------
    # Save lens
    # ---------------------------

    lens.save(
        "results/jlens_wikitext_n2.pt"
    )
    print("\nFinished!")
    print(f"Total time: {elapsed / 60:.2f} minutes")
    print(f"Time / prompt: {elapsed / len(prompts):.2f} seconds")






if __name__ == "__main__":
    main()