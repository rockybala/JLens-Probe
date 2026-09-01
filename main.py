
import config

from utils import (
    compare_lenses,
    load_lens,
    load_model,
    show_tokens,
)

def main():

    model, tokenizer = load_model(
        config.MODEL_NAME,
        device=config.DEVICE,
    )

    lens = load_lens(
        config.LENS_REPO,
        config.LENS_FILE,
        config.LENS_REVISION,
    )

    prompt = config.DEFAULT_PROMPT

    print("\nTokens:")
    show_tokens(tokenizer, prompt)

    layers = [
        model.n_layers // 4,
        model.n_layers // 2,
        model.n_layers // 4 * 3,
        model.n_layers - 2,
    ]

    compare_lenses(
        lens,
        model,
        tokenizer,
        prompt,
        layers=layers,
        position=-2,
        k=config.TOP_K,
    )





if __name__ == "__main__":
    main()