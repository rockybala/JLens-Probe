# experiments/01_basic_readouts.py

from config import *
from utils import *


model, tokenizer = load_model(
    MODEL_NAME,
    device="cpu",
)

lens = load_lens(
    LENS_REPO,
    LENS_FILE,
    LENS_REVISION,
)

prompt = (
    "Fact: The currency used in "
    "the country shaped like a boot is"
)

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
    layers,
    position=-2,
)