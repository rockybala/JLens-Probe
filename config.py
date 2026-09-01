MODEL_NAME = "Qwen/Qwen3.5-4B"
DEVICE = "cpu" #auto (for Mac's Apple gpu)

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000" ##branch to use

LENS_FILES = {
    "Qwen/Qwen3.5-4B":
        "qwen3.5-4b/jlens/Salesforce-wikitext/"
        "Qwen3.5-4B_jacobian_lens_n1000.pt",

    "Qwen/Qwen3.6-27B":
        "qwen3.6-27b/jlens/Salesforce-wikitext/"
        "Qwen3.6-27B_jacobian_lens_n1000.pt",
}

LENS_FILE = LENS_FILES[MODEL_NAME]

DEFAULT_PROMPT = (
    "Fact: The currency used in the country shaped like a boot is"
)

TOP_K = 5