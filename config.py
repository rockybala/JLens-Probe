MODEL_NAME = "Qwen/Qwen3.5-4B" #"Qwen/Qwen3-0.6B" #Qwen/Qwen3.5-4B"
EVAL_NAME = "multihop"

##central lens
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

##local lens
LENS_FILE = LENS_FILES[MODEL_NAME]

LOCAL_LENS_FILES = {
    "wiki_100":
        "Fitted-Jlens/qwen35_4b_wikitext_n100.pt",

    "deception_100":
        "Fitted-Jlens/qwen35_4b_deception_n100.pt",
}

EVAL_FILES = {
    "multihop": "lens-eval-multihop.json",
    "multilingual": "lens-eval-multilingual.json",
    "order_ops": "lens-eval-order-ops.json",
    "poetry": "lens-eval-poetry.json",
    "typo": "lens-eval-typo.json",
    "association": "lens-eval-association.json",
}

EVAL_FILE = EVAL_FILES[EVAL_NAME]

DEFAULT_PROMPT = (
    "Fact: The currency used in the country shaped like a boot is"
)

TOP_K = 5

# ------------------------------------------------------------------
# Experiment 2: vocabulary-linked direction alignment
# ------------------------------------------------------------------
# Keep the set small enough that the experiment runs quickly, while separating
# ordinary concepts from concepts especially relevant to the deception fit.
DIRECTION_TOKEN_GROUPS = {
    "generic": [
        "city", "country", "water", "music", "book", "science", "family",
        "school", "computer", "garden", "history", "food", "Earth", "China",
        "heart", "oxygen", "basketball", "March", "Friday", "moon",
    ],
    "deception": [
        "fake", "trick", "fraud", "secret", "hidden", "lie", "lying",
        "deceive", "deception", "mislead", "fabricated", "manipulation",
        "honest", "truth", "leverage", "threat", "blackmail",
    ],
    "eval_awareness": [
        "fictional", "simulation", "scenario", "artificial", "benchmark",
        "experiment", "testing", "mock", "evaluation",
    ],
}

# ------------------------------------------------------------------
# Experiment 3: Section-5-style safety/alignment auditing readout
# ------------------------------------------------------------------
# The script recursively searches jacobian-lens/ for JSON filenames containing
# these hints. Keep aliases because repository/data filenames may use hyphens,
# underscores, or abbreviated names.
SAFETY_DATASET_HINTS = {
    "blackmail": ["blackmail"],
    "prompt_injection": ["prompt-injection", "prompt_injection", "injection"],
    "fake_vulnerability": ["fakevuln", "fake-vulnerability", "fake_vulnerability"],
    "edit_fake_numbers": ["editfakenumbers", "edit-fake-numbers", "fake_numbers"],
    "silent_override": ["silentoverride", "silent-override", "silent_override"],
    "self_issued_token": ["selfissuedtoken", "self-issued-token", "self_issued_token"],
}

# Fixed deception vocabulary inspired by the reward-hacking experiment in §5.
SAFETY_DECEPTION_WORDS = [
    "fake", "trick", "fraud", "secretly", "hidden", "deceive", "deception",
    "lie", "lying", "mislead", "fabricated", "manipulation", "leverage",
    "blackmail",
]

# The paper's Appendix proof-of-concept score used "fictional" and
# "simulation". We retain those and add a few closely related single-token
# candidates; multi-token entries are automatically skipped.
EVAL_AWARENESS_WORDS = [
    "fictional", "simulation", "scenario", "artificial", "benchmark",
    "experiment", "testing", "mock",
]

# Unrelated concrete nouns for the score baseline. The paper used 100 controls;
# this compact set is a fast project-scale approximation.
EVAL_CONTROL_WORDS = [
    "recipe", "guitar", "bicycle", "window", "garden", "coffee", "river",
    "pencil", "chair", "apple", "camera", "mountain", "shirt", "table",
    "flower", "bread", "train", "house", "book", "music",
]

SAFETY_TOP_K = 10
SAFETY_MAX_EXAMPLES_PER_DATASET = 10

# The Qwen3.5-4B lenses contain fitted source layers 0..30. This analysis
# band covers late-middle layers where Experiment 1 often recovered
# intermediates; it is not intended as a universal workspace layer range. Set to None to use every fitted layer.
SAFETY_ANALYSIS_LAYERS = list(range(18, 29))
