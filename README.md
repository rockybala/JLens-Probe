# JLens-Probe

**Testing the robustness of Jacobian Lens representations across fitting distributions, with a focus on safety-relevant concepts.**

This repository contains experiments built around the **Jacobian Lens (J-Lens)** for Qwen3.5-4B. The project asks a simple interpretability question:

> **How much does a J-Lens representation depend on the data distribution used to fit the lens?**

To study this, the code compares three lenses applied to the **same underlying model activations**:

1. **Anthropic-1000** — a pretrained J-Lens fit on 1,000 WikiText examples.
2. **Wiki-100** — a locally fitted lens using 100 WikiText examples.
3. **Deception-100** — a locally fitted lens using 100 deception-oriented examples constructed from DeceptionBench/Qwen generations.

The Wiki-100 comparison provides a small-data baseline: it helps distinguish changes caused by fitting on fewer examples from changes associated with a different fitting distribution. The Deception-100 comparison then asks whether fitting on safety-relevant data changes the concepts or directions exposed by the lens.

## Research questions

The repository contains several complementary probes:

- **Intermediate concept recovery:** Do the different fitted lenses recover the same known intermediate concepts in multi-hop reasoning prompts?
- **Direction alignment:** For a fixed vocabulary concept and layer, how aligned are the corresponding J-Lens directions across fits?
- **Sparse concept recovery:** Given the exact same residual activation, do different fitted lenses select the same sparse token concepts?
- **Safety-relevant readout:** Do lenses fitted on different distributions differ when reading deception- or evaluation-awareness-related signals from alignment-audit prompts?

The broader motivation is methodological: representation-based monitoring is only useful if we understand how robust the representation tool itself is to choices made during fitting.

## Repository structure

```text
JLens-Probe/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── config.py
├── utils.py
├── main.py
├── fit_lens.py
├── build_deception_DeceptionBench.py
├── experiment_1_intermediate_recovery.py
├── experiment_2_direction_alignment.py
├── experiment_2b_sparse_concepts.py
├── experiment_3_safety_readout.py
├── data/
│   └── ...                         # local fitting/evaluation data
├── Fitted-Jlens/
│   ├── qwen35_4b_wikitext_n100.pt
│   └── qwen35_4b_deception_n100.pt
├── results/
│   ├── experiment_1/
│   ├── experiment_2/
│   ├── experiment_2b/
│   └── experiment_3/
└── jacobian-lens/                  # local clone of the upstream J-Lens repo
```

Large fitted-lens checkpoints are ignored by default in `.gitignore`; if you want to publish them, use Git LFS or a model/data host rather than ordinary Git history.

## Experimental setup

### Model

The current configuration uses:

```text
Qwen/Qwen3.5-4B
```

The configuration also contains a path entry for Qwen3.6-27B for future scaling experiments.

### Lenses

`config.py` defines the three lenses used throughout the comparisons:

| Lens | Fit distribution | Number of fitting examples | Source |
|---|---|---:|---|
| `anthropic_1000` | WikiText | 1000 | pretrained `neuronpedia/jacobian-lens` checkpoint |
| `wiki_100` | WikiText | 100 | locally fitted |
| `deception_100` | deception-oriented corpus | 100 | locally fitted |

The main comparisons are deliberately structured so that **Anthropic-1000 vs Wiki-100** estimates variation from fit size / refitting, while **Wiki-100 vs Deception-100** more directly probes distributional dependence at matched sample count.

## Deception fitting corpus

`build_deception_DeceptionBench.py` uses `PKU-Alignment/DeceptionBench` and Qwen3.5-4B to generate paired responses for each scenario:

- a **baseline response** to the scenario's inner prompt;
- a **scenario response** conditioned on the supplied system prompt, outer prompt, and response format.

Generations are written incrementally to JSONL so long runs can be resumed. The selected deception-oriented examples used for fitting the local lens should be placed under `data/`.

> **Note:** dataset construction/selection is a research choice. Keep the exact data file and selection procedure with any published results so the fitted lens is reproducible.

## Experiment 1 — Intermediate concept recovery

**File:** `experiment_1_intermediate_recovery.py`

This experiment evaluates whether known intermediate concepts from J-Lens evaluation tasks become visible in the lens output. For each prompt and intermediate concept, it:

1. determines the readout position immediately before the target continuation;
2. applies each J-Lens at that position;
3. computes the vocabulary rank of the known intermediate token at every fitted layer;
4. records its best layer/rank; and
5. summarizes recovery using `pass@k` for several values of `k`.

The current evaluation is configured for the multihop dataset, with alternatives such as multilingual, order-of-operations, poetry, typo, and association tasks exposed in `config.py`.

Outputs are saved under:

```text
results/experiment_1/
```

## Experiment 2 — Vocabulary-linked direction alignment

**File:** `experiment_2_direction_alignment.py`

For token `v` and layer `l`, the experiment defines the J-Lens direction in source residual space as

```text
d_(v,l) = J_l^T w_v
```

where `w_v` is the model's unembedding vector for token `v`.

It compares the same token direction across the three fitted lenses using cosine similarity. Tokens are grouped into:

- **generic concepts**;
- **deception-related concepts**; and
- **evaluation-awareness concepts**.

The script saves detailed token-level results and layer-wise summaries as JSON/CSV under:

```text
results/experiment_2/
```

This experiment is useful for separating a general refitting/sample-size effect from a distribution-specific effect.

## Experiment 2b — Sparse concept recovery

**File:** `experiment_2b_sparse_concepts.py`

This experiment asks a stricter question:

> For the **same residual activation** `h_l`, do the three fitted J-Lenses identify the same sparse set of token concepts?

For each layer, it constructs token-indexed directions `J_l^T w_v` and greedily reconstructs the residual using a non-negative sparse combination of `K=25` concept directions. The vocabulary search is performed over the full vocabulary at every pursuit step.

The exact gradient-pursuit implementation used in the reference work is not released. This repository therefore uses an **approximate full-vocabulary non-negative greedy pursuit**, with projected-gradient refitting of the selected coefficients. Results from this experiment should not be described as an exact reproduction of the unpublished algorithm.

Outputs are saved under:

```text
results/experiment_2b/
```

## Experiment 3 — Safety-relevant readout

**File:** `experiment_3_safety_readout.py`

This experiment applies all three lenses to safety/alignment-audit prompts and measures two simple readouts:

1. **Deception-token top-k count:** counts configured deception-related tokens appearing in J-Lens top-k predictions across selected layers/positions.
2. **Evaluation-awareness score:** mean log-probability of evaluation-awareness tokens minus mean log-probability of unrelated control tokens.

The script searches a local `jacobian-lens/` checkout for configured audit datasets, including blackmail, prompt injection, fake-vulnerability/editing, silent-override, and self-issued-token style cases.

The current analysis band is layers 18–28 for Qwen3.5-4B. This was chosen as a project-scale analysis region based on earlier intermediate-recovery behavior; it should **not** be interpreted as a universal or externally established “workspace” layer range.

Outputs are saved under:

```text
results/experiment_3/
```

## Utilities

`utils.py` contains shared functionality for:

- loading Hugging Face models and wrapping them for J-Lens;
- loading pretrained and locally fitted lenses;
- device selection (CUDA, Apple MPS, or CPU);
- finding single-token representations;
- computing token ranks;
- choosing readout positions;
- applying J-Lens and ordinary Logit Lens; and
- generating interactive J-Lens slices.

`main.py` is a small exploratory script for comparing J-Lens and Logit Lens outputs on a single prompt. `fit_lens.py` is a minimal fitting example and is mainly useful as a starting point; the research experiments are the numbered scripts above.

## Installation

A GPU is strongly recommended for the full experiments.

```bash
git clone <YOUR-REPOSITORY-URL>
cd JLens-Probe

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the upstream Jacobian Lens package according to its repository instructions. The experiment-3 dataset discovery currently expects a local checkout at:

```text
JLens-Probe/jacobian-lens/
```

You will also need access to the configured Hugging Face model/checkpoints.

## Adding local data and fitted lenses

Place your project data under:

```text
data/
```

and the two locally fitted Qwen3.5-4B lenses at:

```text
Fitted-Jlens/qwen35_4b_wikitext_n100.pt
Fitted-Jlens/qwen35_4b_deception_n100.pt
```

If you use different filenames, update `LOCAL_LENS_FILES` in `config.py`.

## Running the experiments

Intermediate recovery:

```bash
python experiment_1_intermediate_recovery.py
```

Direction alignment:

```bash
python experiment_2_direction_alignment.py
```

Sparse concept recovery:

```bash
python experiment_2b_sparse_concepts.py
```

Safety readout:

```bash
python experiment_3_safety_readout.py
```

The model, evaluation dataset, vocabulary groups, safety words, analysis layers, and local lens paths can be changed in `config.py`.

## Results and plots

Machine-readable outputs are written into the corresponding subdirectory of `results/`. Plots can be committed alongside the JSON/CSV outputs, for example:

```text
results/
├── experiment_1/
│   ├── multihop_debug_results.json
│   └── intermediate_recovery.png
├── experiment_2/
│   ├── direction_alignment_detailed.json
│   ├── direction_alignment_summary.json
│   ├── direction_alignment_summary.csv
│   └── direction_alignment_by_layer.png
├── experiment_2b/
│   ├── ...
│   └── sparse_concept_overlap.png
└── experiment_3/
    ├── ...
    └── safety_readout.png
```

For an application-facing repository, adding a small number of plots that communicate the central findings is more useful than committing every diagnostic figure. Good candidates are: (1) layer-wise direction alignment for the three lens pairs, (2) intermediate recovery/pass@k across lenses, and (3) a compact safety-readout comparison if Experiment 3 is complete.

## Interpretation and limitations

This repository is an exploratory robustness study rather than a claim that a particular lens is “correct.” Important limitations include:

- only one primary model is currently evaluated;
- the locally fitted lenses use relatively small fitting corpora;
- the deception corpus is constructed/selected and may encode dataset-specific properties;
- vocabulary-linked concepts depend on tokenizer granularity;
- sparse recovery uses an approximate implementation because the reference gradient-pursuit code is unavailable;
- safety readouts based on hand-selected vocabularies are probes, not definitive measures of deception or evaluation awareness; and
- observed differences between lenses do not by themselves establish a causal effect of the fitting distribution.

Natural next steps are to repeat the study across model scales, fitting-set sizes and random seeds; use held-out safety tasks; quantify uncertainty; test robustness to concept vocabulary choices; and add causal interventions where possible.

## Status

**Research prototype / work in progress.** The repository is intended to make the experimental logic and current results inspectable and reproducible while the project is still being developed.

## Acknowledgements

This project builds on the Jacobian Lens implementation and pretrained lens checkpoints. Please cite the original Jacobian Lens work and upstream repository when using or extending this code, and follow the licenses/terms of the underlying model and datasets.
