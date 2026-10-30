# JLens-Probe

**How sensitive are Jacobian Lens representations to the distribution used to fit the lens?**

This project compares Jacobian lenses fitted on WikiText and deception-oriented data for Qwen3.5-4B. The aim is to test whether changing the fitting distribution changes the concepts exposed by the lens, while keeping the underlying language model fixed.

The repository compares three lenses:

- **WikiText-1000** — a public J-Lens checkpoint fitted on 1,000 WikiText examples and distributed through `neuronpedia/jacobian-lens`.
- **WikiText-100** — a locally fitted lens using 100 WikiText examples.
- **Deception-100** — a locally fitted lens using 100 deception-oriented examples.

Comparing WikiText-1000 with WikiText-100 provides a same-distribution, different-sample-size baseline. Comparing WikiText-100 with Deception-100 isolates the effect of changing the fitting distribution at matched sample count.

## Experiments

### 1. Intermediate concept recovery

`experiment_1_intermediate_recovery.py`

Tests whether each lens recovers known intermediate concepts in multi-hop reasoning examples. For each example, the script measures the vocabulary rank of the target intermediate token across fitted layers and reports `pass@k`.

In the committed multihop results, all three lenses recover a substantial fraction of the same intermediate concepts. For example, `pass@20` is about 0.62 for WikiText-1000 and WikiText-100, and 0.65 for Deception-100. This experiment checks that refitting has not simply destroyed the basic J-Lens readout.

![Intermediate concept recovery](results/experiment1.png)

### 2. Vocabulary-linked direction alignment

`experiment_2_direction_alignment.py`

For token `v` and layer `l`, the J-Lens direction in source residual space is

```text
d_(v,l) = J_l^T w_v
```

where `w_v` is the model's unembedding vector for token `v`.

The script compares the same token-linked direction across lenses using cosine similarity. The analysis includes generic, deception-related, and evaluation-awareness vocabulary.

The WikiText-100 lens is already fairly aligned with WikiText-1000 at layer 0 (mean cosine about 0.81 across the configured vocabulary) and becomes very closely aligned in later layers. The Deception-100 lens differs much more strongly in early and middle layers: its mean cosine with WikiText-1000 is about 0.25 at layer 0 and 0.55 at layer 10, before increasing in later layers. These results suggest that fitting distribution can substantially affect the directions exposed by the lens, especially away from the final layers.

![Direction alignment](results/experiment2.png)

### 2b. Sparse concept recovery

`experiment_2b_sparse_concepts.py`

Tests a stricter question: for the **same residual activation**, do different fitted lenses select the same sparse token concepts?

For each layer, the script constructs token-indexed directions `J_l^T w_v` and greedily reconstructs the residual with a non-negative sparse combination of `K=25` concept directions.

The exact gradient-pursuit implementation used in the reference work is not public. This script therefore uses a full-vocabulary non-negative greedy pursuit with projected-gradient refitting. It should be treated as an approximate implementation rather than an exact reproduction of the unpublished sparse-decomposition algorithm.

![Sparse concept recovery](results/experiment2b.png)

### 3. Safety-relevant readout *(in progress)*

`experiment_3_safety_readout.py`

This script is intended to compare the three lenses on alignment-audit prompts using two simple readouts: deception-related tokens in the J-Lens top-k predictions, and an evaluation-awareness score based on configured token sets. Experiment 3 results are not included in the current repository.

## Deception fitting corpus

`build_deception_DeceptionBench.py` generates candidate responses from `PKU-Alignment/DeceptionBench` using Qwen3.5-4B. Generation uses fixed-seed sampling and is saved incrementally so longer runs can be resumed.

The final 100-example fitting set used in this repository is included at:

```text
data/deception_prompts_100_jlens.json
```

The current script does **not** fully automate the final selection/construction of these 100 examples: `build_deception_prompts()` is still a placeholder. The committed JSON file should therefore be treated as the exact fitting corpus for the reported local deception lens. A future version should make this selection step fully reproducible.

## Repository structure

```text
JLens-Probe/
├── README.md
├── LICENSE
├── requirements.txt
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
│   └── deception_prompts_100_jlens.json
├── Fitted-Jlens/
└── results/
    ├── experiment1.png
    ├── experiment2.png
    ├── experiment2b.png
    ├── experiment_1/
    ├── experiment_2/
    └── experiment_2b/
```

The fitted `.pt` checkpoints are not committed because of their size. Their expected paths are defined in `config.py`.

## Installation

A GPU is recommended for the full experiments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiments also require the upstream Jacobian Lens package. Install it following the instructions in the `anthropics/jacobian-lens` repository. Experiment 3 additionally expects a local checkout at `jacobian-lens/` because it searches that checkout for alignment-audit datasets.

You will also need access to the configured Hugging Face model and J-Lens checkpoint.

## Local fitted lenses

Place the two locally fitted checkpoints at:

```text
Fitted-Jlens/qwen35_4b_wikitext_n100.pt
Fitted-Jlens/qwen35_4b_deception_n100.pt
```

If you use different filenames, update `LOCAL_LENS_FILES` in `config.py`.

## Running

```bash
python experiment_1_intermediate_recovery.py
python experiment_2_direction_alignment.py
python experiment_2b_sparse_concepts.py
python experiment_3_safety_readout.py
```

Model choice, evaluation data, vocabulary groups, safety words, analysis layers, and local lens paths are configured in `config.py`.

## Notes on interpretation

This repository is a small robustness study, not a claim that one fitting distribution is generally better than another. The experiments currently use one model, small local fitting sets, and limited evaluation tasks. The sparse-decomposition experiment is approximate, and the deception-corpus selection step is not yet fully automated. Results should be interpreted with those limitations in mind.

## License and upstream work

This repository is released under the MIT License. The Jacobian Lens implementation used by the experiments comes from the upstream `anthropics/jacobian-lens` project and is subject to its own license. Model weights, datasets, and downloaded checkpoints remain subject to their respective terms.
