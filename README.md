This repository contains code for the paper:

> Oneata, Dan, Desmond Elliott, and Stella Frank.
> [Seeing What Tastes Good: Revisiting Multimodal Distributional Semantics in the Billion Parameter Era.](https://arxiv.org/abs/2506.03994)
> ACL Findings, 2025.

For an overview of the paper and an interactive exploration of the results, see the [project page](https://danoneata.github.io/seeing-what-tastes-good).

## Data

**Norm data.**
The paper introduces the dataset McRae × THINGS.
This dataset is obtained by pairing the concepts in the THINGS dataset with McRae attributes using GPT-4o.
The resulting data is available in the [`data/mcrae-x-things.json`](data/mcrae-x-things.json) file, which contains a list of positive concept–attribute pairs;
if a concept–attribute pair is missing, then it is a negative pair (the concept does not have that attribute).
Each attribute can be categorised to a higher type (e.g. taxonomic, functional, visual-color);
this mapping is available in the [`data/mcrae-x-things-taxonomy.json`](data/mcrae-x-things-taxonomy.json) file.

**Concepts.**
To represent the concepts visually, we use the images in the [THINGS dataset](https://osf.io/jum2f/).
To represent the concepts as text, we either use the concept names (e.g. "apple") or contextual sentences (e.g. "The apple fell from the tree and rolled down the hill").
We provide three variants of contextual sentences differing in terms of number of sentences and whether they are constrained to exclude attributes or not.

| Num. sentences | Constrained? | Path |
| --- | --- | --- |
| 10 | ✗ | [link](data/things/gpt4o_concept_context_sentences_v2.jsonl) |
| 50 | ✗ | [link](data/things/gpt4o_50_concept_context_sentences_v2.jsonl) |
| 50 | ✓ | [link](data/things/gpt4o_50_constrained_concept_context_sentences_v2.jsonl) |

## Setup

The code uses PyTorch (for feature extraction), which we recommend installing via conda:

```bash
conda create -n probing-norms python=3.12
conda activate probing-norms
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then you can install this code as a library with:

```bash
pip install -e .
```

**Data.**
You will need to download the [THINGS dataset](https://osf.io/jum2f/files/osfstorage), for example using `wget`:
```bash
wget https://files.osf.io/v1/resources/jum2f/providers/osfstorage/?zip= -O things.zip
```

The code assumes that the THINGS dataset is located at `data/things-data`.
If you want to use a different path, you can set using the `DIR_THINGS` environment variable (see the [`probing-norms/data.py`](https://github.com/danoneata/seeing-what-tastes-good/blob/master/probing_norms/data.py#L30) script);
for example:
```bash
export DIR_THINGS=/path/to/things
```

## Evaluating a single model

To evaluate the performance of a single model (let's say the Swin-V2 model, code as `swin-v2-ssl`), we have to perform three steps.

**Feature extraction.**
Extract the features for the THINGS concepts:
```bash
python probing_norms/extract_features_image.py -d things -f swin-v2-ssl
```
For a language model, we need to use the `probing_norms/extract_features_text.py` script instead. For example:
```bash
python probing_norms/extract_features_text.py -d things -f numberbatch -m word
```

**Model training.**
The next step is to train linear probes for both the McRae × THINGS and the Binder datasets:
```bash
python probing_norms/predict.py --feature-type swin-v2-ssl --norms-type mcrae-x-things --split-type repeated-k-fold --embeddings-level concept --classifier-type linear-probe
python probing_norms/predict.py --feature-type swin-v2-ssl --norms-type binder-dense --split-type repeated-k-fold-simple --embeddings-level concept --classifier-type linear-regression
```
Since the Binder dataset has continuous ratings, the settings differ for the two datasets:
- `--classifier-type`: the probe is a linear classifier for McRae × THINGS, and a linear regressor for Binder.
- `--split-type`: we use a stratified repeated k-fold split for McRae × THINGS, while for Binder we don't have to use stratification.

**Results generation.**
Finally, we evaluate the performance in terms of F₁ selectivity for McRae × THINGS and root mean squared error (RMSE) for Binder:
```bash
python probing_norms/get_results.py paper-table-main-acl-camera-ready:swin-v2-ssl
```

## Replicating the results in the paper

We provide the scripts to replicate the various tables and figures from the paper.
Prior to running these scripts, you need to extract the features for all models and train the corresponding probes.
The list of models used in the paper is available in the [`probing_norms/get_results.py`](probing_norms/get_results.py) script, under the `MAIN_TABLE_MODELS` variable:

```python
MAIN_TABLE_MODELS = [
    # Vision models
    "random-siglip",
    "vit-mae-large",
    "max-vit-large",
    "max-vit-large-in21k",
    "dino-v2",
    "swin-v2-ssl",
    # Vision-language models
    "llava-1.5-7b",
    "qwen2.5-vl-3b-instruct",
    "clip",
    "pali-gemma-224",
    "siglip-224",
    # Language models
    "glove-840b-300d-word",
    "fasttext-word",
    "numberbatch-word",
    "clip-word",
    "deberta-v3-contextual-layers-0-to-6-word",
    "gemma-2b-contextual-layers-9-to-18-seq-last-word",
]
```

### Table 2: Main results

```bash
python probing_norms/get_results.py paper-table-main-acl-camera-ready
```

### Figure 2: Correlation between models

```bash
python probing_norms/get_results.py get-correlation-between-models-2
```

### Figure 3: Rankings of models

```bash
python probing_norms/get_results.py ranking-plot
```

### Table 3: Possible confounds

```bash
python probing_norms/scripts/eval_norm_correlations.py
```

### Figure 4: Per attribute model comparison

```bash
python probing_norms/get_results.py compare-two-models-scatterplot-2
```

### Figure 5: Per attribute type results on McRae × THINGS

```bash
python run probing_norms/get_results.py per-metacategory-mcrae-mapped:mcrae-x-things
```

### Table 5: Main results (detailed)

```bash
python probing_norms/get_results.py paper-table-main-full-acl-camera-ready
```

### Table 6: Contextualised language models

```bash
python probing_norms/get_results.py results-contextualised-language-models
```

### Figure 8: Per attribute results on Binder

```bash
python probing_norms/get_results.py binder-norms
```

### Figure 9: Per attribute type results on Binder

```bash
python probing_norms/get_results.py per-metacategory-binder
```

### Figure 10: Qualitative results

```bash
streamlit run probing_norms/scripts/show_results_per_feature_norm_v2.py
```
