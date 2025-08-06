import csv
import json
import pickle
import pdb
import os
import random
import sys

from collections import Counter
from functools import partial
from itertools import combinations

import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from adjustText import adjust_text
from matplotlib import gridspec
from matplotlib import pyplot as plt
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
    silhouette_score,
)
from umap import UMAP
from tqdm import tqdm

from probing_norms.data import DIR_LOCAL, DATASETS, load_mcrae_x_things
from probing_norms.utils import (
    cache_df,
    cache_json,
    cache_np,
    read_json,
    read_file,
    multimap,
)
from probing_norms.scripts.prepare_mcrae_norms_grouped import (
    load_feature_new_to_features_mcrae,
)
from probing_norms.predict import (
    NORMS_LOADERS,
    FEATURE_TYPE_TO_MODALITY,
    get_binary_labels,
)

NORMS_MODEL = "chatgpt-gpt3.5-turbo"
DATASET_NAME = "things"


def accuracy_within(true, pred, delta=1):
    """Calculate accuracy within a certain delta."""
    return 100 * np.mean(np.abs(true - pred) <= delta)


SCORE_FUNCS = {
    "score-accuracy": accuracy_score,
    "score-precision": partial(precision_score, zero_division=0),
    "score-recall": recall_score,
    "score-f1": f1_score,
    "score-roc-auc": roc_auc_score,
    # regression metrics
    "score-rmse": root_mean_squared_error,
    "score-mse": mean_squared_error,
    "score-mae": mean_absolute_error,
    "score-r2": r2_score,
    "score-accuracy-delta-1": partial(accuracy_within, delta=1),
    "score-accuracy-delta-0.5": partial(accuracy_within, delta=0.5),
}

SPLIT_TO_SCORE_FUNCS = {
    "leave-one-concept-out": ["score-accuracy"],
    "repeated-k-fold": ["score-precision", "score-recall", "score-f1"],
    "repeated-k-fold-instance": ["score-precision", "score-recall", "score-f1"],
}

FEATURE_TYPES = [
    "pali-gemma-224",
    "siglip-224",
    "vit-mae-large",
    "dino-v2",
    "swin-v2",
    "max-vit-large",
    "random-siglip",
    "clip",
]


OUTPUT_PATH = "output/{}-predictions/{}/{}/{}-{}-{}-{}"

MAIN_TABLE_MODELS = [
    "random-siglip",
    "vit-mae-large",
    "max-vit-large",
    "max-vit-large-in21k",
    "dino-v2",
    "swin-v2-ssl",
    #
    # "clip-dfn2b",
    "llava-1.5-7b",
    "qwen2.5-vl-3b-instruct",
    "clip",
    "pali-gemma-224",
    "siglip-224",
    #
    "glove-840b-300d-word",
    "fasttext-word",
    "numberbatch-word",
    "clip-word",
    "deberta-v3-contextual-layers-0-to-6-word",
    "gemma-2b-contextual-layers-9-to-18-seq-last-word",
]


# Names used for plotting or other output.
METACATEGORY_SHORT_NAMES = {
    # "encyclopaedic": "encycl.",
    "visual-colour": "visual: colour",
    "visual-form_and_surface": "visual: form & surface",
    "visual-motion": "visual: motion",
}

METACATEGORY_NAMES = {
    "visual-colour": "visual: colour",
    "visual-form_and_surface": "visual: form & surface",
    "visual-motion": "visual: motion",
}

FEATURE_NAMES = {
    "random-predictor": "Random predictor",
    "random-siglip": "Random SigLIP",
    "vit-mae-large": "ViT-MAE",
    "max-vit-large": "Max ViT (IN-1K)",
    "max-vit-large-in21k": "Max ViT (IN-21K)",
    "swin-v2": "Swin-V2 (FT)",
    "swin-v2-ssl": "Swin-V2",
    "dino-v2": "DINOv2",
    "siglip-224": "SigLIP",
    "pali-gemma-224": "PaliGemma",
    "llava-1.5-7b": "LLaVA-1.5",
    "qwen2.5-vl-3b-instruct": "Qwen2.5-VL",
    "clip": "CLIP (image)",
    "clip-dfn2b": "CLIP DFN-2B (image)",
    "glove-6b-300d-word": "GloVe 6B",
    "glove-840b-300d-word": "GloVe 840B",
    "numberbatch-word": "Numberbatch",
    "fasttext-word": "FastText",
    "deberta-v3-contextual-last-word": "DeBERTa v3",
    "deberta-v3-contextual-layers-0-to-6-word": "DeBERTa v3",
    "gemma-2b-word": "Gemma",
    "gemma-2b-contextual-last-word": "Gemma",
    "gemma-2b-contextual-layers-9-to-18-seq-last-word": "Gemma",
    "clip-word": "CLIP (text)",
}

NORMS_NAMES = {
    "mcrae-x-things": "McRae×THINGS",
    "mcrae-mapped": "McRae++",
    "binder-4": "Binder (binarised)",
    "binder-median": "Binder (binarised)",
    "binder-dense": "Binder",
}

SCORE_NAMES = {
    "score-accuracy": "Accuracy",
    "score-precision": "Precision",
    "score-recall": "Recall",
    "score-f1": "F1",
    "score-f1-selectivity": "F1 selectivity",
    "score-f1-selectivity-norm": "F1 selectivity (norm)",
    "score-roc-auc": "ROC AUC",
    "score-mse": "MSE",
    "score-rmse": "RMSE",
    "score-r2": "R²",
    "score-accuracy-delta-1": "Accuracy (Δ=1)",
    "score-accuracy-delta-0.5": "Accuracy (Δ=0.5)",
}

METRIC_TO_FMT = {
    "score-f1-selectivity": "{:.1f}",
    "score-f1-selectivity-norm": "{:.1f}",
    "score-f1": "{:.1f}",
    "score-precision": "{:.1f}",
    "score-recall": "{:.1f}",
    "score-rmse": "{:.2f}",
    "score-mae": "{:.2f}",
}

# Second level of type aggregation.
BINDER_METACATEGORIES_2 = {
    "Vision": "Sensory",
    "Audition": "Sensory",
    "Somatic": "Sensory",
    "Gustation": "Sensory",
    "Olfaction": "Sensory",
    "Motor": "Motor",
    "Spatial": "Space",
    "Temporal": "Time",
    "Causal": "Time",
    "Social": "Social",
    "Cognition": "Social",
    "Emotion": "Emotion",
    "Drive": "Drive",
    "Attention": "Drive",
}


def normalize_feature_name(text):
    SEP = "_-_"
    if SEP in text:
        prefix, text = text.split(SEP)
        assert prefix in {"beh", "inbeh", "eg", "has_units", "worn_by_men"}, prefix
    text = text.replace("_", " ").strip()
    return text


def evaluate_binary_classification(data, score_types):
    true = np.array([datum["true"] for datum in data])
    pred = np.array([datum["pred"] for datum in data])
    pred = (pred > 0.5).astype(int)
    return [
        {
            "score-type": score_type,
            "score": 100 * SCORE_FUNCS[score_type](true, pred),
        }
        for score_type in score_types
    ]


def evaluate_regression(data, score_types):
    true = np.array([datum["true"] for datum in data])
    pred = np.array([datum["pred"] for datum in data])
    return [
        {
            "score-type": score_type,
            "score": SCORE_FUNCS[score_type](true, pred),
        }
        for score_type in score_types
    ]


def load_result_path(path, evaluate):
    results = read_json(path)
    scores = [score for result in results for score in evaluate(result["preds"])]
    df = pd.DataFrame(scores)
    df = df.groupby("score-type")["score"].mean()
    return df.to_dict()


def load_result(
    classifier_type,
    embeddings_level,
    split_type,
    feature_type,
    feature_norm_str,
    feature_id,
):
    path = (
        OUTPUT_PATH.format(
            classifier_type,
            embeddings_level,
            split_type,
            DATASET_NAME,
            feature_type,
            feature_norm_str,
            feature_id,
        )
        + ".json"
    )
    evaluate_func = (
        evaluate_regression
        if classifier_type == "linear-regression"
        else evaluate_binary_classification
    )
    if classifier_type == "linear-regression":
        score_types = [
            "score-rmse",
            "score-mse",
            "score-mae",
            "score-r2",
            "score-accuracy-delta-1",
            "score-accuracy-delta-0.5",
        ]
    else:
        score_types = SPLIT_TO_SCORE_FUNCS[split_type]
    scores_dict = load_result_path(
        path, partial(evaluate_func, score_types=score_types)
    )
    return {
        "level": embeddings_level,
        "model": feature_type,
        "split": split_type,
        **scores_dict,
    }


def load_result_features(
    classifier_type,
    embeddings_level,
    split_type,
    feature_type,
    norms_type,
):
    """Aggregates results for all features."""

    def do():
        norms_loader = NORMS_LOADERS[norms_type]()
        _, feature_to_id, features_selected = norms_loader()
        return [
            {
                "norms-type": norms_type,
                "feature": feature,
                **load_result(
                    classifier_type,
                    embeddings_level,
                    split_type,
                    feature_type,
                    norms_loader.get_suffix(),
                    feature_to_id[feature],
                ),
            }
            for feature in tqdm(features_selected)
        ]

    os.makedirs("tmp", exist_ok=True)
    path = "tmp/{}-{}-{}-{}-{}.json".format(
        classifier_type,
        embeddings_level,
        split_type,
        feature_type,
        norms_type,
    )
    return cache_json(path, do)


def load_result_random_predictor(norms_loader):
    level = "concept"
    split = "repeated-k-fold"

    def get_random_vector(n, p):
        return np.random.choice([0, 1], size=n, p=[1 - p, p])

    def evaluate_feature(num_total, num_pos):
        n = num_total
        p = num_pos / n
        true = get_random_vector(n, p)
        pred = get_random_vector(n, p)
        return [
            {
                "score-type": score_type,
                "score": 100 * SCORE_FUNCS[score_type](true, pred),
            }
            for score_type in SPLIT_TO_SCORE_FUNCS[split]
        ]

    feature_to_concepts, _, features_selected = norms_loader()
    selected_concepts = norms_loader.load_concepts()
    num_concepts = len(selected_concepts)
    return [
        {
            "model": "random-predictor",
            "split": split,
            "level": level,
            "feature": feature,
            **{
                result["score-type"]: result["score"]
                for result in evaluate_feature(
                    num_concepts, len(feature_to_concepts[feature])
                )
            },
        }
        for feature in tqdm(features_selected)
    ]


def get_score_random_features_concept_level(norms_type):
    norms_loader = NORMS_LOADERS[norms_type]()
    _, _, features_selected = norms_loader()

    def get1(feature):
        feature_to_concepts, _, _ = norms_loader()
        num_concepts_total = len(norms_loader.load_concepts())
        num_concepts = len(set(feature_to_concepts[feature]))
        return 100 * num_concepts / num_concepts_total

    return {feature: get1(feature) for feature in tqdm(features_selected)}


def get_score_random_features_instance_level(norms_type):
    norms_loader = NORMS_LOADERS[norms_type]()
    feature_to_concepts, _, features_selected = norms_loader()
    dataset = DATASETS["things"]()

    def get1(feature):
        binary_labels = get_binary_labels(
            dataset.labels,
            feature,
            feature_to_concepts,
            dataset.class_to_label,
        )
        return 100 * np.mean(binary_labels)

    return {feature: get1(feature) for feature in tqdm(features_selected)}


def get_score_random_features(norms_type, embeddings_level="concept"):
    """Estimates the performance of a random predictor on each of the
    features.

    """
    PATHS = {
        "concept": f"tmp/score-random-{norms_type}.json",
        "instance": f"tmp/score-random-{norms_type}-instance.json",
    }
    FUNCS = {
        "concept": get_score_random_features_concept_level,
        "instance": get_score_random_features_instance_level,
    }

    os.makedirs("tmp", exist_ok=True)
    path = PATHS[embeddings_level]
    func = FUNCS[embeddings_level]
    return cache_json(path, func, norms_type)


def load_taxonomy_mcrae_x_things():
    """Collapse the McRae taxonomy on the grouped feature norms."""

    def do():
        concept_feature = load_mcrae_x_things()
        features1 = sorted(set(f for _, f in concept_feature))
        feature1_to_features = load_feature_new_to_features_mcrae()

        taxonomy_mcrae = load_taxonomy_mcrae()

        SPECIAL_CASES = {
            "made_of_wood": "visual-form_and_surface",
            "worn_in_winter": "encyclopaedic",
        }

        def get_taxonomy(feature1):
            features = feature1_to_features[feature1]
            taxonomies = [taxonomy_mcrae[f] for f in features]
            try:
                assert len(set(taxonomies)) == 1
                return taxonomies[0]
            except AssertionError:
                return SPECIAL_CASES[feature1]

        return {f: get_taxonomy(f) for f in features1}

    return cache_json("data/mcrae-x-things-taxonomy.json", do)


def load_taxonomy_mcrae():
    cols = ["Feature", "BR_Label"]
    path = "data/norms/mcrae/CONCS_FEATS_concstats_brm.txt"
    path = DIR_LOCAL / path
    df = pd.read_csv(path, sep="\t")
    df = df[cols]
    df = df.drop_duplicates()
    feature_metacategory = df.values.tolist()
    return dict(feature_metacategory)


def load_taxonomy_ours():
    def get_majority(annotations):
        counts = Counter(annotations)
        top_elem, *_ = counts.most_common(1)
        metacategory, count = top_elem
        assert count >= 2
        return metacategory

    def get1(row):
        feature, _, annot1, annot2, annot3, *_ = row
        metacategory = get_majority([annot1, annot2, annot3])
        return feature, metacategory
        # return {
        #     "feature": feature,
        #     "metacategory": metacategory,
        # }

    path = "data/gpt-feature-norms-taxonomy.csv"
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        _ = next(reader)
        _ = next(reader)
        results = [get1(row) for i, row in enumerate(reader) if i <= 99]
        results = dict(results)

    return results


def load_taxonomy_binder():
    def parse_line(line):
        metacategory, feature, *_ = line.split(" ")
        return feature, metacategory

    path = "data/binder-types.txt"
    path = DIR_LOCAL / path
    return dict(read_file(path, parse_line))


def load_taxonomy_binder_2():
    taxonomy = load_taxonomy_binder()
    return {k: BINDER_METACATEGORIES_2[v] for k, v in taxonomy.items()}


def get_results_levels_and_splits():
    classifier_type = "linear-probe"
    norms_loader = NORMS_LOADERS["generated-gpt35"]()
    _, feature_to_id, features_selected = norms_loader()

    levels_and_splits = [
        ("instance", "leave-one-concept-out"),
        ("concept", "leave-one-concept-out"),
        ("concept", "repeated-k-fold"),
    ]
    results = [
        {
            "feature": feature,
            **load_result(
                classifier_type,
                level,
                split,
                feature_type,
                norms_loader.get_suffix(),
                feature_to_id[feature],
            ),
        }
        for feature in tqdm(features_selected)
        for level, split in levels_and_splits
        for feature_type in FEATURE_TYPES
    ]
    df = pd.DataFrame(results)
    df = df.groupby(["level", "split", "model"])["score"].mean()
    df = df.reset_index()
    df = df.pivot_table(
        index=["level", "split"],
        columns="model",
        values="score",
    )
    print(df.to_csv())


def plot_results_per_metacategory(
    results, order_models=None, order_metacategory=None, metric=None,
):
    metric = metric or "score-f1-selectivity"

    df = pd.DataFrame(results)
    df["modality"] = df["model"].map(FEATURE_TYPE_TO_MODALITY)
    df["model"] = df["model"].map(FEATURE_NAMES)

    idxs = df["model"] == df["model"].iloc[0]

    model_performance = df.groupby(["model", "modality"])[metric].mean()
    model_performance = model_performance.reset_index()

    if order_models:
        order_models = [FEATURE_NAMES[m] for m in order_models]
    else:
        order_models = model_performance.sort_values(["modality", metric])["model"]

    if not order_metacategory:
        order_metacategory = sorted(df["metacategory"].unique())

    def update_metacategory(name_orig):
        name = METACATEGORY_SHORT_NAMES.get(name_orig, name_orig)
        name = name.replace(": ", "\n")
        name = name + f"\n({count_metacategory[name_orig]})"
        return name

    count_metacategory = Counter(df[idxs]["metacategory"])
    df["metacategory"] = df["metacategory"].map(update_metacategory)
    order_metacategory = [update_metacategory(m) for m in order_metacategory]

    modalities = ["image", "text"]
    modality_to_color = {
        "image": "flare_r",
        "text": "crest_r",
    }
    num_models = Counter(model_performance["modality"])
    palette = [
        c
        for m in modalities
        for c in sns.color_palette(modality_to_color[m], n_colors=num_models[m])
    ]

    # fig, ax = plt.subplots(figsize=(3.75, 20))
    sns.set(style="whitegrid", context="poster", font="Arial")
    fig, ax = plt.subplots(figsize=(24, 3.75))
    sns.barplot(
        data=df,
        y=metric,
        x="metacategory",
        hue="model",
        hue_order=order_models,
        order=order_metacategory,
        palette=palette,
        # errorbar=None,
        err_kws={"linewidth": 1},
        ax=ax,
        legend=False,
    )
    # sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=6, title="", framealpha=0.0)
    ax.set_xlabel("")
    ax.set_ylabel(SCORE_NAMES[metric])
    st.pyplot(fig)
    return fig


def get_results_per_metacategory(level, split):
    classifier_type = "linear-probe"
    taxonomy = load_taxonomy_ours()
    features = list(taxonomy.keys())

    norms_loader = NORMS_LOADERS["generated-gpt35"]()
    _, feature_to_id, features_selected = norms_loader()

    results = [
        {
            "metacategory": taxonomy[feature],
            "feature": feature,
            **load_result(
                classifier_type,
                level,
                split,
                feature_type,
                norms_loader.get_suffix(),
                feature_to_id[feature],
            ),
        }
        for feature in tqdm(features)
        for feature_type in FEATURE_TYPES
    ]

    plot_results_per_metacategory(results)


def get_results_per_metacategory_mcrae_mapped(norms_type="mcrae-mapped"):
    LOAD_TAXONOMY = {
        "mcrae-mapped": load_taxonomy_mcrae,
        "mcrae-x-things": load_taxonomy_mcrae_x_things,
    }
    assert norms_type in LOAD_TAXONOMY

    classifier_type = "linear-probe"
    taxonomy = LOAD_TAXONOMY[norms_type]()
    level = "concept"
    split = "repeated-k-fold"
    models = [
        # "random-siglip",
        "vit-mae-large",
        "max-vit-large-in21k",
        "dino-v2",
        "swin-v2-ssl",
        #
        "clip",
        "pali-gemma-224",
        "siglip-224",
        #
        "glove-840b-300d-word",
        "fasttext-word",
        "clip-word",
        "deberta-v3-contextual-layers-0-to-6-word",
        "gemma-2b-contextual-layers-9-to-18-seq-last-word",
    ]

    scores_random = get_score_random_features(norms_type)
    results = [
        r
        for m in models
        for r in load_result_features(classifier_type, level, split, m, norms_type)
    ]

    for r in results:
        r["metacategory"] = taxonomy[r["feature"]]
        r["score-f1-selectivity"] = r["score-f1"] - scores_random[r["feature"]]

    fig = plot_results_per_metacategory(results, models)
    path = "output/plots/per-metacategory-{}.pdf".format(norms_type)
    fig.savefig(path, bbox_inches="tight", transparent=True)
    # path = "output/plots/per-metacategory-{}.png".format(norms_type)
    # fig.savefig(path, bbox_inches="tight", transparent=True, dpi=800)


def get_results_per_metacategory_binder():
    # classifier_type = "linear-probe"
    # norms_type = "binder-median"
    # level = "concept"
    # split = "repeated-k-fold"
    classifier_type = "linear-regression"
    norms_type = "binder-dense"
    level = "concept"
    split = "repeated-k-fold-simple"
    metric = "score-rmse"
    taxonomy = load_taxonomy_binder()
    models = [
        # "random-siglip",
        "vit-mae-large",
        "max-vit-large-in21k",
        "dino-v2",
        "swin-v2-ssl",
        #
        "clip",
        "pali-gemma-224",
        "siglip-224",
        #
        "glove-840b-300d-word",
        "fasttext-word",
        "clip-word",
        "deberta-v3-contextual-layers-0-to-6-word",
        "gemma-2b-contextual-layers-9-to-18-seq-last-word",
    ]

    scores_random = get_score_random_features(norms_type)
    results = [
        r
        for m in models
        for r in load_result_features(classifier_type, level, split, m, norms_type)
    ]

    for r in results:
        r["metacategory"] = BINDER_METACATEGORIES_2[taxonomy[r["feature"]]]
        if metric == "score-f1-selectivity":
            r["score-f1-selectivity"] = r["score-f1"] - scores_random[r["feature"]]

    order_metacategory = [
        "Sensory",
        "Motor",
        "Space",
        "Time",
        "Social",
        "Emotion",
        "Drive",
    ]
    fig = plot_results_per_metacategory(results, models, order_metacategory, metric)
    # fig.savefig("output/plots/per-metacategory-binder.pdf", bbox_inches="tight")
    fig.savefig(
        "output/plots/per-metacategory-binder.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    # fig.savefig("output/plots/per-metacategory-binder.png", bbox_inches="tight", transparent=True, dpi=800)


def get_results_binder_norms(
    classifier_type="linear-probe",
    level="concept",
    split="repeated-k-fold",
    norm_type="binder-median",
    metric="score-f1-selectivity",
):

    MODELS = [
        # "random-siglip",
        "vit-mae-large",
        "max-vit-large-in21k",
        "dino-v2",
        "swin-v2-ssl",
        #
        "clip",
        "pali-gemma-224",
        "siglip-224",
        #
        "glove-840b-300d-word",
        "fasttext-word",
        "clip-word",
        "deberta-v3-contextual-layers-0-to-6-word",
        "gemma-2b-contextual-layers-9-to-18-seq-last-word",
    ]

    results = [
        {
            "model": model,
            **result,
        }
        for model in MODELS
        for result in load_result_features(
            classifier_type, level, split, model, norm_type
        )
    ]
    df = pd.DataFrame(results)
    if metric == "score-f1-selectivity":
        score_random = get_score_random_features(norm_type)
        df[metric] = df["score-f1"] - df["feature"].map(score_random)
    if metric in {"score-r2", "score-mse", "score-rmse", "score-mae"}:
        fmt = ".1f"
    else:
        fmt = ".0f"
    df["modality"] = df["model"].map(FEATURE_TYPE_TO_MODALITY)
    df["model"] = df["model"].map(FEATURE_NAMES)

    model_performance = df.groupby(["model", "modality"])[metric].mean()
    model_performance = model_performance.reset_index()
    # order_models = model_performance.sort_values(["modality", metric])["model"]
    order_models = [FEATURE_NAMES[m] for m in MODELS]
    order_features = read_file("data/binder-types.txt", lambda x: x.split()[1])

    taxonomy1 = load_taxonomy_binder()
    taxonomy2 = {k: BINDER_METACATEGORIES_2[v] for k, v in taxonomy1.items()}

    counts_taxonomy2 = Counter(taxonomy2.values())

    # df["metacategory-1"] = df["feature"].map(taxonomy)
    # df["metacategory-2"] = df["metacategory-1"].map(METACATEOGY_GROUPS)
    df = df.pivot_table(index="model", columns="feature", values=metric)
    df = df[order_features]
    df = df.reindex(order_models)

    rows = [
        ["Sensory", "Motor"],
        ["Space", "Time", "Social", "Emotion", "Drive"],
    ]

    rows_sizes = [[counts_taxonomy2[elem] for elem in row] for row in rows]

    st.write(df)

    vmin = df.min().min()
    vmax = df.max().max()

    def make1(ax, df, metacategory, to_hide_y):
        cols = [c for c in df.columns if taxonomy2[c] == metacategory]
        sns.heatmap(
            df[cols],
            annot=True,
            # square=True,
            cbar=False,
            fmt=fmt,
            cmap="rocket_r",
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(metacategory)
        if to_hide_y:
            ax.set_yticks([])
            ax.set_yticklabels([])
        # ax.set_yticklabels([])
        # ax.set_xticklabels([])

    def annotate_ax(ax):
        ax.text(0.5, 0.5, "ax%d" % (i + 1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)

    sns.set(style="whitegrid", font="Arial")

    fig = plt.figure(figsize=(12.75, 11.50))
    gs = gridspec.GridSpec(2, sum(rows_sizes[0]), hspace=0.4)
    for i, row in enumerate(rows):
        s = 0
        for j, elem in enumerate(row):
            e = s + rows_sizes[i][j]
            fig.add_subplot(gs[i, s:e])
            s = e

    i = 0
    for row in rows:
        for j, elem in enumerate(row):
            make1(fig.axes[i], df, elem, j > 0)
            # annotate_ax(fig.axes[i])
            i += 1

    # fig.tight_layout()
    st.pyplot(fig)
    fig.savefig("output/plots/binder-results-per-norm.pdf", bbox_inches="tight")


def get_classifiers_agreement_binder_norms():
    level = "concept"
    split = "repeated-k-fold"

    thresh = 4
    norms_loader = NORMS_LOADERS[f"binder-{thresh}"]()
    feature_to_concepts, feature_to_id, features_selected = norms_loader()

    def get_agreement(level, split, feature_type, feature, kwargs_path):
        def get_path(*, feature_norm_str, feature_id):
            return OUTPUT_PATH.format(
                "linear-probe",
                level,
                split,
                DATASET_NAME,
                feature_type,
                feature_norm_str,
                feature_id,
            )

        def cossim(c1, c2):
            norm1 = np.linalg.norm(c1)
            norm2 = np.linalg.norm(c2)
            return np.dot(c1, c2) / (norm1 * norm2)

        path = get_path(**kwargs_path) + ".pkl"
        with open(path, "rb") as f:
            outputs = pickle.load(f)

        coefs = [output["clf"].coef_.squeeze() for output in outputs]
        sims = [cossim(c1, c2) for c1, c2 in combinations(coefs, 2)]

        sims_mean = np.mean(sims)
        sims_std = np.std(sims)

        return {
            "feature": feature,
            "level": level,
            "model": feature_type,
            "split": split,
            "sim-mean": sims_mean,
            "sim-std": sims_std,
            "sim": "{:.2f}±{:.1f}".format(sims_mean, 2 * sims_std),
        }

    results = [
        get_agreement(
            level,
            split,
            feature_type,
            feature,
            {
                "feature_norm_str": norms_loader.get_suffix(),
                "feature_id": feature_to_id[feature],
            },
        )
        for feature in tqdm(features_selected)
        for feature_type in FEATURE_TYPES
    ]
    df = pd.DataFrame(results)
    df = df.pivot_table(index="feature", columns="model", values="sim-mean")
    df = df.reset_index()
    df["num-concepts"] = df["feature"].apply(lambda x: len(feature_to_concepts[x]))
    df.to_csv("output/classifier-agreement-binder-norms-2.csv")


class LoadClassifiers:
    def __init__(self, classifier_type, embeddings_level, split_type, norms_type):
        self.classifier_type = classifier_type
        self.embeddings_level = embeddings_level
        self.split_type = split_type
        self.norms_type = norms_type

        self.norms_loader = NORMS_LOADERS[norms_type]()
        _, self.feature_to_id, self.features_selected = self.norms_loader()

    def clf_to_numpy(self, clf):
        w = clf.coef_.squeeze()
        b = clf.intercept_
        return np.hstack((w, b))

    def load_clfs(self, path):
        with open(path + ".pkl", "rb") as f:
            res = pickle.load(f)
        return [self.clf_to_numpy(r["clf"]) for r in res]

    def load_clfs_feature(self, model, feature_id):
        path = OUTPUT_PATH.format(
            self.classifier_type,
            self.embeddings_level,
            self.split_type,
            DATASET_NAME,
            model,
            self.norms_type,
            feature_id,
        )
        return self.load_clfs(path)

    def load_clfs_features(self, model):
        clfs = [
            clf
            for feature in tqdm(self.features_selected)
            for clf in self.load_clfs_feature(model, self.feature_to_id[feature])
        ]
        return np.stack(clfs)

    def __call__(self, model, to_use_intercept=True, to_l2_normalize=True):
        os.makedirs("tmp", exist_ok=True)
        X = cache_np(
            "tmp/clfs-{}.npy".format(model),
            self.load_clfs_features,
            self.classifier_type,
            self.embeddings_level,
            self.split_type,
            model,
            self.norms_type,
        )

        if not to_use_intercept:
            X = X[:, :-1]

        if to_l2_normalize:
            norm = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / norm

        return X


def analyse_classifiers():
    from probing_norms.scripts.eval_norm_correlations import (
        get_best_supercategory_matches,
    )

    load_classifiers = LoadClassifiers(
        classifier_type="linear-probe",
        embeddings_level="concept",
        split_type="repeated-k-fold",
        norms_type="mcrae-x-things",
    )

    PROJ_METHODS = {
        "UMAP": UMAP,
        "PCA": PCA,
        "t-SNE": TSNE,
    }

    AGG_FUNCS = {
        "mean": lambda xs: np.mean(xs, axis=0),
        "first": lambda xs: xs[0],
        "none": None,
    }

    MODELS = [
        "random-siglip",
        "dino-v2",
        "swin-v2-ssl",
        "clip",
        "pali-gemma-224",
        "siglip-224",
        "clip-word",
        "gemma-2b-contextual-layers-9-to-18-seq-last-word",
    ]
    MODEL_NAMES = [FEATURE_NAMES[m] for m in MODELS]

    def aggregate(values, labels, func):
        if func is None:
            return values, labels
        else:
            labels_agg = sorted(set(labels))
            labels = np.array(labels)
            label_to_values = {label: values[labels == label] for label in labels_agg}
            values_agg = [func(label_to_values[label]) for label in labels_agg]
            values_agg = np.vstack(values_agg)
            return values_agg, labels_agg

    METRICS = {
        "silhouette": silhouette_score,
        "davies-bouldin": davies_bouldin_score,
        "calinski-harabasz": calinski_harabasz_score,
    }

    def evaluate(X, y):
        return {m: METRICS[m](X, y) for m in METRICS}

    st.set_page_config(layout="wide")

    with st.sidebar:
        model_name = st.selectbox(
            "Model", MODEL_NAMES, index=MODEL_NAMES.index("CLIP (image)")
        )
        proj_method_name = st.selectbox(
            "Projection method",
            list(PROJ_METHODS.keys()),
            help="How to project the classifier weights into 2D space.",
        )
        to_use_intercept = st.checkbox(
            "Use intercept",
            value=True,
            help="Whether to use the intercept term as part of the classifier weights.",
        )
        to_l2_normalize = st.checkbox(
            "L2 normalize",
            value=True,
            help="Whether to L2 normalize the classifier weights (corresponds to using cosine similarity).",
        )
        agg_classifiers = st.selectbox(
            "Aggregate classifiers",
            list(AGG_FUNCS.keys()),
            help="We train ten classifiers for each attribute based on two 2 × 5-fold cross validation splits. How to aggregate them.",
        )
        st.markdown("---")
        st.markdown("### Clustering hyperparameters")
        clustering_space = st.selectbox("Clustering space", ["original", "UMAP"])
        if clustering_space == "UMAP":
            clustering_space_dim = st.number_input(
                "Clustering space dimension",
                min_value=2,
                value=10,
            )
        else:
            clustering_space_dim = None
        min_cluster_size = st.number_input(
            "Minimum cluster size",
            min_value=2,
            # max_value=100,
            value=5,
        )
        min_samples = st.number_input(
            "Minimum samples",
            min_value=1,
            # max_value=100,
            value=min_cluster_size,
        )

    path = "/tmp/attribute-to-supercategory.json"
    attribute_to_supercategory_and_score = cache_json(
        path,
        get_best_supercategory_matches,
        load_classifiers.norms_type,
    )
    attribute_to_supercategory = {
        a: cs[0] for a, cs in attribute_to_supercategory_and_score.items()
    }
    attribute_to_supercategory_score = {
        a: cs[1] for a, cs in attribute_to_supercategory_and_score.items()
    }

    model = MODELS[MODEL_NAMES.index(model_name)]
    Proj = PROJ_METHODS[proj_method_name]

    taxonomy = load_taxonomy_mcrae_x_things()
    features_selected = load_classifiers.features_selected
    X = load_classifiers(model, to_use_intercept, to_l2_normalize)
    F = [feature for feature in features_selected for _ in range(10)]
    X1, F = aggregate(X, F, AGG_FUNCS[agg_classifiers])

    proj = Proj(n_components=2)
    X2 = proj.fit_transform(X1)

    df = pd.DataFrame(X2, columns=["x₁", "x₂"])
    df["feature"] = F
    df["taxonomy"] = df["feature"].map(taxonomy)
    df["supercategory"] = df["feature"].map(attribute_to_supercategory)
    df["supercategory-score"] = df["feature"].map(attribute_to_supercategory_score)

    # Y = df["taxonomy"].values
    # scores = evaluate(X1, Y)
    # st.write(scores)

    W = 400
    H = 350

    def make_fig(type_):
        TYPES_STR = {
            "taxonomy:N": "attribute type",
            "supercategory:N": "supercategory",
        }
        selector = alt.selection_interval(empty=True)
        tooltip = ["feature", "taxonomy", "supercategory", "supercategory-score"]

        if type_ == "taxonomy:N":
            kwargs = {}
        else:
            kwargs = {"opacity": "supercategory-score:Q"}

        fig1 = (
            alt.Chart(df)
            .mark_circle(size=60)
            .add_params(selector)
            .encode(
                x=alt.X("x₁:Q").scale(zero=False),
                y=alt.Y("x₂:Q").scale(zero=False),
                color=type_,
                tooltip=tooltip,
                **kwargs,
            )
            .properties(
                width=W,
                height=H,
                title="Color denotes {}. Select area to zoom in.".format(
                    TYPES_STR[type_]
                ),
            )
        )

        fig2 = (
            alt.Chart(df)
            .mark_text()
            .encode(
                x=alt.X("x₁:Q").scale(zero=False),
                y=alt.Y("x₂:Q").scale(zero=False),
                text="feature",
                opacity=alt.condition(selector, alt.value(1.0), alt.value(0.0)),
                tooltip=tooltip,
            )
            .transform_filter(selector)
            .interactive()
            .properties(
                width=W,
                height=H,
                title="Attribute names in the selected area",
            )
        )

        fig = fig1 | fig2
        fig = fig.configure_legend(
            title=None,
            orient="right",
        ).configure_view(
            strokeWidth=0,
        )
        return fig

    fig1 = make_fig("taxonomy:N")
    st.altair_chart(fig1)

    fig2 = make_fig("supercategory:N")
    st.altair_chart(fig2)

    if clustering_space == "original":
        X3 = X1
    else:
        proj = UMAP(n_components=clustering_space_dim)
        X3 = proj.fit_transform(X1)

    C = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(
        X3
    )
    df["cluster"] = C

    selection = alt.selection_point(
        fields=["cluster"],
        bind="legend",
        # empty="all",
    )
    fig3 = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=alt.X("x₁:Q").scale(zero=False),
            y=alt.Y("x₂:Q").scale(zero=False),
            color=alt.Color("cluster:N", title="HDBSCAN cluster"),
            # shape=alt.Shape("taxonomy:N", title="Taxonomy"),
            tooltip=["feature", "taxonomy", "cluster"],
            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.2)),
        )
        .interactive()
        .properties(
            width=1.5 * W,
            height=1.5 * H,
            title="HDBSCAN clusters",
        )
        .add_params(selection)
    )

    fig3 = fig3.configure_legend(
        title=None,
        orient="right",
    ).configure_view(
        strokeWidth=0,
    )

    st.altair_chart(fig3)

    num_clusters = len(set(C))
    has_noise = -1 in C
    num_noisy_samples = sum(C == -1)
    frac_noise_samples = 100 * num_noisy_samples / len(C)

    st.markdown(
        """
    - Number of clusters: {}
    - Number of clusters (without noise category): {}
    - Number of noisy samples: {}
    - Fraction of noisy samples: {:.1f}%
    """.format(
            num_clusters,
            num_clusters - (1 if has_noise else 0),
            num_noisy_samples,
            frac_noise_samples,
        )
    )


def analyse_classifiers_quantiative(labels_type):
    from probing_norms.scripts.eval_norm_correlations import (
        get_best_supercategory_matches,
    )

    load_classifiers = LoadClassifiers(
        classifier_type="linear-probe",
        embeddings_level="concept",
        split_type="repeated-k-fold",
        norms_type="mcrae-x-things",
    )

    AGG_FUNCS = {
        "mean": lambda xs: np.mean(xs, axis=0),
        "first": lambda xs: xs[0],
        "none": None,
    }

    MODELS = [
        "random-siglip",
        "dino-v2",
        "swin-v2-ssl",
        "clip",
        "pali-gemma-224",
        "siglip-224",
        "clip-word",
        "gemma-2b-contextual-layers-9-to-18-seq-last-word",
    ]

    def load_supercategory():
        data = cache_json(
            "/tmp/attribute-to-supercategory.json",
            get_best_supercategory_matches,
            "mcrae-x-things",
        )
        return {a: cs[0] for a, cs in data.items()}

    ATTRIBUTE_TO_LABEL = {
        "taxonomy": load_taxonomy_mcrae_x_things,
        "supercategory": load_supercategory,
    }

    def aggregate(values, labels, func):
        if func is None:
            return values, labels
        else:
            labels_agg = sorted(set(labels))
            labels = np.array(labels)
            label_to_values = {label: values[labels == label] for label in labels_agg}
            values_agg = [func(label_to_values[label]) for label in labels_agg]
            values_agg = np.vstack(values_agg)
            return values_agg, labels_agg

    METRICS = {
        "silhouette": silhouette_score,
        "davies-bouldin": davies_bouldin_score,
        "calinski-harabasz": calinski_harabasz_score,
    }

    def evaluate(X, y):
        return {m: METRICS[m](X, y) for m in METRICS}

    # taxonomy = load_taxonomy_mcrae_x_things()
    attribute_to_label = ATTRIBUTE_TO_LABEL[labels_type]()
    features_selected = load_classifiers.features_selected

    def do1(model, i, to_shuffle_labels=False):
        X = load_classifiers(model, to_use_intercept=True, to_l2_normalize=True)
        F = [feature for feature in features_selected for _ in range(10)]
        I = [i for _ in features_selected for i in range(10)]

        F = np.array(F)
        I = np.array(I)

        X = X[I == i]
        F = F[I == i]

        # X, F = aggregate(X, F, AGG_FUNCS["mean"])
        T = [attribute_to_label[f] for f in F]
        if to_shuffle_labels:
            random.shuffle(T)

        return evaluate(X, T)

    bools = [False, True]
    results = [
        {
            "model": model,
            "shuffle labels?": to_shuffle,
            "i": i,
            **do1(model, i, to_shuffle),
        }
        for model in MODELS
        for i in range(5)
        for to_shuffle in bools
    ]
    df = pd.DataFrame(results)

    df = df.groupby(["model", "shuffle labels?"]).mean()
    df = df.reset_index()

    df = df.pivot_table(
        index="model",
        columns="shuffle labels?",
        values=list(METRICS.keys()),
    )
    df = df.reindex(MODELS)
    df = df.reset_index()
    df["model"] = df["model"].map(FEATURE_NAMES)

    st.write(df)
    st.markdown("```\n" + df.to_csv() + "\n```")


def get_results_paper_table_main_row(*modCels, norm_types=None):
    assert len(models) > 0, "At least one model must be provided."

    classifier_type = "linear-probe"
    embeddings_level = "concept"
    split_type = "repeated-k-fold"
    # norm_types = ["mcrae-x-things", "mcrae-mapped", "binder-median"]
    norm_types = norm_types or ["mcrae-x-things", "binder-median"]

    scores_random_features = {k: get_score_random_features(k) for k in norm_types}

    def add_f1_sel(results, norm_type):
        for r in results:
            r["score-f1-selectivity"] = (
                r["score-f1"] - scores_random_features[norm_type][r["feature"]]
            )
        return results

    results = [
        r
        for n in norm_types
        for m in models
        for r in add_f1_sel(
            load_result_features(classifier_type, embeddings_level, split_type, m, n), n
        )
    ]
    cols = [
        "model",
        "norms-type",
        "score-f1",
        "score-precision",
        "score-recall",
        "score-f1-selectivity",
    ]
    df = pd.DataFrame(results)
    df = df[cols]
    df = df.groupby(["norms-type", "model"]).mean()
    df = df.reset_index()
    df = df.pivot_table(index="model", columns="norms-type")
    cols = [
        (s, n)
        for n in norm_types
        for s in ["score-precision", "score-recall", "score-f1", "score-f1-selectivity"]
    ]
    df = df[cols]
    df = df.reindex(models)
    df = df.reset_index()
    df["model"] = df["model"].map(lambda x: FEATURE_NAMES.get(x, x))
    print(df.to_latex(float_format="%.1f", index=False))


def compute_f1_sel_df(df, norms_type, **_):
    scores_random_features = get_score_random_features(norms_type)
    return df["score-f1"] - df["feature"].map(scores_random_features)


def compute_f1_sel_norm_df(df, norms_type, **_):
    scores_random_features = get_score_random_features(norms_type)
    num = df["score-f1"] - df["feature"].map(scores_random_features)
    den = 100 - df["feature"].map(scores_random_features)
    return 100 * num / den


COMPUTE_METRICS = {
    "score-f1-selectivity": compute_f1_sel_df,
    "score-f1-selectivity-norm": compute_f1_sel_norm_df,
}

def add_metric(df, metric, **kwargs):
    df[metric] = COMPUTE_METRICS[metric](df, **kwargs)
    return df


def get_results_paper_table_main_acl_camera_ready(*models):
    SETTINGS = [
        {
            "classifier_type": "linear-probe",
            "embeddings_level": "concept",
            "split_type": "repeated-k-fold",
            "norms_type": "mcrae-x-things",
            "metric": "score-f1-selectivity",
        },
        {
            "classifier_type": "linear-regression",
            "embeddings_level": "concept",
            "split_type": "repeated-k-fold-simple",
            "norms_type": "binder-dense",
            "metric": "score-rmse",
        },
    ]

    def get_results_1(models, metric, **settings):
        results = [
            result
            for model in models
            for result in load_result_features(feature_type=model, **settings)
        ]
        df = pd.DataFrame(results)

        if metric in COMPUTE_METRICS:
            df = add_metric(df, **settings)

        cols = ["model", metric]
        df = df[cols]
        df = df.groupby("model").aggregate(["mean", "min", "max"])
        return df

    if len(models) == 0:
        models = MAIN_TABLE_MODELS

    dfs = {s["norms_type"]: get_results_1(models, **s) for s in SETTINGS}
    df = pd.concat(dfs, axis=1)
    df = df.reindex(models)
    df = df.reset_index()
    df["model"] = df["model"].map(lambda x: FEATURE_NAMES.get(x, x))

    def row_to_string_1(row, setting):
        m = setting["metric"]
        n = setting["norms_type"]
        fmt = METRIC_TO_FMT[m]
        return [
            fmt.format(row[(n, m, "mean")]),
            r"\annot{" + fmt.format(row[(n, m, "min")]) + r"}",
            r"\annot{--}",
            r"\annot{" + fmt.format(row[(n, m, "max")]) + r"}",
        ]

    def concat(xss):
        return [x for xs in xss for x in xs]

    def row_to_string(row):
        return " & ".join(
            [row["model"].item()] + concat(row_to_string_1(row, s) for s in SETTINGS)
        )

    print(" \\\\ \n".join([row_to_string(row) for _, row in df.iterrows()]))


def get_results_paper_table_full_main_acl_camera_ready():
    SETTINGS = [
        {
            "classifier_type": "linear-probe",
            "embeddings_level": "concept",
            "split_type": "repeated-k-fold",
            "norms_type": "mcrae-x-things",
            "metrics": [
                "score-precision",
                "score-recall",
                "score-f1",
                "score-f1-selectivity",
            ],
        },
        {
            "classifier_type": "linear-probe",
            "embeddings_level": "concept",
            "split_type": "repeated-k-fold",
            "norms_type": "binder-median",
            "metrics": [
                "score-precision",
                "score-recall",
                "score-f1",
                "score-f1-selectivity",
            ],
        },
        {
            "classifier_type": "linear-regression",
            "embeddings_level": "concept",
            "split_type": "repeated-k-fold-simple",
            "norms_type": "binder-dense",
            "metrics": [
                "score-rmse",
                "score-mae",
            ],
        },
    ]

    def get_results_1(models, metrics, **settings):
        results = [
            result
            for model in models
            for result in load_result_features(feature_type=model, **settings)
        ]
        df = pd.DataFrame(results)

        if "score-f1-selectivity" in metrics:
            norm_type = settings["norms_type"]
            scores_random_features = get_score_random_features(norm_type)
            df["score-f1-selectivity"] = df["score-f1"] - df["feature"].map(
                scores_random_features
            )

        cols = ["model"] + metrics
        df = df[cols]
        df = df.groupby("model").mean()
        return df

    dfs = {s["norms_type"]: get_results_1(MAIN_TABLE_MODELS, **s) for s in SETTINGS}
    df = pd.concat(dfs, axis=1)
    df = df.reindex(MAIN_TABLE_MODELS)
    df = df.reset_index()
    df["model"] = df["model"].map(lambda x: FEATURE_NAMES.get(x, x))

    def row_to_string_1(row, n, m):
        key = (n, m)
        fmt = METRIC_TO_FMT[m]
        return fmt.format(row[key])

    def row_to_string(row):
        return " & ".join(
            [row["model"].item()]
            + [
                row_to_string_1(row, s["norms_type"], m)
                for s in SETTINGS
                for m in s["metrics"]
            ]
        )

    print(" \\\\ \n".join([row_to_string(row) for _, row in df.iterrows()]))


def get_results_paper_table_main():
    get_results_paper_table_main_row(*MAIN_TABLE_MODELS)


def get_results_contextualised_language_models():
    models = [
        "gemma-2b-no-space-word",
        "gemma-2b-word",
        "gemma-2b-contextual-layer-1-word",
        "gemma-2b-contextual-last-word",
        "gemma-2b-contextual-last-seq-last-word",
        "gemma-2b-contextual-layers-0-to-6-word",
        "gemma-2b-contextual-layers-0-to-9-word",
        "gemma-2b-contextual-layers-9-to-18-word",
        "gemma-2b-contextual-layers-9-to-18-seq-last-word",
        "gemma-2b-contextual-50-last-word",
        "gemma-2b-contextual-50-constrained-last-word",
        #
        "deberta-v3-contextual-last-word",
        "deberta-v3-contextual-layers-0-to-4-word",
        "deberta-v3-contextual-layers-0-to-6-word",
        #
        "gpt2-contextual-last-word",
        #
        "bert-base-uncased-contextual-layers-0-to-4-word",
        "bert-base-uncased-contextual-layers-0-to-6-word",
    ]
    get_results_paper_table_main_row(*models, norm_types=["mcrae-x-things"])


def get_results_all_one_norm(feature):
    classifier_type = "linear-probe"
    embeddings_level = "concept"
    splits_type = "repeated-k-fold"
    norms_type = "mcrae-mapped"

    feature_to_random_score = get_score_random_features(norms_type)
    results = [
        result
        for m in MAIN_TABLE_MODELS
        for result in load_result_features(
            classifier_type, embeddings_level, splits_type, m, norms_type
        )
        if result["feature"] == feature
    ]
    for r in results:
        r["score-f1-selectivity"] = (
            r["score-f1"] - feature_to_random_score[r["feature"]]
        )
    df = pd.DataFrame(results)
    df = df.pivot_table(index="model", columns="feature", values="score-f1-selectivity")
    df = df.reindex(MAIN_TABLE_MODELS)
    df = df.reset_index()
    df["model"] = df["model"].map(FEATURE_NAMES)
    print(df.to_string(index=False))


def get_results_per_feature_norm():
    EMBS = ["siglip-224", "fasttext-word", "glove-840b-300d-word", "gemma-2b-word"]
    classifier_type = "linear-probe"
    embeddings_level = "concept"
    splits_type = "repeated-k-fold"
    norms_type = "mcrae-mapped"
    norms_loader = NORMS_LOADERS[norms_type]()
    feature_to_concepts, feature_to_id, features_selected = norms_loader()

    import random

    def random_score(feature, score_func):
        n = len(feature_to_concepts[feature])
        total_num_concepts = 1854
        pred = [random.random() <= 0.2 for _ in range(total_num_concepts)]
        true = np.zeros(total_num_concepts)
        true[:n] = 1
        return score_func(true, pred)

    results = [
        result
        for m in EMBS
        for result in load_result_features(
            classifier_type, embeddings_level, splits_type, m, norms_type
        )
    ]
    df = pd.DataFrame(results)
    df = df.pivot_table(index="feature", columns="model", values="score")
    df = df.reset_index()
    df["num-concepts"] = df["feature"].apply(lambda x: len(feature_to_concepts[x]))
    df["f1-random-pred"] = df["feature"].apply(
        partial(random_score, score_func=f1_score)
    )
    df["recall-random-pred"] = df["feature"].apply(
        partial(random_score, score_func=recall_score)
    )
    # cols = ["feature", "num-concepts", "f1-random-pred"] + EMBS
    # df = df[cols]
    # df.to_csv("output/results-per-feature-2.csv")

    fig, ax = plt.subplots()
    # sns.scatterplot(data=df, x="siglip-224", y="fasttext-word", ax=ax)
    # ax.plot([0, 100], [0, 100], color="gray", linestyle="--")
    # ax.set_aspect("equal", adjustable="box")
    fig = sns.lmplot(data=df, x="num-concepts", y="gemma-2b-word")
    # fig = sns.lmplot(data=df, x="num-concepts", y="recall-random-pred")
    fig.set_axis_labels("Number of concepts", "Score")
    st.pyplot(fig)


def get_results_random_predictor():
    split = "repeated-k-fold"

    def load_results(norms_type):
        norms_loader = NORMS_LOADERS[norms_type]()
        results = load_result_random_predictor(norms_loader)
        df = pd.DataFrame(results)
        cols_score = SPLIT_TO_SCORE_FUNCS[split]
        df = df.groupby("model")[cols_score].mean()
        return df

    dfs = {
        NORMS_NAMES[norms_type]: load_results(norms_type)
        for norms_type in ["mcrae-mapped", "binder-median"]
    }
    df = pd.concat(dfs, axis=1)
    df = df.reset_index()
    df["model"] = df["model"].map(FEATURE_NAMES)
    print(df.to_latex(float_format="%.1f", index=False))


def compare_two_models_scatterplot_ax(ax, model1, model2, legend="auto"):
    classifier_type = "linear-probe"
    embeddings_level = "concept"
    splits_type = "repeated-k-fold"
    norms_type = "mcrae-x-things"
    # norms_loader = NORMS_LOADERS[norms_type]()
    metric = "score-f1-selectivity"
    taxonomy = load_taxonomy_mcrae()

    results = [
        r
        for m in [model1, model2]
        for r in load_result_features(
            classifier_type, embeddings_level, splits_type, m, norms_type
        )
    ]

    df = pd.DataFrame(results)
    df = add_metric(df, metric, norms_type=norms_type)
    df["metacategory"] = df["feature"].map(taxonomy)
    df["metacategory"] = df["metacategory"].map(lambda x: METACATEGORY_NAMES.get(x, x))
    cols = ["feature", "metacategory", "model", metric]
    df = df[cols]
    df = df.set_index(["feature", "metacategory", "model"]).unstack(-1)
    df = df.reset_index()
    df.columns = ["-".join(c for c in cols if c).strip() for cols in df.columns.values]

    st.write(df)

    def pareto_front(points, dominates):
        return [
            p1
            for p1 in points
            if not any(dominates(p2, p1) for p2 in points if p1 != p2)
        ]

    def add_texts(ax, df):
        cols = [
            f"{metric}-{model1}",
            f"{metric}-{model2}",
            "feature",
        ]
        points = df[cols].values.tolist()
        points = [tuple(p) for p in points]
        xs = [x for x, _, _ in points]
        ys = [y for _, y, _ in points]
        points1 = pareto_front(
            points,
            dominates=lambda p, q: p[0] > q[0] and p[1] < q[1],
        )
        points1 = [p for p in points1 if p[0] >= 30]
        points2 = pareto_front(
            points,
            dominates=lambda p, q: p[0] < q[0] and p[1] > q[1],
        )
        points2 = [p for p in points2 if p[1] >= 30]
        points = points1 + points2
        points = list(set(points))
        texts = [
            ax.text(
                x,
                y,
                normalize_feature_name(word),
                ha="center",
                va="center",
                size=8,
            )
            for x, y, word in points
        ]
        adjust_text(
            texts,
            x=xs,
            y=ys,
            ax=ax,
            expand=(1.2, 2.1),
            force_text=(0.2, 0.6),
            # force_points=1.0,
            arrowprops=dict(arrowstyle="-", color="b", alpha=0.5),
        )

    metacategories = sorted(df["metacategory"].unique())
    sns.scatterplot(
        df,
        x=f"{metric}-{model1}",
        y=f"{metric}-{model2}",
        hue="metacategory",
        hue_order=metacategories,
        # marker="metacategory",
        legend=legend,
        ax=ax,
    )
    add_texts(ax, df)
    if legend:
        # sns.move_legend(ax, "lower right", bbox_to_anchor=(1, 1), ncol=2, title="")
        sns.move_legend(
            ax, "upper left", bbox_to_anchor=(1, 1), ncol=1, title="", framealpha=0.0
        )
    ax.plot([0, 100], [0, 100], color="gray", linestyle="--")
    ax.set_xlabel("{} · {}".format(SCORE_NAMES[metric], FEATURE_NAMES[model1]))
    ax.set_ylabel("{} · {}".format(SCORE_NAMES[metric], FEATURE_NAMES[model2]))
    ax.set_aspect("equal", adjustable="box")


def compare_two_models_scatterplot(model1, model2):
    sns.set(style="whitegrid", font="Arial")
    fig, ax = plt.subplots()
    compare_two_models_scatterplot_ax(ax, model1, model2, legend="auto")
    st.pyplot(fig)
    fig.savefig(
        f"output/plots/scatterplot-{model1}-vs-{model2}.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def compare_two_models_scatterplot_2():
    sns.set(style="whitegrid", font="Arial")
    models = [
        ["gemma-2b-contextual-layers-9-to-18-seq-last-word", "swin-v2-ssl"],
        ["clip-word", "clip"],
    ]
    fig, axs = plt.subplots(figsize=(10, 10), ncols=2, nrows=1, sharex=True)
    compare_two_models_scatterplot_ax(axs[0], *models[0], legend="auto")
    compare_two_models_scatterplot_ax(axs[1], *models[1], legend=False)
    sns.move_legend(axs[0], "lower left", bbox_to_anchor=(0, 1), ncol=5, title="")
    st.pyplot(fig)
    fig.savefig(
        f"output/plots/scatterplot-model-comparison.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def get_correlation_between_models(norms_type):
    classifier_type = "linear-probe"
    embeddings_level = "concept"
    splits_type = "repeated-k-fold"
    norms_loader = NORMS_LOADERS[norms_type]()

    models = [
        "random-siglip",
        "vit-mae-large",
        "max-vit-large",
        "max-vit-large-in21k",
        # "swin-v2",
        "dino-v2",
        "swin-v2-ssl",
        #
        "llava-1.5-7b",
        "qwen2.5-vl-3b-instruct",
        "clip",
        "pali-gemma-224",
        "siglip-224",
        #
        "glove-840b-300d-word",
        "fasttext-word",
        "numberbatch-word",
        "clip-word",
        "deberta-v3-contextual-layers-0-to-6-word",
        "gemma-2b-contextual-layers-9-to-18-seq-last-word",
    ]

    results = [
        r
        for m in models
        for r in load_result_features(
            classifier_type, embeddings_level, splits_type, m, norms_type
        )
    ]
    cols = ["feature", "model", "score-f1"]
    # feature_to_random_score = {
    # feature: get_score_random(norms_loader, feature)
    # for feature in norms_loader.load_concepts()
    # }

    df = pd.DataFrame(results)
    df = df[cols]
    df["model"] = df["model"].map(lambda x: FEATURE_NAMES.get(x, x))
    # df["score-random"] = df["feature"].map(lambda x: get_score_random(norms_loader, x))
    # df["score-f1-selectivity"] = df["score-f1"] - df["score-random"]
    df = df.pivot_table(index="feature", columns="model", values="score-f1")
    corr_matrix = df.corr()

    sns.set(style="whitegrid", font="Arial")
    fig, ax = plt.subplots(figsize=(7, 7))
    models_names = [FEATURE_NAMES[m] for m in models]
    corr_matrix = corr_matrix.loc[models_names, models_names]
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        square=True,
        cbar=False,
        cmap=sns.cm.rocket_r,
        ax=ax,
    )

    for text in ax.texts:
        t = text.get_text()
        if t.startswith("0."):
            t1 = t.lstrip("0")
        elif t == "1.00":
            t1 = "1.0"
        else:
            t1 = t
        text.set_text(t1)

    ax.set_xlabel("")
    ax.set_ylabel("")
    st.pyplot(fig)

    fig.savefig(
        f"output/plots/correlation-between-models-{norms_type}.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def get_correlation_between_models_2():
    SETTINGS = [
        {
            "classifier_type": "linear-probe",
            "embeddings_level": "concept",
            "split_type": "repeated-k-fold",
            "norms_type": "mcrae-x-things",
            "metric": "score-f1",
        },
        {
            "classifier_type": "linear-regression",
            "embeddings_level": "concept",
            "split_type": "repeated-k-fold-simple",
            "norms_type": "binder-dense",
            "metric": "score-rmse",
        },
    ]

    models = [
        "random-siglip",
        "vit-mae-large",
        "max-vit-large",
        "max-vit-large-in21k",
        # "swin-v2",
        "dino-v2",
        "swin-v2-ssl",
        #
        "llava-1.5-7b",
        "qwen2.5-vl-3b-instruct",
        "clip",
        "pali-gemma-224",
        "siglip-224",
        #
        "glove-840b-300d-word",
        "fasttext-word",
        "numberbatch-word",
        "clip-word",
        "deberta-v3-contextual-layers-0-to-6-word",
        "gemma-2b-contextual-layers-9-to-18-seq-last-word",
    ]

    def plot1(ax, metric, **settings):
        results = [
            r for m in models for r in load_result_features(**settings, feature_type=m)
        ]
        cols = ["feature", "model", metric]

        df = pd.DataFrame(results)
        df = df[cols]
        df["model"] = df["model"].map(lambda x: FEATURE_NAMES.get(x, x))
        # df["score-random"] = df["feature"].map(lambda x: get_score_random(norms_loader, x))
        # df["score-f1-selectivity"] = df["score-f1"] - df["score-random"]
        df = df.pivot_table(index="feature", columns="model", values=metric)
        corr_matrix = df.corr()

        models_names = [FEATURE_NAMES[m] for m in models]
        corr_matrix = corr_matrix.loc[models_names, models_names]
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt="0.2f",
            square=True,
            cbar=False,
            cmap=sns.cm.rocket_r,
            ax=ax,
        )
        norms_type = settings["norms_type"]
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(NORMS_NAMES[norms_type])
        for text in ax.texts:
            t = text.get_text()
            if t.startswith("0."):
                t1 = t.lstrip("0")
            elif t == "1.00":
                t1 = "1.0"
            else:
                t1 = t
            text.set_text(t1)

    sns.set(style="whitegrid", font="Arial")
    fig, axs = plt.subplots(
        figsize=(11, 19.5), ncols=2, nrows=1, sharex=True, sharey=True
    )
    plot1(axs[0], **SETTINGS[0])
    plot1(axs[1], **SETTINGS[1])
    fig.tight_layout()
    st.pyplot(fig)

    fig.savefig(
        f"output/plots/correlation-between-models.png",
        bbox_inches="tight",
        transparent=True,
        dpi=800,
    )
    fig.savefig(
        f"output/plots/correlation-between-models.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def prepare_results_for_stella():
    classifier_type = "linear-probe"
    embeddings_level = "concept"
    splits_type = "repeated-k-fold"
    norms_type = "mcrae-mapped"
    norms_loader = NORMS_LOADERS[norms_type]()
    # models = ["dino-v2", "gemma-2b-contextual-last-word"]
    models = [
        "pali-gemma-224",
        "siglip-224",
        "vit-mae-large",
        "dino-v2",
        "swin-v2",
        "swin-v2-ssl",
        "max-vit-large",
        "max-vit-large-in21k",
        "random-siglip",
        "clip",
        "fasttext-word",
        "glove-840b-300d-word",
        "deberta-v3-contextual-layers-0-to-6-word",
        "gemma-2b-contextual-layers-9-to-18-seq-last-word",
        "clip-word",
    ]

    feature_to_concepts, _, _ = norms_loader()
    taxonomy = load_taxonomy_mcrae()

    results = [
        {
            "model": m,
            **result,
        }
        for m in models
        for result in load_result_features(
            classifier_type, embeddings_level, splits_type, m, norms_type
        )
    ]

    # Add metacategory
    for r in results:
        concepts = sorted(set(feature_to_concepts[r["feature"]]))
        r["concepts"] = concepts
        r["metacategory"] = taxonomy[r["feature"]]
        r["score-random"] = get_score_random(norms_loader, r["feature"])
        r["score-f1-selectivity"] = r["score-f1"] - r["score-random"]

    with open("output/results-for-stella.json", "w") as f:
        json.dump(results, f, indent=2)


def prepare_classifiers_for_stella(model):
    norms_type = "mcrae-mapped"
    norms_loader = NORMS_LOADERS[norms_type]()
    _, feature_to_id, features_selected = norms_loader()
    level = "concept"
    split = "repeated-k-fold"

    def get_path(feature):
        feature_id = feature_to_id[feature]
        return OUTPUT_PATH.format(
            "linear-probe",
            level,
            split,
            DATASET_NAME,
            model,
            norms_loader.get_suffix(),
            feature_id,
        )

    def load_pickle(path):
        with open(path + ".pkl", "rb") as f:
            return pickle.load(f)

    def get_clfs(data):
        return [output["clf"] for output in data]

    data = {
        feature: get_clfs(load_pickle(get_path(feature)))
        for feature in tqdm(features_selected)
    }

    with open(f"output/classifiers-for-stella-{model}.pkl", "wb") as f:
        pickle.dump(data, f)


def get_results_multiple_classifiers():
    embeddings_level = "concept"
    split_type = "repeated-k-fold"
    feature_type = "gemma-2b-contextual-last-word"
    feature_norm_str = "mcrae-mapped"
    classifiers = [
        "linear-probe",
        "linear-probe-std",
        "knn-3",
    ]

    results = [
        {
            "classifier": c,
            **r,
        }
        for c in classifiers
        for r in load_result_features(
            c, embeddings_level, split_type, feature_type, feature_norm_str
        )
    ]
    df = pd.DataFrame(results)
    cols = ["classifier", "score-f1", "score-precision", "score-recall"]
    df = df[cols]
    df = df.groupby("classifier").mean()
    print(df)


def get_norm_completeness():
    NAMES = {
        "mcrae": "McRae",
        "generated-gpt35": "Hansen",
        "mcrae-mapped": "McRae++",
    }

    def do1(norm_type, use_selected):
        norm_loader = NORMS_LOADERS[norm_type]()
        feature_to_concepts, _, features_selected = norm_loader()
        if use_selected:
            features = features_selected
        else:
            features = feature_to_concepts.keys()
        num_norms = len(features)
        num_norms_with_n_concepts = sum(
            1 for f in features if len(feature_to_concepts[f]) >= 10
        )
        use_selected_str = "✓" if use_selected else "✗"
        print(
            "{:10s} {:s} → {:5d} ({:.1f}%)".format(
                NAMES[norm_type],
                use_selected_str,
                num_norms_with_n_concepts,
                100 * num_norms_with_n_concepts / num_norms,
            )
        )

    do1("mcrae", False)
    do1("generated-gpt35", False)
    do1("mcrae-mapped", False)
    do1("mcrae-mapped", True)


def aggregate_predictions_for_demo(norms_type):
    CLASSIFIER_TYPE = "linear-probe"
    EMBEDDINGS_LEVEL = "concept"
    SPLIT_TYPE = "repeated-k-fold"
    DATASET_NAME = "things"

    norms_loader = NORMS_LOADERS[norms_type]()
    _, feature_to_id, features_selected = norms_loader()
    concepts = norms_loader.load_concepts()
    num_concepts = len(concepts)

    def get_path_results(model, feature):
        feature_id = feature_to_id[feature]
        path = OUTPUT_PATH.format(
            CLASSIFIER_TYPE,
            EMBEDDINGS_LEVEL,
            SPLIT_TYPE,
            DATASET_NAME,
            model,
            norms_loader.get_suffix(),
            feature_id,
        )
        return path + ".json"

    def get_five_folds(results):
        data = [datum for fold in range(5) for datum in results[fold]["preds"]]
        data = sorted(data, key=lambda datum: datum["i"])
        indices = [datum["i"] for datum in data]
        assert indices == list(range(num_concepts))
        preds = [datum["pred"] for datum in data]
        return preds

    results = [
        [
            get_five_folds(read_json(get_path_results(model, feature)))
            for model in MAIN_TABLE_MODELS
        ]
        for feature in features_selected
    ]

    path = f"static/demo/predictions-{norms_type}.npz"
    results_np = np.array(results)
    results_np = results_np.astype(np.float32)
    np.savez(path, results=results_np)


def get_results_binder_norms_regression():
    CLASSIFIER_TYPE = "linear-regression"
    EMBEDDINGS_LEVEL = "concept"
    SPLIT_TYPE = "repeated-k-fold-simple"
    NORMS_TYPE = "binder-dense"
    # DATASET_NAME = "things"

    results = [
        r
        for m in MAIN_TABLE_MODELS
        for r in load_result_features(
            CLASSIFIER_TYPE, EMBEDDINGS_LEVEL, SPLIT_TYPE, m, NORMS_TYPE
        )
    ]

    df = pd.DataFrame(results)
    cols = df.columns.tolist()
    cols = ["model"] + [c for c in cols if c.startswith("score-")]
    df = df[cols].groupby(["model"]).mean()
    df = df.reindex(MAIN_TABLE_MODELS)
    df = df.reset_index()
    df["model"] = df["model"].map(FEATURE_NAMES)
    print(df.to_csv("output/binder-regression-results.csv", index=False))


def show_ranking_plot():
    def get_results_1(models, classifier_type, split_type, norm_type, metrics):
        embeddings_level = "concept"

        if "score-f1-selectivity" in metrics:
            scores_random_features = get_score_random_features(norm_type)

            def add_f1_sel(results):
                for r in results:
                    r["score-f1-selectivity"] = (
                        r["score-f1"] - scores_random_features[r["feature"]]
                    )
                return results

        else:

            def add_f1_sel(results):
                return results

        results = [
            result
            for model in models
            for result in add_f1_sel(
                load_result_features(
                    classifier_type,
                    embeddings_level,
                    split_type,
                    model,
                    norm_type,
                )
            )
        ]
        cols = ["model", "norms-type"] + metrics
        df = pd.DataFrame(results)
        df = df[cols]
        df = df.pivot_table(index="model", columns="norms-type")
        return df

    SETTINGS = [
        {
            "classifier_type": "linear-probe",
            "split_type": "repeated-k-fold",
            "norm_type": "mcrae-x-things",
            "metrics": [
                "score-f1",
                "score-precision",
                "score-recall",
                "score-f1-selectivity",
            ],
        },
        {
            "classifier_type": "linear-probe",
            "split_type": "repeated-k-fold",
            "norm_type": "binder-median",
            "metrics": [
                "score-f1",
                "score-precision",
                "score-recall",
                "score-f1-selectivity",
            ],
        },
        {
            "classifier_type": "linear-regression",
            "split_type": "repeated-k-fold-simple",
            "norm_type": "binder-dense",
            "metrics": [
                "score-rmse",
                "score-mse",
                "score-mae",
                "score-r2",
                "score-accuracy-delta-1",
                "score-accuracy-delta-0.5",
            ],
        },
    ]

    SCORE_NAMES = {
        "score-accuracy": "Acc.",
        "score-precision": "P",
        "score-recall": "R",
        "score-f1": "F1",
        "score-f1-selectivity": "F1 sel.",
        "score-roc-auc": "ROC AUC",
        "score-rmse": "RMSE",
        "score-mse": "MSE",
        "score-mae": "MAE",
        "score-r2": "R²",
        "score-accuracy-delta-1": "Acc. (Δ=1)",
        "score-accuracy-delta-0.5": "Acc. (Δ=0.5)",
    }

    def load_data():
        dfs = [get_results_1(MAIN_TABLE_MODELS, **setting) for setting in SETTINGS]
        return dfs[0].join(dfs[1:])

    df = load_data()
    for e in ("mse", "rmse", "mae"):
        df[(f"score-{e}", "binder-dense")] = -df[(f"score-{e}", "binder-dense")]
    df = df.rank(ascending=False, method="min")

    # Keep models order to have them displayed on the left.
    def get_models_order(df, i):
        last_col = df.columns[i]
        idxs = df[last_col].values.argsort()
        models = df.index.tolist()
        return [models[i] for i in idxs]

    models_left = get_models_order(df, 0)
    models_right = get_models_order(df, -1)

    ORDER = [
        ("score-f1-selectivity", "mcrae-x-things"),
        ("score-rmse", "binder-dense"),
        # ("score-mae", "binder-dense"),
        # ("score-accuracy-delta-0.5", "binder-dense"),
        # ("score-f1-selectivity", "binder-median"),
    ]

    def sort_func(ss):
        return ss.map(lambda x: ORDER.index(x))

    df = df.unstack()
    df = df.reset_index()
    df = df.rename(columns={"level_0": "metric", 0: "rank"})
    df["setting"] = df[["metric", "norms-type"]].apply(
        lambda x: (x["metric"], x["norms-type"]), axis=1
    )
    NORMS_NAMES = {
        "mcrae-x-things": "McRae×THINGS",
        "binder-median": "Binder",
        "binder-dense": "Binder",
    }
    df["setting-str"] = df["setting"].map(
        lambda x: "{}\n{}".format(NORMS_NAMES[x[1]], SCORE_NAMES[x[0]])
    )
    df = df[df["setting"].isin(ORDER)]
    df = df.sort_values(by="setting", key=sort_func)

    modalities = ["image", "text"]
    modality_to_color = {
        "image": "flare_r",
        "text": "crest_r",
    }
    num_models = Counter([FEATURE_TYPE_TO_MODALITY[m] for m in MAIN_TABLE_MODELS])
    modality_to_palette = {
        modality: sns.color_palette(modality_to_color[modality], num_models[modality])
        for modality in modalities
    }
    modality_to_models = {
        modality: [
            model
            for model in MAIN_TABLE_MODELS
            if modality == FEATURE_TYPE_TO_MODALITY[model]
        ]
        for modality in modalities
    }
    palette = {
        model: color
        for modality in modalities
        for model, color in zip(
            modality_to_models[modality], modality_to_palette[modality]
        )
    }

    sns.set(style="whitegrid", font="Arial")
    # camera ready
    fig, ax = plt.subplots(figsize=(4, 5))
    # poster
    # fig, ax = plt.subplots(figsize=(2, 7))
    sns.lineplot(
        data=df,
        x="setting-str",
        y="rank",
        hue="model",
        legend=False,
        marker="o",
        sort=False,
        palette=palette,
        ax=ax,
    )
    ax.set_xlabel("")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    ax.set_ylabel("")
    ax.set_yticks(range(1, len(models_left) + 1))
    ax.set_yticklabels([FEATURE_NAMES.get(m, m) for m in models_left])

    ax2 = ax.secondary_yaxis("right")
    ax2.tick_params(length=0)
    ax2.set_yticks(range(1, len(models_right) + 1))
    ax2.set_yticklabels([FEATURE_NAMES.get(m, m) for m in models_right])
    # ax2.set_ytickslabels([str(r) for r in range(1, len(models) + 1)])
    # ax2.set_ylabel("Rank")

    ax.invert_yaxis()
    ax.grid(axis="y")
    st.pyplot(fig)

    fig.savefig("output/plots/ranking-plot.pdf", bbox_inches="tight", transparent=True)


def get_results_instance_level(*models):
    # METRIC = "score-f1"
    METRIC = "score-f1-selectivity"
    SETTINGS = [
        {
            "classifier_type": "linear-probe",
            "embeddings_level": "instance",
            "split_type": "repeated-k-fold-instance",
            "norms_type": "mcrae-x-things",
            "metric": METRIC,
        },
        {
            "classifier_type": "linear-probe",
            "embeddings_level": "concept",
            "split_type": "repeated-k-fold",
            "norms_type": "mcrae-x-things",
            "metric": METRIC,
        },
    ]

    def get_results_1(models, metric, **settings):
        results = [
            result
            for model in models
            for result in load_result_features(feature_type=model, **settings)
        ]
        df = pd.DataFrame(results)

        if metric in COMPUTE_METRICS:
            df = add_metric(df, **settings)

        return df

    if len(models) == 0:
        models = MAIN_TABLE_MODELS

    cols = ["model", "feature", "level", METRIC]
    dfs = {s["embeddings_level"]: get_results_1(models, **s) for s in SETTINGS}
    df = pd.concat(dfs, axis=0)
    df = df[cols]

    # For each model, find top 5 and bottom 5 features with largest difference
    # between concept and instance F₁ selectivity.
    df_pivot = df.pivot_table(
        index=["model", "feature"], columns="level", values=METRIC
    ).reset_index()

    df_pivot["diff"] = df_pivot["concept"] - df_pivot["instance"]

    def get_top5(model):
        df_model = df_pivot[df_pivot["model"] == model]
        df_model = df_model.sort_values("diff", ascending=False)
        top5 = df_model.head(5)
        # bot5 = df_model.tail(5)
        # selected10 = pd.concat([top5, bot5])
        return ", ".join(
            "{} ({:.2f})".format(row["feature"], row["diff"])
            for _, row in top5.iterrows()
        )

    model_to_top5 = {model: get_top5(model) for model in models}

    df = df.groupby(["model", "level"])[METRIC].mean()
    df = df.reset_index()
    df = df.pivot(index="model", columns="level", values=METRIC)
    df = df.reindex(models)
    df = df.reset_index()
    df["top5"] = df["model"].map(model_to_top5)
    df["model"] = df["model"].map(lambda x: FEATURE_NAMES.get(x, x))
    print(df.to_csv(sep="|"))


FUNCS = {
    "levels-and-splits": get_results_levels_and_splits,
    "per-metacategory": partial(
        get_results_per_metacategory,
        "concept",
        "repeated-k-fold",
    ),
    "per-metacategory-mcrae-mapped": get_results_per_metacategory_mcrae_mapped,
    "per-metacategory-binder": get_results_per_metacategory_binder,
    "binder-norms": get_results_binder_norms,
    "classifier-agreement-binder-norms": get_classifiers_agreement_binder_norms,
    "analyse-classifiers": analyse_classifiers,
    "analyse-classifiers-quantitative": analyse_classifiers_quantiative,
    "paper-table-main-row": get_results_paper_table_main_row,
    "paper-table-main": get_results_paper_table_main,
    "paper-table-main-acl-camera-ready": get_results_paper_table_main_acl_camera_ready,
    "paper-table-main-full-acl-camera-ready": get_results_paper_table_full_main_acl_camera_ready,
    "per-feature-norm": get_results_per_feature_norm,
    "random-predictor": get_results_random_predictor,
    "results-all-one-norm": get_results_all_one_norm,
    "results-contextualised-language-models": get_results_contextualised_language_models,
    "compare-two-models-scatterplot": compare_two_models_scatterplot,
    "compare-two-models-scatterplot-2": compare_two_models_scatterplot_2,
    "correlation-between-models": get_correlation_between_models,
    "correlation-between-models-2": get_correlation_between_models_2,
    "results-multiple-classifiers": get_results_multiple_classifiers,
    "results-for-stella": prepare_results_for_stella,
    "classifiers-for-stella": prepare_classifiers_for_stella,
    "get-norm-completeness": get_norm_completeness,
    "aggregate-predictions-for-demo": aggregate_predictions_for_demo,
    "binder-regression": get_results_binder_norms_regression,
    "binder-norms-regression": lambda metric: get_results_binder_norms(
        classifier_type="linear-regression",
        level="concept",
        split="repeated-k-fold-simple",
        norm_type="binder-dense",
        metric=metric,
    ),
    "ranking-plot": show_ranking_plot,
    "results-instance": get_results_instance_level,
}


if __name__ == "__main__":
    what = sys.argv[1]
    what, *args = what.split(":")
    FUNCS[what](*args)
