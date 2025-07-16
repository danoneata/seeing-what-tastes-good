import pdb

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import f1_score
from toolz import first

from probing_norms.data import DIR_THINGS
from probing_norms.utils import cache_df, cache_json, read_json, multimap
from probing_norms.predict import NORMS_LOADERS, DATASETS
from probing_norms.get_results import OUTPUT_PATH, FEATURE_NAMES, normalize_feature_name, load_taxonomy_mcrae, METACATEGORY_NAMES

st.set_page_config(layout="wide")


def main():
    CLASSIFIER_TYPE = "linear-probe"
    EMBEDDINGS_LEVEL = "concept"
    SPLIT_TYPE = "repeated-k-fold"
    DATASET_NAME = "things"

    dataset = DATASETS[DATASET_NAME]()

    MODELS = [
        # "dino-v2",
        # "fasttext-word",
        "swin-v2-ssl",
        "gemma-2b-contextual-layers-9-to-18-seq-last-word",
        "clip",
        "clip-word",
    ]

    norms_type = "mcrae-x-things"
    norms_loader = NORMS_LOADERS[norms_type]()
    feature_to_concepts, feature_to_id, features_selected = norms_loader()

    concepts = norms_loader.load_concepts()
    num_concepts_total = len(concepts)
    labels = np.arange(num_concepts_total)
    taxonomy = load_taxonomy_mcrae()

    with st.sidebar:
        feature = st.selectbox("Feature norm:", features_selected)
        sort_by = st.selectbox("Sort by:", ["concept name", "concept label"] + ["score {}".format(m) for m in MODELS], index=2)

    def evaluate(results):
        def eval1(data):
            true = np.array([datum["true"] for datum in data])
            pred = np.array([datum["pred"] for datum in data])
            pred = (pred > 0.5).astype(int)
            return f1_score(true, pred)

        num_pos = len(set(feature_to_concepts[feature]))
        scores = [eval1(result["preds"]) for result in results]
        scores = np.array(scores)
        score = 100 * np.mean(scores)
        score_random = 100 * num_pos / num_concepts_total
        return score - score_random

    def get_path(model):
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
        return [
            datum
            for fold in range(5)
            for datum in results[fold]["preds"]
        ]

    def prepare_results(results):
        true = np.zeros(num_concepts_total)
        pred = np.zeros(num_concepts_total)
        for datum in get_five_folds(results):
            i = datum["i"]
            true[i] = datum["true"]
            pred[i] = datum["pred"] >= 0.5
        return true, pred

    def get_tp_fp_fn(true, pred):
        def mapc(idxs):
            return [dataset.label_to_class[label] for label in idxs]

        tp, *_ = np.where([t and p for t, p in zip(true, pred)])
        fp, *_ = np.where([not t and p for t, p in zip(true, pred)])
        fn, *_ = np.where([t and not p for t, p in zip(true, pred)])
        return {
            "tp": mapc(tp),
            "fp": mapc(fp),
            "fn": mapc(fn),
        }

    def get_image_path(concept):
        return DIR_THINGS / "object_images_CC0" / (concept + ".jpg")

    results = {model: read_json(get_path(model)) for model in MODELS}
    scores = {m: evaluate(r) for m, r in results.items()}
    results_2 = {m: prepare_results(r) for m, r in results.items()}
    results_tp_fp_fn = {
        m: get_tp_fp_fn(*prepare_results(r)) for m, r in results.items()
    }

    num_models = len(MODELS)
    # num_categories = 3
    # table = [st.columns(num_models) for _ in range(1 + num_categories)]
    # table = list(zip(*table))

    # for i, model in enumerate(MODELS):
    #     table[i][0].markdown("## {}".format(model))
    #     for j, category in enumerate(["tp", "fp", "fn"]):
    #         table[i][j + 1].markdown("### {}".format(category))
    #         table[i][j + 1].markdown(", ".join(results_tp_fp_fn[model][category]))

    def sort_func(concept):
        if sort_by == "concept name":
            return concept
        elif sort_by == "concept label":
            return concept not in feature_to_concepts[feature]
        else:
            _, model = sort_by.split()
            return -results_2[model][1][concepts.index(concept)]

    concepts_order = sorted(concepts, key=sort_func)

    for i, concept in enumerate(concepts_order):
        cols = st.columns(1 + num_models)
        label_str = "+" if concept in feature_to_concepts[feature] else "-"
        c = concepts.index(concept)
        cols[0].markdown("#{} · {} · {}".format(i, concept, label_str))
        cols[0].image(str(get_image_path(concept)))
        for i in range(num_models):
            model = MODELS[i]
            model_name = FEATURE_NAMES[model]
            true, pred = results_2[model]
            t = true[c]
            p = pred[c]
            b = p >= 0.5
            # pred_str = "+" if p else "-"
            is_correct_str = "✓" if t == b else "✗"
            cols[i + 1].markdown("{} · pred: {:.1f} · is correct: {}".format(model_name, p, is_correct_str))

    return 

    predicted_concepts = [
        concept
        for model in MODELS
        for category in ["tp", "fp", "fn"]
        for concept in results_tp_fp_fn[model][category]
    ]
    predicted_concepts = sorted(set(predicted_concepts))

    def get_pred_type(result, concept):
        if concept in result["tp"]:
            return "TP"
        elif concept in result["fp"]:
            return "FP"
        elif concept in result["fn"]:
            return "FN"
        else:
            return "TN"

    st.markdown("---")

    cols = st.columns(1 + num_models)
    cols[0].markdown("## Concept")
    for i in range(num_models):
        model = MODELS[i]
        model_name = FEATURE_NAMES[model]
        cols[i + 1].markdown("## {} ({:.1f})".format(model_name, scores[model]))

    for c in predicted_concepts:
        cols = st.columns(1 + num_models)
        cols[0].image(str(get_image_path(c)))
        cols[0].markdown("{}".format(c))
        for i in range(num_models):
            cols[i + 1].markdown(get_pred_type(results_tp_fp_fn[MODELS[i]], c))

    return

    r = [
        {
            "concept": c,
            "model": m,
            "type": get_pred_type(results_tp_fp_fn[m], c),
        }
        for m in MODELS
        for c in predicted_concepts
    ]
    df = pd.DataFrame(r)
    df = df.pivot_table(
        index="concept",
        columns="model",
        values="type",
        aggfunc=lambda x: x,
    )
    st.write(df)

    def transpose(table):
        return list(zip(*table))

    def get_image_block(concept):
        path = "images/qualitative-results/{}.jpg".format(concept)
        return r"\multirow{4}{*}{\includegraphics[width=\ww]{" + path + r"}}"

    def get_pred_and_true(results, concept):
        label = dataset.class_to_label[concept]
        pred = first(r for r in get_five_folds(results) if r["i"] == label)
        p = pred["pred"] > 0.5
        t = pred["true"]
        return p, t

    def pred_to_str(pred_true):
        pred, true = pred_true
        if pred:
            s = r"\yes"
        else:
            s = r"\no"
        if true == pred:
            s = macro("correct", s)
        else:
            s = macro("wrong", s)
        return s

    def generate_table(concept):
        block_image = [get_image_block(concept)]
        block_preds = [get_pred_and_true(results[m], concept) for m in MODELS]
        # block_preds = ["\\" + p for p in block_preds]
        block_preds = [pred_to_str(p) for p in block_preds]
        block_image = block_image + (num_models - 1) * [""]
        table = list(zip(block_image, block_preds))
        return table

    def vcat(tables):
        return [row for table in tables for row in table]

    def hcat(tables):
        table = vcat(map(transpose, tables))
        return transpose(table)

    def macro(name, arg):
        return "\\" + name + r"{" + arg + r"}"

    def table_to_latex(table):
        return "\n".join(" & ".join(row) + r" \\" for row in table)

    import random

    random.seed(1337)

    SCORES = {
        "TP": 3,
        "FP": 2,
        "FN": 1,
        "TN": 1,
    }

    CONCEPTS_SELECTED = {
        # "has_4_legs": ["anteater", "mole", "goat", "stool", "dog"],
        "has_4_legs": ["tablecloth", "altar", "kangaroo", "stool", "dog"],
        # "is_dangerous": ["razor", "dynamite", "axe", "tumbleweed", "mole"],
        "is_dangerous": ["dynamite", "razor", "bison", "corkscrew", "cheesecake"],
        # "made_of_wood": ["bow3", "dynamite", "axe", "loveseat", "ski"],
        "made_of_wood": ["bow3", "puppet", "axe", "cardboard", "ski"],
        # "tastes_sweet": ["plum", "raisin", "watermelon", "pineapple", "lavender"],
        "tastes_sweet": ["plum", "raisin", "cake_mix", "tomato_sauce", "crystal1"],
    }

    def score_concept(concept):
        preds_trues = [get_pred_and_true(results[m], concept) for m in MODELS]
        preds, trues = zip(*preds_trues)
        assert len(set(trues)) == 1
        corrects = [p == t for p, t in zip(preds, trues)]
        return first(trues), sum(corrects)

    def is_positive_str(concept):
        if concept in feature_to_concepts[feature]:
            return r"\cyes"
        else:
            return r"\cno"

    # predicted_concepts_ss = list(predicted_concepts)[:5]
    try:
        predicted_concepts_ss = CONCEPTS_SELECTED[feature]
    except KeyError:
        predicted_concepts_ss = random.sample(predicted_concepts, 5)
    predicted_concepts_ss = sorted(
        predicted_concepts_ss,
        key=score_concept,
        reverse=True,
    )
    for c in predicted_concepts_ss:
        st.write(get_image_path(c))

    st.write(" ".join(predicted_concepts_ss))

    blocks = [generate_table(c) for c in predicted_concepts_ss]
    header_left = [[FEATURE_NAMES[m], "{:.1f}".format(scores[m])] for m in MODELS]
    header_top = [
        ["", ""]
        + [e for elem in predicted_concepts_ss for e in [macro("concept", elem), is_positive_str(elem)]]
    ]
    table = hcat([header_left] + blocks)
    table = vcat([header_top] + [table])
    metacategory = METACATEGORY_NAMES.get(taxonomy[feature], taxonomy[feature])
    metacategory = metacategory.replace("&", "\&")
    first_row = (
        r"& & \multicolumn{10}{c}{"
        + macro("attribute", normalize_feature_name(feature))
        + " "
        + "("
        + metacategory
        + ")"
        + r"} \\"
        + "\n"
    )
    table_latex = first_row + table_to_latex(table)
    st.code(table_latex)


if __name__ == "__main__":
    main()
