from functools import partial

import math

import altair as alt
import streamlit as st
import pandas as pd
import numpy as np

from scipy.stats import describe
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from probing_norms.predict import (
    DATASETS,
    NORMS_LOADERS,
    load_embeddings,
    get_binary_labels,
)

st.set_page_config(layout="wide")


def main():
    dataset_name = "things"
    dataset = DATASETS[dataset_name]()

    # SELECTED = ["dino-v2", "pali-gemma-224", "siglip-224", "fasttext-word", "random-siglip", "gemma-2b-word"]
    SELECTED = [
        # "siglip-224",
        # "fasttext-word",
        # "clip-word",
        "dino-v2",
        "dino-v3-vitl16",
        # "gemma-2b-contextual-last-word",
        # "gemma-2b-contextual-50-last-word",
        # "gemma-2b-contextual-50-constrained-last-word",
        # "gpt2-contextual-last-word",
    ]
    data = {f: load_embeddings(dataset_name, f, "concept") for f in SELECTED}

    norm_type = "mcrae-x-things"
    norm_loader = NORMS_LOADERS[norm_type]()
    feature_to_concepts, feature_to_id, features_selected = norm_loader()

    with st.sidebar:
        feature = st.selectbox("Select feature type", features_selected)

    get_binary_labels_1 = partial(
        get_binary_labels,
        feature=feature,
        feature_to_concepts=feature_to_concepts,
        class_to_label=dataset.class_to_label,
    )

    def show(embs, labels, col):
        pca = PCA(n_components=2)
        embs_2d = pca.fit_transform(embs)

        df = pd.DataFrame(embs_2d, columns=["x", "y"])
        df["has-feature"] = get_binary_labels_1(labels)
        df["concept"] = [dataset.label_to_class[label] for label in labels]

        scatter = (
            alt.Chart(df)
            .mark_circle()
            .encode(
                x="x",
                y="y",
                color=alt.Color("has-feature"),
                tooltip="concept",
            )
            .properties(height=500)
            .interactive()
        )
        # col.write(describe(embs.flatten()))
        col.altair_chart(scatter, use_container_width=True)

    num_cols = 2
    num_rows = int(math.ceil(len(SELECTED) / num_cols))
    table = [st.columns(num_cols) for _ in range(num_rows)]
    for i, f in enumerate(SELECTED):
        r = i // num_cols
        c = i % num_cols
        table[r][c].markdown(f)
        show(*data[f], table[r][c])

    def l2_norm(embs):
        return embs / np.linalg.norm(embs, axis=1, keepdims=True)

    def get_closest_concepts(embs, labels):
        binary_labels = get_binary_labels_1(labels)
        idxs, *_ = np.where(binary_labels)
        # embs = l2_norm(embs)
        dists = squareform(pdist(embs))
        np.fill_diagonal(dists, np.inf)
        return [
            {
                "concept": dataset.label_to_class[labels[i]],
                "closest": dataset.label_to_class[labels[dists[i].argmin()]],
                # "distance": dists[i, j],
            }
            for i in idxs
        ]

    closest_concepts = [
        {"embedding": e, **r} for e in SELECTED for r in get_closest_concepts(*data[e])
    ]
    df = pd.DataFrame(closest_concepts)
    df = df.pivot_table(
        index="concept",
        columns="embedding",
        values="closest",
        aggfunc="first",
    )
    df = df[SELECTED]
    st.markdown("---")
    st.markdown("#### Closest concept for each positive concept by each of the embedding models")
    st.write(df)


if __name__ == "__main__":
    main()
