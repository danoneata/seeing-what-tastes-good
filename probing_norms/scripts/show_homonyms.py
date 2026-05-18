import csv

from itertools import groupby

import streamlit as st

from probing_norms.data import (
    DATASETS,
    DIR_THINGS,
    DIR_GPT3_NORMS,
    load_things_concept_mapping,
)
from probing_norms.predict import NORMS_LOADERS
from probing_norms.utils import multimap, read_file
from probing_norms.extract_features_text import HFModelContextual

dataset = DATASETS["things"]()
concept_image = [
    (dataset.label_to_class[dataset.labels[i]], dataset.image_files[i])
    for i in range(len(dataset))
]
concept_to_images = multimap(concept_image)
norms_loader = NORMS_LOADERS["mcrae-mapped"]()
feature_to_concepts, _, features_selected = norms_loader()
concepts = norms_loader.load_concepts()
contexts = HFModelContextual.load_context("gpt4o_concept_context_sentences_v2")
concept_feature = [
    (concept, feature)
    for feature, concepts in feature_to_concepts.items()
    for concept in concepts
    if feature in features_selected
]
concept_to_features = multimap(set(concept_feature))


def get_concept_to_category_asked():
    def get_category(question):
        question = question.replace("?", "")
        words = question.split()
        return " ".join(words[6:])

    def parse_line(elems):
        _, _, question, concept = elems
        category = get_category(question)
        return concept, category

    def read_csv(path):
        with open(path) as csvfile:
            for i, row in enumerate(csv.reader(csvfile)):
                if i == 0:
                    pass
                else:
                    yield row

    path = (
        DIR_GPT3_NORMS
        / "data/gpt_3_feature_norm/generation_questions/all_concepts.csv"
    )
    return dict(map(parse_line, read_csv(path)))


concept_to_word_and_category_1 = load_things_concept_mapping()
concept_to_word_and_category_2 = get_concept_to_category_asked()


def get_image_path(concept):
    return DIR_THINGS / "object_images_CC0" / (concept + ".jpg")


def prepare(concept):
    *chars, n = concept
    word = "".join(chars)
    return word, int(n), concept


def itemize(xs):
    return "\n".join("1. {}".format(x) for x in xs)


def code(x):
    return f"```\n{x}\n```"


homonyms = [prepare(concept) for concept in concepts if concept[-1].isdigit()]
groups = groupby(homonyms, key=lambda x: x[0])

st.set_page_config(layout="wide")

for num, (k, group) in enumerate(groups, start=1):
    group = list(group)
    st.markdown("### {} · {}".format(num, k))
    cols = st.columns(4)
    for i, (word, n, concept) in enumerate(group):
        word_and_category_1 = concept_to_word_and_category_1[concept]
        word_and_category_2 = concept_to_word_and_category_2[concept]
        wcs = [word_and_category_1, word_and_category_2]
        cols[i].markdown("`{}` →".format(concept))
        cols[i].markdown(itemize(wcs))
        cols_ = cols[i].columns(3)
        for j, col in enumerate(cols_):
            path = dataset.get_image_path(concept_to_images[concept][j])
            col.image(path)
        cols[i].markdown("#### Contextual sentences")
        cols[i].markdown(itemize(contexts[concept][:3]))
        cols[i].markdown("#### Attributes")
        cols[i].markdown(code("\n".join(sorted(concept_to_features[concept]))))
    st.markdown("---")
