import json
from pathlib import Path
from typing import List, Union


# Load Stopwords
def load_stopwords(
    dir_stopwords: Path,
    use_stopwords: Union[str, List[str]] = [
        "administración",
        "municipios",
        "common_stopwords",
    ],
):
    stop_words = {}
    if not use_stopwords:
        return stop_words

    if isinstance(use_stopwords, str):
        if use_stopwords == "all":
            use_stopwords = [el.stem for el in dir_stopwords.iterdir() if el.is_file()]
        else:
            use_stopwords = [use_stopwords]

    for el in use_stopwords:
        with dir_stopwords.joinpath(f"{el}.txt").open("r", encoding="utf-8") as f:
            stop_words = {*stop_words, *set([w.strip() for w in f.readlines()])}

    # stop_words = list(
    #     set(
    #         list(stop_words)
    #         + [w.lower() for w in stop_words]
    #         + [w.upper() for w in stop_words]
    #         + [" ".join([el.capitalize() for el in w.split()]) for w in stop_words]
    #     )
    # )
    stop_words.update(set([w.lower() for w in stop_words]))
    stop_words.update(set([w.replace(" ", "-") for w in stop_words]))
    stop_words = list(stop_words)

    stop_words = sorted(stop_words, key=len, reverse=True)
    return stop_words


def load_vocabulary(dir_vocabulary: Path):
    with dir_vocabulary.open("r", encoding="utf8") as f:
        vocabulary = json.load(f)
    return vocabulary
