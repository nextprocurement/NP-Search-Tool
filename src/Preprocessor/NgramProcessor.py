from collections import Counter
from typing import List, Union

import numpy as np
import pandas as pd
from Levenshtein import distance
from nltk.util import ngrams
from tqdm import tqdm


def find_ngrams(text: List[str], n: int = 2, **kwargs):
    """
    Get a list of all ngrams of `n` words that appear in a text.
    """
    n_grams = list(ngrams(text, n, **kwargs))
    return n_grams


def _compute_ngram_counts(
    corpus: List[List[str]],
    n: int = 2,
):
    """
    Parameters
    ----------
    corpus: list(list(str))
    min_appearance: int
        minimum number of appearances for a word in the entire vocabulary to be considered
    n: int
        ngram size

    Returns
    -------
    wc: Counter
        WordCounter: number of total appearances of word.
    cc: Counter
        CoappearanceCounter: number of total appearances of ngram.
    """
    word_counts = Counter()
    ngrams = Counter()

    [
        (word_counts.update(Counter(d)), ngrams.update(Counter(find_ngrams(d, n))))
        for d in corpus
    ]

    return word_counts, ngrams


def _compute_co_occurrence_counts(corpus: List[List[str]]):
    """
    Computes the total number of co-appearances in a text of two terms.

    Parameters
    ----------
    corpus: list(list(str))
    min_appearance: int
        minimum number of appearances for a word in the entire vocabulary to be considered
    n: int
        ngram size
    """
    word_counts = Counter()
    co_occurrence_counts = dict()

    for d in corpus:
        word_counts.update(Counter(d))
        for word in d:
            w_occ = co_occurrence_counts.get(word, Counter())
            w_occ.update(Counter([w for w in d if not w == word]))
            co_occurrence_counts[word] = w_occ
    co_occurrence_counts = dict(
        [
            (tuple(sorted([w1, w2])), occ)
            for w1, v in co_occurrence_counts.items()
            for w2, occ in v.items()
        ]
    )
    return word_counts, co_occurrence_counts


def _filter_co_appearance(
    word_counts: dict,
    ngrams: dict,
    min_appearance_word: int = 0,
    min_appearance_ngram: int = 0,
):
    """
    Obtain a filtered version of ngrams by number of appearances.

    Parameters
    ----------
    word_counts: dict
        word appearance dictionary. {"term1": num_appearances}
    ngrams: dict
        co-appearance dictionary. {("term1", "term2"): num_appearances}
    min_appearance_word: int
        minimum number of appearances for a term in the entire vocabulary to be considered
    min_appearance_ngram: int
        minimum number of appearances for a term tuple to be considered

    Returns
    -------
    word_counts, ngrams, total_words, total_ngrams
    """

    valid_vocab = set(word_counts.keys())
    valid_ngrams = set(ngrams.keys())

    if min_appearance_word or min_appearance_ngram:
        if min_appearance_word:
            valid_vocab = set(
                [w for w in valid_vocab if word_counts[w] >= min_appearance_word]
            )
        if min_appearance_ngram:
            valid_ngrams = set(
                [ng for ng in valid_ngrams if ngrams[ng] >= min_appearance_ngram]
            )

        # Get common vocabulary that satisfies both options
        valid_ngram_vocab = []
        for el in valid_ngrams:
            valid_ngram_vocab.extend(el)
        valid_ngram_vocab = set(valid_ngram_vocab)

        # Filter final elements
        valid_vocab = valid_vocab & valid_ngram_vocab
        valid_ngrams = set(
            [ng for ng in valid_ngrams if all([el in valid_vocab for el in ng])]
        )
        valid_ngram_vocab = []
        for el in valid_ngrams:
            valid_ngram_vocab.extend(el)
        valid_ngram_vocab = set(valid_ngram_vocab)
        valid_vocab = valid_vocab & valid_ngram_vocab

    # Output
    word_counts = {v: word_counts[v] for v in valid_vocab}
    ngrams = {v: ngrams[v] for v in valid_ngrams}
    total_words = sum(word_counts.values())
    total_ngrams = sum(ngrams.values())

    return word_counts, ngrams, total_words, total_ngrams


def get_ngrams_in_corpus(
    corpus: List[List[str]],
    min_appearance_word: int = 0,
    min_appearance_ngram: int = 0,
    n: Union[int, List[int]] = 2,
    verbose=False,
):
    """
    Get ngrams from a corpus filtered by appearance.

    Parameters
    ----------
    corpus: list(list(str))
    min_appearance: int
        minimum number of appearances for a word in the entire vocabulary to be considered
    n: int | list(int)
        ngram size
        If list: find ngrams for all sizes

    Returns
    -------
    word_counts, co_occurrence_counts, total_words, total_ngrams
    """
    word_counts = {}
    co_occurrence_counts = {}
    total_words = 0
    total_ngrams = 0
    if isinstance(n, int):
        word_counts, co_occurrence_counts = _compute_ngram_counts(corpus, n)
    else:
        if verbose:
            # t = trange(len(n))
            pbar = tqdm(total=len(n), leave=True)
        # else:
        #     t = range(len(n))
        for el in n:
            # el = n[i]
            if verbose:
                pbar.set_description(f"{el}-gram")
                pbar.refresh()
            wc, cc = _compute_ngram_counts(corpus, el)
            wc, cc, tw, tn = _filter_co_appearance(
                wc, cc, min_appearance_word, min_appearance_ngram
            )
            word_counts = {**word_counts, **wc}
            co_occurrence_counts = {**co_occurrence_counts, **cc}
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()
    (
        word_counts,
        co_occurrence_counts,
        total_words,
        total_ngrams,
    ) = _filter_co_appearance(
        word_counts, co_occurrence_counts, min_appearance_word, min_appearance_ngram
    )

    return word_counts, co_occurrence_counts, total_words, total_ngrams


def PMI(
    word_counts: dict, co_occurrence_counts: dict, total_words: int, total_ngrams: int
):
    """
    Computes the Pointwise Mutual Information.
    """
    pmi = {}
    for words, co_occurrence_count in co_occurrence_counts.items():
        p_words = [word_counts[w] / total_words for w in words]
        p_words_prod = np.prod(p_words)
        p_word1_word2 = co_occurrence_count / total_ngrams

        pmi_value = np.log2(p_word1_word2 / p_words_prod)
        pmi[words] = pmi_value

    return pmi


def compute_pmi_ngram_corpus(
    corpus: List[List[str]],
    min_appearance_word: int = 0,
    min_appearance_ngram: int = 0,
    n: Union[int, List[int]] = 2,
    verbose=False,
):
    """
    Compute Pointwise Mutual Information of ngrams in corpus.

    Parameters
    ----------
    corpus: list(list(str))
    min_appearance: int
        minimum number of appearances for a word in the entire vocabulary to be considered
    n: int | list(int)
        ngram size
        If list: find ngrams for all sizes

    Returns
    -------
    pmi: dict
        PMI of ngrams
    """
    word_counts, co_occurrence_counts, total_words, total_ngrams = get_ngrams_in_corpus(
        corpus=corpus,
        min_appearance_word=min_appearance_word,
        min_appearance_ngram=min_appearance_ngram,
        n=n,
        verbose=verbose,
    )
    pmi = PMI(word_counts, co_occurrence_counts, total_words, total_ngrams)
    return pmi


def replace_ngrams(text: str, ngrams: dict = {}, ngram_size: int = 4):
    """
    Replace the selected ngrams in a text
    """
    words = text
    if not ngram_size:
        ngram_size = max([len(k) for k in ngrams.keys()])
    for i in range(ngram_size, 1, -1):
        words = words.split()
        tuples = find_ngrams(words, i, pad_right=True)
        result = []
        skip = 0
        for t in tuples:
            if skip:
                skip -= 1
            else:
                append = ngrams.get(t, None)
                if append:
                    result.append(append)
                    skip = i - 1
                else:
                    result.append(t[0])
        words = " ".join(result)
    return str(words)


def suggest_ngrams(
    corpus: List[List[str]],
    min_appearance_word: int = 0,
    min_appearance_ngram: int = 0,
    n: Union[int, List[int]] = 2,
    stop_words: List[str] = [],
    sw_proportion: float = 0.3,
):
    """
    Obtain potential ngrams in corpus.

    Parameters
    ----------
    corpus: list(list(str))
    min_appearance: int
        minimum number of appearances for a word in the entire vocabulary to be considered
    n: int | list(int)
        ngram size
        If list: find ngrams for all sizes
    stop_words: list(str)
        List of stopwords to avoid
    sw_proportion: float
        If more than sw_proportion are stopwords, the ngram will be discarded.

    Returns
    -------
    proposed_ngrams: Counter(list(str), int)
        ngram, number of appearances
    """

    def contains_ngram(ngram, el):
        return "-".join(ngram) in "-".join(el)

    def custom_scaler(n_els, app, c_max, c_min, skew=0.5):
        n_els_weight = skew
        app_weight = 1 - skew
        # n_els_normalized = ((n_els - 1) / (10 - 1))
        # n_els_normalized = (((n_els - 1) / (10 - 1)))
        n_els_normalized = 1 / np.sqrt(np.log(n_els + 1))
        app_normalized = np.log(((c_max - c_min + 1) / (app - c_min + 1)))
        # print(app, app-100, c_max, c_max-100)
        # app_normalized = app
        # print(n_els, n_els_normalized, app_normalized)
        scaled_value = n_els_weight * n_els_normalized + app_weight * app_normalized
        return scaled_value

    # Obtain all ngrams
    word_counts, co_occurrence_counts, total_words, total_ngrams = get_ngrams_in_corpus(
        corpus=corpus,
        min_appearance_word=min_appearance_word,
        min_appearance_ngram=min_appearance_ngram,
        n=n,
        verbose=True,
    )
    # Compute PMI
    pmis = PMI(word_counts, co_occurrence_counts, total_words, total_ngrams)

    # Filter out ngrams with more than sw_proportion of elements in stopwords
    valid_keys = []
    pbar = tqdm(total=len(co_occurrence_counts), desc="Find valid ngrams", leave=True)
    for i, (k, v) in enumerate(co_occurrence_counts.items()):
        count = Counter("-".join(k).split("-"))
        c_stw = sum([count.get(s, 0) for s in stop_words])
        c = sum(count.values())
        if not c_stw / c >= sw_proportion:
            valid_keys.append((k, v))
        if not (i + 1) % 100:
            pbar.update(100)
    pbar.update((i + 1) % 100)
    pbar.close()
    ex = sorted(valid_keys, key=lambda x: len(x[0]))

    # Convert to dataframe
    all_ngrams = pd.DataFrame(
        ex, columns=["ngram", "count"]
    )  # .set_index("ngram")["count"]
    all_ngrams["_len"] = all_ngrams["ngram"].apply(len)
    data = all_ngrams.sort_values(by="_len")[["ngram", "count"]]

    # Iterate over all ngrams (shortest first)
    proposed_ngrams = []
    pbar = tqdm(total=len(data), desc="Reduce ngrams", leave=True)
    # with tqdm(total=len(data), desc="Obtain ") as pbar:
    while len(data):
        el = data.iloc[0]
        idx = el.name
        ngram = el["ngram"]
        # Filter ngrams. Obtain all that contain short sub-ngram
        sub = data.loc[
            data["ngram"]
            .apply(lambda x: x if contains_ngram(ngram, x) else None)
            .dropna()
            .index
        ]

        # Create scale factor based on length and appearances
        n_els = sub["ngram"].apply(lambda x: len("-".join(x))).values
        # n_els = sub["ngram"].apply(lambda x: len("-".join(x).split("-"))).values
        c_max = sub["count"].max()
        c_min = sub["count"].min()
        scales = custom_scaler(n_els, sub["count"], c_max, c_min, 0.5)
        # Compute average distance among ngrams
        vals = (
            sub["ngram"]
            .apply(
                lambda x: np.mean(
                    [distance("-".join(x), "-".join(el)) for el in sub["ngram"]]
                )
            )
            .values
        )
        # Compute scores based on scaled distances
        best_score = np.argmin(vals * scales)
        # Get best value and remove the rest
        if pmis[sub.iloc[best_score]["ngram"]] > 10:
            proposed_ngrams.append((sub.iloc[best_score]["ngram"], sub["count"].sum()))
            data = data.drop(sub.index)
            pbar.update(len(sub))
        else:
            data = data.drop(idx)
            pbar.update(1)
    pbar.close()

    return Counter(dict(proposed_ngrams))


if __name__ == "__main__":
    # Load data
    corpus = pd.read_parquet("../../data/metadata/df_processed_full.parquet")[
        "normalized_text"
    ]
    sug_ngrams_ng = suggest_ngrams(corpus, ngram_size=4)
    with open("../../data/ngrams/ngrams.txt", "w", encoding="utf-8") as f:
        f.writelines([" ".join([n for n in ng]) + "\n" for ng in sug_ngrams_ng])
