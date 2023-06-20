import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import regex
from Levenshtein import distance
from nltk.util import ngrams
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[2]))

import argparse

from src.utils import load_item_list, replace_company_types


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


def _compute_co_occurrence_counts(corpus: List[List[str]], verbose=False):
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

    n = len(corpus)
    if verbose:
        pbar = tqdm(total=n, desc="Computing co-occurrence", leave=True)
    for i in range(n):
        d = corpus[i]
        word_counts.update(Counter(d))
        for word in d:
            w_occ = co_occurrence_counts.get(word, Counter())
            w_occ.update(Counter([w for w in d if not w == word]))
            co_occurrence_counts[word] = w_occ
        if verbose and (not (i + 1) % 100):
            pbar.update(100)
    if verbose:
        pbar.update((i + 1) % 100)
        pbar.close()
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
        words = regex.split(r"[\s\.,]+", words)
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
    proposed_ngrams: Counter(list(str), int, float, float)
        ngram, number of appearances
        "ngram", "count", "pmi", "score"
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

    def filter_ngram(ng, l=3):
        new_ng = []
        sub_ng = ng[: len(ng) // 2]
        for i, n in enumerate(sub_ng):
            # if len(n)<l:
            if n in stop_words:
                continue
            new_ng.extend(sub_ng[i:])
            break
        sub_ng = ng[len(ng) // 2 :][::-1]
        for i, n in enumerate(sub_ng):
            # if len(n)<l:
            if n in stop_words:
                continue
            new_ng.extend(sub_ng[i:][::-1])
            break
        new_ng = [el for el in new_ng if "." not in el]
        if len(new_ng) > 1:
            return tuple(new_ng)

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
    all_ngrams["_len"] = all_ngrams["ngram"].apply(lambda x: len("-".join(x)))
    data = all_ngrams.sort_values(by="_len")[["ngram", "count"]]
    data["ngram"] = data["ngram"].apply(filter_ngram)
    data = data.dropna()

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

        counts = dict()
        for s1, s2 in combinations(range(len(sub)), 2):
            seq1 = sub.iloc[s1]["ngram"]
            seq2 = sub.iloc[s2]["ngram"]

            for n in range(2, len(seq1)):
                for ng in find_ngrams(seq1, n):
                    if "-".join(ng) in "-".join(seq2):
                        counts[seq1] = (
                            counts.get(seq1, 0)
                            + sub.iloc[s1]["count"]
                            + sub.iloc[s2]["count"]
                        )
                        # counts[seq1] = counts.get(seq1, 0)+1

        # Get best value and remove the rest
        if counts:
            aux = pd.DataFrame(
                data=Counter(counts).most_common(), columns=["ngram", "count"]
            )
            # Create scale factor based on length and appearances
            n_els = aux["ngram"].apply(lambda x: len("-".join(x))).values
            # n_els = sub["ngram"].apply(lambda x: len("-".join(x).split("-"))).values
            c_max = aux["count"].max()
            c_min = aux["count"].min()
            scales = custom_scaler(n_els, aux["count"], c_max, c_min, 0.5)
            aux["pmi"] = aux["ngram"].apply(lambda x: pmis.get(x, 0))
            aux["score"] = aux["pmi"] * scales
            data = data.drop(sub.index)
            pbar.update(len(sub))
            proposed_ngrams.append(aux.iloc[aux["score"].argmin()].values)
        else:
            data = data.drop(idx)
            pbar.update(1)
    pbar.close()

    return proposed_ngrams


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Script parameters")

    # Add the arguments
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/metadata/df_processed_full.parquet",
        help="The path to the corpus location",
    )
    parser.add_argument(
        "--stop-words",
        type=str,
        default="data/stopwords",
        help="The path to the stop words",
    )
    parser.add_argument(
        "--use-stopwords",
        nargs="+",
        type=str,
        default="all",
        help="List of resource files. Can be a single string or a list of strings.",
    )
    parser.add_argument(
        "--min_appearance_word",
        type=int,
        default=1,
        help="The minimum number of appearances for a word",
    )
    parser.add_argument(
        "--min_appearance_ngram",
        type=int,
        default=1,
        help="The minimum number of appearances for an n-gram",
    )
    parser.add_argument(
        "--n", type=int, default=2, help="The maximum size of an n-gram"
    )
    parser.add_argument(
        "--sw_proportion",
        type=float,
        default=0.1,
        help="Ignore ngram if the proportion of stopwords is greater than this value",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/ngrams/new_ngrams.txt",
        help="The path to the output directory",
    )

    # Execute the parse_args() method
    args = parser.parse_args()
    print("Config:")
    print("-" * 120)
    for k, v in vars(args).items():
        print(f"{f'{k}:':<30}{v}")
    print("-" * 120)

    # Load data
    print("Loading corpus...")
    corpus = pd.read_parquet(args.input_file)
    corpus = corpus.loc[corpus["lang"] == "es", "normalized_text"]
    # print(corpus["lang"].value_counts())
    # print(sum(corpus["lang"].apply(lambda x: "ca" in x)))
    stop_words = load_item_list(args.stop_words, use_item_list=args.use_stopwords)
    use_stopwords = sorted(
        list(set([w.lower() for w in stop_words])), key=len, reverse=True
    )

    # Suggest
    print("Cleaning corpus...")
    batch = 20_000
    text_words = []
    total = len(corpus)
    pbar = tqdm(total=total, desc="Cleaning corpus", leave=True)
    for i in range(0, total, batch):
        sub_c = corpus[i : i + batch]
        text_words.extend(
            sub_c.apply(replace_company_types, remove_type=True)
            .apply(str.split)
            .tolist()
        )
        pbar.update(min(batch, total - i))
    pbar.close()

    print("Suggesting ngrams...")
    sug_ngrams_ng = suggest_ngrams(
        corpus=text_words,
        min_appearance_word=args.min_appearance_word,
        min_appearance_ngram=args.min_appearance_ngram,
        n=list(range(2, args.n)),
        stop_words=use_stopwords,
        sw_proportion=args.sw_proportion,
    )
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.writelines([" ".join(ng[0]) + "\n" for ng in sug_ngrams_ng])

