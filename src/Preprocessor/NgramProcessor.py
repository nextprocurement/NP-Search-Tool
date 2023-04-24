from collections import Counter
from typing import List, Union

import numpy as np
import pandas as pd
from nltk.util import ngrams


def find_ngrams(text: List[str], n: int = 2, **kwargs):
    """
    Get a list of all ngrams of `n` words that appear in a text.
    """
    n_grams = list(ngrams(text, n, **kwargs))
    return n_grams


def _compute_word_counts(
    corpus: List[List[str]],
    min_appearance_word: int = 0,
    min_appearance_ngram: int = 0,
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
    """
    word_counts = Counter()
    ngrams = Counter()

    [
        (word_counts.update(Counter(d)), ngrams.update(Counter(find_ngrams(d, n))))
        for d in corpus
    ]

    valid_vocab = set(word_counts.keys())
    valid_ngrams = set(ngrams.keys())

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


def PMI(word_counts, co_occurrence_counts, total_words, total_ngrams):
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
    pmi = {}
    if isinstance(n, int):
        (
            word_counts,
            co_occurrence_counts,
            total_words,
            total_ngrams,
        ) = _compute_word_counts(corpus, min_appearance_word, min_appearance_ngram, n)
        pmi = PMI(word_counts, co_occurrence_counts, total_words, total_ngrams)
    else:
        for el in n:
            (
                word_counts,
                co_occurrence_counts,
                total_words,
                total_ngrams,
            ) = _compute_word_counts(
                corpus, min_appearance_word, min_appearance_ngram, el
            )
            pmi.update(
                PMI(word_counts, co_occurrence_counts, total_words, total_ngrams)
            )
    return pmi


def replace_ngrams(text: str, ngrams: dict = {}, ngram_size: int = 4):
    """
    Replace the selected ngrams in a text
    """
    words = text
    if not ngram_size:
        ngram_size = max([len(k) for k in ngrams.keys()])
    for i in list(range(2, ngram_size + 1))[::-1]:
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


def suggest_ngrams(corpus: pd.Series, ngram_size: int = 4, stop_words: list = []):
    """
    Generate a list of potential ngrams of size 2 up to size `ngram_size` from a corpus.
    Parameters
    ----------
    corpus: pd.Series
        Series of text.
    ngram_size: int
        Size of the desired ngrams. min=2.
    stop_words: list
        If the ngram is in this list, they will not be included.
    """
    sug_ngrams = []
    pmi_ngrams = dict()
    if isinstance(corpus[0], str):
        corpus = corpus.str.split()
    t = range(ngram_size, 1, -1)
    for i in t:
        # t.set_description(f"{len(pmi_ngrams)} ngrams. {i}-grams.")
        # t.refresh()
        aux_ng = compute_pmi_ngram_corpus(
            corpus,
            min_appearance_word=1000,
            min_appearance_ngram=500,
            n=i,
        )
        pmi_ngrams.update(aux_ng)

    # Choose suitable ngrams
    pmi_ngrams = Counter(pmi_ngrams).most_common()
    sug_ngrams = [el[0] for el in pmi_ngrams if el[1] >= 10]
    sug_ngrams = {ng: "-".join(ng) for ng in sug_ngrams}
    sug_ngrams = {
        k: v for k, v in sug_ngrams.items() if all([el not in stop_words for el in k])
    }

    return sug_ngrams


if __name__=="__main__":
    # Load data
    corpus = pd.read_parquet("../../data/metadata/df_processed_full.parquet")["normalized_text"]
    sug_ngrams_ng = suggest_ngrams(corpus, ngram_size=4)
    with open("../../data/ngrams/ngrams.txt", "w", encoding="utf-8") as f:
        f.writelines([" ".join([n for n in ng])+"\n" for ng in sug_ngrams_ng])
