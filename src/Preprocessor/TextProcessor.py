import logging
from typing import List, Union

import enchant
import nltk
import pandas as pd
import regex
import spacy
from nltk.stem.snowball import SnowballStemmer
from unidecode import unidecode

from .Lemmatizer import Lemmatizer
from .NgramProcessor import replace_ngrams


class TextPreprocessor:
    def __init__(
        self,
        methods: List[str] = ["clean_text"],
        stopwords: List[str] = [],
        vocabulary: dict = {},
        ngrams: List[str] = [],
        logger: logging.Logger = None,
    ):
        """
        Parameters
        ----------
        methods: list(str) [Optional]
            The elements to use in the pipeline:
            - lowercase
            - remove_tildes
            - remove_extra_spaces
            - remove_punctuation
            - remove_urls
            - --stem_text--
            - lemmatize_text
            - pos_tagging
            - clean_text
            - convert_ngrams
            - remove_stopwords
            - correct_spelling
            - tokenize_text

        stopwords: list(str) [Optional]
            List of strings to remove from text.
        vocabulary: dict [Optional]
            Dictionary where keys are terms and values are the transformation of that term (e.g. {term in femenine: term in masculine})
        ngrams: list(str) [Optional]
            list of ngrams to find in text and convert into unique word (e.g. arrtificial intelligence->arrtificial-intelligence)
        """
        # Set logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

        # Params
        self.methods = methods
        self.stemmer = SnowballStemmer("spanish")
        self.spellchecker = enchant.Dict("es_ES")  # Initialize the spellchecker
        # self.spellchecker = SpellChecker(language="es", distance=1)
        # self.spell = Speller(lang="es", fast=False)

        # Lemmatization type
        if vocabulary:
            self.logger.info("Using custom lemmatizer.")
            self.vocabulary = vocabulary
            self.lemmatizer = Lemmatizer(vocabulary)
            self.nlp = spacy.load(
                "es_dep_news_trf",
                disable=["tok2vec", "parser", "attribute_ruler", "lemmatizer", "ner"],
            )  # es_core_news_lg | es_dep_news_trf
        else:
            self.logger.info("Using predefined lemmatizer.")
            self.nlp = spacy.load(
                "es_dep_news_trf"
            )  # es_core_news_lg | es_dep_news_trf

        # Stopword definition
        if stopwords:
            self.stopwords = set(stopwords)

            self._stw_cased = set(self.stopwords)
            self._stw_uncased = set([w.lower() for w in self.stopwords])
            pattern = (
                r"(?<![a-zA-Z\u00C0-\u024F\d\-\·\.\'])(?:"
                + "|".join(
                    regex.escape(stopword)
                    for stopword in sorted(list(self._stw_cased), key=len, reverse=True)
                )
                + r")(?![a-zA-Z\u00C0-\u024F\d\-\·\.\'])"
            )
            self._stopwords_regex_cased = regex.compile(pattern)
            pattern = (
                r"(?<![a-zA-Z\u00C0-\u024F\d\-\_\·\.\'])(?:"
                + "|".join(
                    regex.escape(stopword)
                    for stopword in sorted(
                        list(self._stw_uncased), key=len, reverse=True
                    )
                )
                + r")(?![a-zA-Z\u00C0-\u024F\d\-\_\·\.\'])"
            )
            self._stopwords_regex_uncased = regex.compile(pattern, regex.IGNORECASE)

        # Ngrams
        if ngrams:
            self.ngrams = {tuple(el.split()): el.replace(" ", "-") for el in ngrams}
            self._ngram_size = max([len(k) for k in self.ngrams.keys()])

    def __repr__(self):
        return f"TextPreprocessor(methods={self.methods})"

    def lowercase(self, text: str, rtype="str"):
        result = text.lower()
        return result.split() if rtype == "list" else result

    def remove_tildes(self, text: str, rtype="str"):
        result = unidecode(text)
        return result.split() if rtype == "list" else result

    def remove_extra_spaces(self, text: str, rtype="str"):
        # result = regex.sub(r"\s+", " ", text)
        result = text.split()
        return result if rtype == "list" else " ".join(result)

    def remove_punctuation(self, text: str, rtype="str"):
        result = regex.sub(r"\p{P}", "", text)
        return result.split() if rtype == "list" else result

    def remove_urls(self, text: str, rtype="str"):
        result = regex.sub(r"(http|www)\S+", " ", text)
        return result.split() if rtype == "list" else result

    # def stem_text(self, text: Union[List[str], str], rtype="list"):
    #     if isinstance(text, str):
    #         text = text.split()
    #     stemmed_words = [self.stemmer.stem(word) for word in text]
    #     return stemmed_words if rtype == "list" else " ".join(stemmed_words)

    def lemmatize_text(self, text: str, rtype="str"):
        doc = self.nlp(text)
        if hasattr(self, "lemmatizer"):
            # lemmatized_words = [self.lemmatizer.lemmatize_spanish(w) for w in doc]
            lemmatized_words = (
                pd.Series(doc).apply(self.lemmatizer.lemmatize_spanish).tolist()
            )
        else:
            lemmatized_words = [token.lemma_ for token in doc]
        return lemmatized_words if rtype == "list" else " ".join(lemmatized_words)

    def tokenize_text(self, text: str):
        words = nltk.word_tokenize(text)
        return words

    def pos_tagging(self, text: str, rtype="str"):
        words = self.tokenize_text(text)
        tagged_words = nltk.pos_tag(words)
        result = [f"{word}/{tag}" for word, tag in tagged_words]
        return result if rtype == "list" else " ".join(result)

    def clean_text(self, text: str, min_len: int = 2, rtype="str"):
        """
        min_len: int
            Minimum number of letters for a word to be included. Must be at least 2.

            Regex pattern that identifies words that may have the following characters:
            - letters: latin letters (including diacritics)
            - numbers
            - special characters: dash, forward and backward slash and vertical bar

            Word consist of a minimum of `min_len` characters with these restrictions:
            - starts with a letter
            - ends with letter or number
            - up to 3 consecutive numbers before another non-numeric character appears
            - words can have multiple special characters, but not consecutive
        """

        if min_len < 1:
            raise ValueError("Minimum length must be at least 1.")

        pattern = (
            f"(?<![a-zA-Z\u00C0-\u024F\d\-\_])"
            f"[a-zA-Z\u00C0-\u024F]"
            f"(?:[a-zA-Z\u00C0-\u024F]|(?!\d{{4}})[\d]|[\-\_\·\.'](?![\-\_\·\.'])){{{min_len - 1},}}"
            f"(?<![\-\_\·\.'])[a-zA-Z\u00C0-\u024F\d]?"
            f"(?![a-zA-Z\u00C0-\u024F\d])"
        )
        cleaned_text = regex.findall(pattern, text, flags=regex.UNICODE)
        if rtype == "str":
            cleaned_text = str(" ".join(cleaned_text))
        # cleaned_text = regex.sub(pattern, "", text)
        return cleaned_text

    def convert_ngrams(self, text: str, include_stopwords: bool = True, rtype="str"):
        """
        Convert ngrams in text to unigrams separated by "-".
        If `include_stopwords`: also replace stopwords with spaces (" ")
        """
        if include_stopwords:
            stw = {
                tuple(s.split()): s.replace(" ", "-")
                for s in self.stopwords
                if " " in s
            }
            ngrams = {**self.ngrams, **stw}
            ngram_size = max([len(k) for k in ngrams.keys()])
        else:
            ngrams = self.ngrams
            ngram_size = self._ngram_size

        ngram_text = replace_ngrams(text, ngrams, ngram_size=ngram_size)
        if rtype == "list":
            ngram_text = ngram_text.split()
        return ngram_text

    def remove_stopwords(
        self,
        text: Union[List[str], str],
        rtype="str",
        ignore_case=True,
        # fast: bool = False,
    ):
        """
        If `fast`: check only words separated by space (" ").
        If there is a stopword that contains space, it will not be removed.
        This is useful if previously `convert_ngrams` has been called with stopwords option.
        """

        """
        # If the stopword pattern does not exist create it
        if not hasattr(self, "_stopwords_regex"):
            print("Create regex first time")
            if ("lowercase" in self.methods) and (
                self.methods.index("lowercase") < self.methods.index("remove_stopwords")
            ):
                stw = list(set([w.lower() for w in self.stopwords]))
            stw = sorted(stw, key=len, reverse=True)
            pattern = (
                r"(?<![a-zA-Z\u00C0-\u024F\d\-\_])(?:"
                + "|".join(regex.escape(stopword) for stopword in stw)
                + r")(?![a-zA-Z\u00C0-\u024F\d\-\_])"
            )
            self._stopwords_regex = regex.compile(pattern, regex.IGNORECASE)
        """
        if isinstance(text, str):
            if ignore_case:
                filtered_text = self._stopwords_regex_uncased.sub("", text)
            else:
                filtered_text = self._stopwords_regex_cased.sub("", text)
            filtered_text = self.remove_extra_spaces(filtered_text, rtype="str")
            if rtype == "list":
                filtered_text = filtered_text.split()
        else:
            if ignore_case:
                filtered_text = [
                    el for el in text if el.lower() not in self._stw_uncased
                ]
            else:
                filtered_text = [el for el in text if el not in self._stw_cased]
            if rtype == "str":
                filtered_text = " ".join(filtered_text)
        return filtered_text

    def correct_spelling(self, text: str, rtype="str"):
        words = text.split()
        corrected_words = []
        for word in words:
            # suggestion = self.spellchecker.correction(word)
            # if suggestion:
            #     corrected_words.append(suggestion)
            # else:
            #     corrected_words.append(word)
            if self.spellchecker.check(word):
                corrected_words.append(word)
            else:
                suggestions = self.spellchecker.suggest(word)
                if len(suggestions) > 0:
                    corrected_words.append(suggestions[0])
                else:
                    corrected_words.append(word)
        return corrected_words if rtype == "list" else " ".join(corrected_words)

    def preprocess(self, text: str, rtype=None):
        for el in self.methods:
            method = ""
            args = ()
            kwargs = {}
            if isinstance(el, str):
                # Method with no arguments
                method = el
            elif isinstance(el, tuple):
                # Method with arguments
                for a in el:
                    if isinstance(a, str):
                        method = a
                    elif isinstance(a, tuple):
                        args = a
                    elif isinstance(a, dict):
                        kwargs = a
            if hasattr(self, method):
                # self.logger.info(f"Processing text: '{method}'.")
                if rtype:
                    text = getattr(self, method)(text, *args, **kwargs, rtype=rtype)
                else:
                    text = getattr(self, method)(text, *args, **kwargs)
            else:
                # self.logger.warning(f"Method '{method}' does not exist.")
                pass
        return text
