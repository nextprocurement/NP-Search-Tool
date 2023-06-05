import logging
from abc import abstractmethod
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import regex
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange

from src.Preprocessor.NgramProcessor import PMI


class BaseModel:
    def __init__(
        self,
        model_dir: Union[str, Path],
        stop_words: list = [],
        word_min_len: int = 2,
        logger:logging.Logger=None,
    ):
        """
        model_dir: str|Path
            Directory where model will be saved

        """
        # Set logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

        # Model params
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.num_topics = None
        self.stop_words = stop_words
        self.word_pattern = (
            f"(?<![a-zA-Z\u00C0-\u024F\d\-\_])"
            f"[a-zA-Z\u00C0-\u024F]"
            f"(?:[a-zA-Z\u00C0-\u024F]|(?!\d{{4}})[\d]|[\-\_\·\.'](?![\-\_\·\.'])){{{word_min_len - 1},}}"
            f"(?<![\-\_\·\.'])[a-zA-Z\u00C0-\u024F\d]?"
            f"(?![a-zA-Z\u00C0-\u024F\d])"
        )
        self.sentence_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

    @abstractmethod
    def train(self, texts) -> None:
        pass

    @abstractmethod
    def predict(self, texts) -> np.ndarray:
        pass

    @abstractmethod
    def get_topics_words(self, n_words: int = 10) -> List[List[str]]:
        """
        Get a list of `n_words` for each topic.
        """
        pass

    def get_topic_words(self, topic: int, n_words: int = 10) -> List[str]:
        """
        Get a list of `n_words` for the selected topic.
        """
        return self.get_topics_words(n_words=n_words)[topic]

    def show_words_per_topic(self, n_words=10) -> pd.DataFrame:
        topic_words = self.get_topics_words(n_words)
        df_topic_words = pd.DataFrame(
            topic_words, columns=[f"Word_{i}" for i in range(n_words)]
        )
        df_topic_words.index.name = "Topic"
        return df_topic_words

    def show_topics_per_document(self, texts) -> pd.DataFrame:
        doc_topics = self.predict(texts)
        df_doc_topics = pd.DataFrame(
            doc_topics, columns=[f"Topic_{i}" for i in range(doc_topics.shape[1])]
        )
        df_doc_topics.index.name = "Document"
        return df_doc_topics

    def get_topics_diversity(self, n_words=20) -> float:
        topic_words = self.get_topics_words(n_words=n_words)
        unique_words = len(set([word for topic in topic_words for word in topic]))
        return unique_words / (self.num_topics * n_words)

    def get_topics_pmi(self):
        """
        Computes the average PMI by topic.
        1. Obtain topic words.
        2. Compute total unique words and total co-occurrence.
        3. Compute PMI by co-occurrence.
        4. Average PMI of word pairs in topic
        """
        # 1.
        n_words = 20
        topic_words = self.get_topics_words(n_words=n_words)

        # 2.
        word_counts = Counter()
        co_occurrence_counts = dict()
        for topic in topic_words:
            word_counts.update(Counter(topic))
            for word in topic:
                w_occ = co_occurrence_counts.get(word, Counter())
                w_occ.update(Counter([w for w in topic if not w == word]))
                co_occurrence_counts[word] = w_occ
        co_occurrence_counts = dict(
            [
                (tuple(sorted([w1, w2])), occ)
                for w1, v in co_occurrence_counts.items()
                for w2, occ in v.items()
            ]
        )
        total_words = sum(word_counts.values())
        total_ngrams = sum(co_occurrence_counts.values())

        # 3.
        pmi = PMI(word_counts, co_occurrence_counts, total_words, total_ngrams)

        # 4.
        topic_pmi = [
            np.mean([pmi[tuple(sorted(c))] for c in combinations(tw, 2)])
            for tw in topic_words
        ]

        return topic_pmi

    # Search methods
    # def search_topics(self, query: str, top_n=5, method="embeddings"):
    #     """
    #     Search `top_n` closest topics based on query or keyword.
    #     """
    #     # remove punctuation from text (commas, dots, etc.)
    #     query = regex.findall(self.word_pattern, query)
    #     query = " ".join(query)
    #     if method == "embeddings":
    #         query_vector = self.sentence_model.encode([query])
    #         topic_vectors = self.sentence_model.encode(
    #             self.get_words_per_topic(10).values
    #         )
    #         top_topics = cosine_similarity(query_vector, topic_vectors)

    #     elif method == "keywords":
    #         top_topics = self.get_topics_per_document([query]).values

    #     top_topics = np.argsort(top_topics, axis=1)[0, ::-1][:top_n]
    #     close_topics = self.get_words_per_topic(10).loc[top_topics]
    #     return close_topics

    def find_close_topics_by_appearance(self, query: str, top_n=5):
        """
        Given a query, find closest topics using word appearance.
        """
        query = query.lower()
        query_words = regex.findall(self.word_pattern, query)
        related_topics = []
        for word in query_words:
            related_topics.extend(
                [n for n, topic in enumerate(self.get_topics_words()) if word in topic]
            )
        return dict(Counter(related_topics).most_common(top_n))

    def find_close_topics_by_embeddings(self, query: str, top_n=5):
        """
        Given a query, find closest topics using embeddings.
        """
        # Encode query
        query_words = regex.findall(self.word_pattern, query.lower())
        query_words = query_words + [" ".join(query_words)]
        query_vector = self.sentence_model.encode(query_words).reshape(
            len(query_words), -1
        )

        # Encode topics
        topics = [" ".join(t) for t in self.get_topics_words()]
        topic_vectors = self.sentence_model.encode(topics)

        # Get closest topics
        similarities = cosine_similarity(query_vector, topic_vectors)
        top = np.argsort(similarities, axis=1)[:, ::-1][:, :top_n]
        most_similar_topics = (
            pd.DataFrame(
                [(i, similarities[n, i]) for n, t in enumerate(top) for i in t],
                columns=["topic", "similarity"],
            )
            .groupby("topic")
            .mean()
        )
        return most_similar_topics["similarity"].nlargest(top_n).to_dict()

    def find_close_words_by_topic(self, query: str, top_n=5):
        """
        Given a query, find closest words using word appearance in topics.
        """
        query = query.lower()
        query_words = regex.findall(self.word_pattern, query)
        related_words = []
        for word in query_words:
            for topic in self.get_topics_words():
                if word in topic:
                    related_words.extend([w for w in topic if not w == word])
        return dict(Counter(related_words).most_common(top_n))

    def find_close_words_in_vocab(
        self, query: str, word_embeddings: pd.Series, top_n=5
    ):
        """
        Given a query and word embeddings, find closest terms.
        The query is divided into individual terms.
        A scaled similarity will be computed taking into account the separated terms and whole query.
        """

        # Set scale parameter
        def scale(size, n):
            return ((n // (size - 1)) + (1 / size)) * np.sqrt(0.5)

        # Encode query
        query_words = regex.findall(self.word_pattern, query.lower())
        query_words = query_words + [" ".join(query_words)]
        query_vector = self.sentence_model.encode(query_words).reshape(
            len(query_words), -1
        )

        # Get closest words
        similarities = cosine_similarity(
            query_vector,
            np.array(
                word_embeddings.drop(query_words, errors="ignore").values.tolist()
            ),
        )
        top = np.argsort(similarities, axis=1)[:, ::-1][:, :top_n]
        size = len(top)
        most_similar_words = (
            pd.DataFrame(
                [
                    (
                        word_embeddings.drop(query_words, errors="ignore").index[i],
                        similarities[n, i],
                        similarities[n, i] * scale(size, n),
                    )
                    for n, t in enumerate(top)
                    for i in t
                ],
                columns=["word", "similarity", "similarity_scaled"],
            )
            .groupby("word")
            .mean()
        )
        return most_similar_words["similarity_scaled"].nlargest(top_n).to_dict()

    # def search_keywords(self, query):
    #     n_words = 20
    #     topic_words = self.get_topics_words(n_words=n_words)
    #     query_words = query.lower().split()
    #     result_docs = set.intersection(*(inverted_index[word] for word in query_words))
    #     return result_docs

    # TODO:
    # - Word Embedding-based Centroid Distance (WE-CD) [Bianchi et al., 2021b]
    # - Word Embedding-based Pairwise Distance (WE-PD) [Terragni et al., 2021]
    # - Word Embedding-based Inverted Rank-Biased Overlap (WE-IRBO) [Terragni et al., 2021]
