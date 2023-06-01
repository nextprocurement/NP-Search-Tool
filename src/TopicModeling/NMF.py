from typing import List, Union

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import trange

from .BaseModel import BaseModel


class NMFModel(BaseModel):
    def train(
        self,
        texts: List[str],
        num_topics: int,
        vectorizer: Union[str, TfidfVectorizer, CountVectorizer] = "tfidf",
        max_df: float = 0.8,
        min_df: Union[float, int] = 1,
    ):
        # Set num topics
        self.num_topics = num_topics

        # Set vectorizer
        if vectorizer == "tfidf":
            vectorizer = TfidfVectorizer(
                token_pattern=self.word_pattern,
                stop_words=self.stop_words,
                # ngram_range=(1, 2),
                # vocabulary=vocabulary,
                max_df=max_df,
                min_df=min_df,
            )
        elif vectorizer == "count":
            vectorizer = CountVectorizer(
                token_pattern=self.word_pattern,
                stop_words=self.stop_words,
                # ngram_range=(1, 2),
                # vocabulary=vocabulary,
                max_df=max_df,
                min_df=min_df,
            )

        self.vectorizer = vectorizer
        tfidf = self.vectorizer.fit_transform(texts)
        self.model = NMF(n_components=self.num_topics, random_state=42).fit(tfidf)

    def predict(self, texts: List[str]):
        tfidf = self.vectorizer.transform(texts)
        return self.model.transform(tfidf)

    def get_topics_words(self, n_words: int = 10):
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(self.model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[: -n_words - 1 : -1]]
            topics.append(top_words)
        return topics
