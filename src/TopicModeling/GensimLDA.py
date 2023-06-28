from typing import List

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore
from tqdm import trange

from src.Preprocessor.NgramProcessor import PMI
from src.utils import load_item_list, load_vocabulary

from .BaseModel import BaseModel


class GensimLDAModel(BaseModel):
    def train(self, texts: List[str], num_topics: int, iterations=400):
        # Set num topics
        self.num_topics = num_topics

        # Convert texts to format
        texts = [t.split() for t in texts]
        self.dictionary = Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        self.logger.info("Texts transformed")
        self.model = LdaModel(
            corpus,
            num_topics=self.num_topics,
            id2word=self.dictionary,
            iterations=iterations,
            random_state=42,
        )
        self.logger.info("Finished training")

        # Corpus predictions
        probs = np.zeros(shape=(len(corpus), 50))
        for d, doctopics in enumerate(self.model.get_document_topics(corpus)):
            for topic_idx, topic_prob in doctopics:
                probs[d][topic_idx] = topic_prob

        # Get topickeys
        topic_keys = dict()
        for topic_idx in range(self.num_topics):
            topic_keys[topic_idx] = " ".join(
                list(zip(*self.model.show_topic(topic_idx, 20)))[0]
            )

        # Save train data
        self._save_train_texts([" ".join(t) for t in texts])
        self._save_doctopics(probs)
        self._save_topickeys(topic_keys)

    def predict(self, texts: List[str]):
        # Convert texts to format
        texts = [t.split() for t in texts]
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        pred = np.zeros(shape=(len(texts), self.num_topics))
        for d, bow in enumerate(corpus):
            for topic_idx, topic_prob in self.model.get_document_topics(bow):
                pred[d][topic_idx] = topic_prob
        return pred

    def get_topics_words(self, n_words: int = 10):
        topics = []
        for topic_idx in range(self.model.num_topics):
            top_words = [
                word for word, _ in self.model.show_topic(topic_idx, topn=n_words)
            ]
            topics.append(top_words)
        return topics
