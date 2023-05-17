import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore
from tqdm import trange

from src.Preprocessor.NgramProcessor import PMI
from src.utils import load_item_list, load_vocabulary

from .BaseModel import BaseModel


class GensimLDAModel(BaseModel):
    def train(self, texts, iterations=400):
        texts = [t.split() for t in texts]
        self.dictionary = Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        self.model = LdaModel(
            corpus,
            num_topics=self.num_topics,
            id2word=self.dictionary,
            iterations=iterations,
            random_state=42,
        )

    def predict(self, texts):
        texts = [t.split() for t in texts]
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        pred = np.zeros(shape=(len(texts), 50))
        for d, bow in enumerate(corpus):
            for topic_idx, topic_prob in self.model.get_document_topics(bow):
                pred[d][topic_idx] = topic_prob
        return pred

    def get_topics_words(self, n_words=10):
        topics = []
        for topic_idx in range(self.model.num_topics):
            top_words = [
                word for word, _ in self.model.show_topic(topic_idx, topn=n_words)
            ]
            topics.append(top_words)
        return topics
