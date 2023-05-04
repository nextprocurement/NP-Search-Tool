import numpy as np
import tomotopy as tp
from tqdm import trange

from .BaseModel import BaseModel


class TomotopyCTModel(BaseModel):
    def train(self, texts, iterations=400):
        self.model = tp.CTModel(k=self.num_topics, min_cf=0, min_df=10, seed=42)
        texts = [t.split() for t in texts]
        for text in texts:
            self.model.add_doc(text)
        # self.model.train(0)
        t = trange(0, iterations, 10, desc="", leave=True)
        for i in t:
            t.set_description(f"Iteration:{i}\tLL:{self.model.ll_per_word:.3f}")
            t.refresh()
            self.model.train(10)

    def predict(self, texts):
        texts = [t.split() for t in texts]
        doc_inst = [self.model.make_doc(text) for text in texts]
        topic_prob, log_ll = self.model.infer(doc_inst)
        return np.array(topic_prob)

    def get_topics_words(self, n_words=10):
        topics = []
        for topic_idx in range(self.num_topics):
            top_words = [
                word for word, _ in self.model.get_topic_words(topic_idx, top_n=n_words)
            ]
            topics.append(top_words)
        return topics
