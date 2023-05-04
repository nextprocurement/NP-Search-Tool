from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import trange
from umap import UMAP

from .BaseModel import BaseModel


class BERTopicModel(BaseModel):
    def train(
        self,
        texts,
        embedding_model=None,
        umap_model=None,
        hdbscan_model=None,
        vectorizer_model=None,
        ctfidf_model=None,
        representation_model=None,
        verbose=True,
    ):
        # Step 1 - Extract embeddings
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )

        # Step 2 - Reduce dimensionality
        if umap_model:
            self.umap_model = umap_model
        else:
            self.umap_model = UMAP(
                n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine"
            )

        # Step 3 - Cluster reduced embeddings
        if hdbscan_model:
            self.hdbscan_model = hdbscan_model
        else:
            self.hdbscan_model = HDBSCAN(
                min_cluster_size=500,
                min_samples=2,
                metric="euclidean",
                prediction_data=True,
            )

        # Step 4 - Tokenize topics
        if vectorizer_model:
            self.vectorizer_model = vectorizer_model
        else:
            vectorizer_model = CountVectorizer(
                token_pattern=self.word_pattern,
                stop_words=self.stop_words,
                # ngram_range=(1, 2),
                # vocabulary=vocabulary,
                max_df=0.8,
                min_df=1,
            )

        # Step 5 - Create topic representation
        if ctfidf_model:
            self.ctfidf_model = ctfidf_model
        else:
            self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        # Step 6 - (Optional) Fine-tune topic representations with
        # a `bertopic.representation` model
        if representation_model:
            self.representation_model = representation_model
        else:
            self.representation_model = MaximalMarginalRelevance(
                diversity=0.3, top_n_words=20
            )

        self.model = BERTopic(
            language="multilingual",
            nr_topics=self.num_topics,
            low_memory=False,
            calculate_probabilities=False,
            seed_topic_list=None,
            embedding_model=self.embedding_model,  # Step 1 - Extract embeddings
            umap_model=self.umap_model,  # Step 2 - Reduce dimensionality
            hdbscan_model=self.hdbscan_model,  # Step 3 - Cluster reduced embeddings
            vectorizer_model=self.vectorizer_model,  # Step 4 - Tokenize topics
            ctfidf_model=self.ctfidf_model,  # Step 5 - Extract topic words
            representation_model=self.representation_model,  # Step 6 - (Optional) Fine-tune topic represenations
            verbose=verbose,
        )
        # self.model.fit(texts)
        topics, probs = self.model.fit_transform(texts)

        # Reduce outliers
        new_topics = self.model.reduce_outliers(
            texts, topics, probabilities=probs, strategy="embeddings", threshold=0.0
        )
        self.model.update_topics(
            docs=texts,
            topics=new_topics,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_model,
        )

    def predict(self, texts):
        self.model.calculate_probabilities = True
        topics, pred = self.model.transform(texts)
        self.model.calculate_probabilities = False
        return pred

    def get_topics_words(self, n_words=10):
        topics = []
        for topic_idx, topic in self.model.get_topics().items():
            top_words = [word for word, _ in topic][:n_words]
            topics.append(top_words)
        return topics
