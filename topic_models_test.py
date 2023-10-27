import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tabulate import tabulate
from umap import UMAP

from src.TopicModeling import BaseModel

sys.path.append(str(Path(__file__).parents[1]))
from src.TopicModeling import (
    BERTopicModel,
    GensimLDAModel,
    MalletLDAModel,
    NMFModel,
    TomotopyCTModel,
    TomotopyLDAModel,
)
from src.utils import load_item_list, set_logger, train_test_split

if __name__ == "__main__":
    # Set logger
    logger = set_logger(console_log=True, file_log=True)

    # Parse args
    parser = argparse.ArgumentParser(description="Process options")
    parser.add_argument(
        "--options", default="config/options.yaml", help="Path to options YAML file"
    )
    args = parser.parse_args()

    with open(args.options, "r") as f:
        options = dict(yaml.safe_load(f))

    #################################

    # Access options
    # Set logger
    dir_logger = Path(options.get("dir_logger", "app.log"))
    console_log = options.get("console_log", True)
    file_log = options.get("file_log", True)
    logger = set_logger(console_log=console_log, file_log=file_log, file_loc=dir_logger)

    # Config
    num_topics = int(options.get("num_topics", 50))
    # Set default values if not provided in the YAML file
    dir_data = Path(options.get("dir_data", "data"))
    # Files directories
    dir_text_processed = Path(options.get("dir_text_processed"))
    dir_output_models = Path(options.get("dir_output_models", "output_models"))
    dir_mallet = Path(options.get("dir_mallet"))
    # List loading options
    use_stopwords = options.get("use_stopwords", False)

    # Load data
    if use_stopwords:
        stop_words = load_item_list(dir_data, "stopwords", use_item_list=use_stopwords)
    else:
        stop_words = []

    #################################

    # Load data
    subsample = int(options.get("subsample", 0))
    df_processed = pd.read_parquet(dir_text_processed).dropna()
    df_sample = df_processed.loc[
        df_processed["preprocessed_text"].apply(lambda x: len(x.split()) > 5),
        "preprocessed_text",
    ]
    if subsample:
        if subsample > len(df_sample):
            logger.warning(
                f"Subsample of {subsample} is larger than population. Setting subsample to max value ({len(df_processed)} samples)."
            )
            subsample = len(df_sample)
        df_sample = df_sample.sample(n=subsample, random_state=42)
    texts_train, texts_test = train_test_split(df_sample, 0.0)
    logger.info("Data loaded.")
    logger.info(f"Train: {len(texts_train)} documents. Test: {len(texts_test)}.")

    # Compare
    models_names = [
        "Mallet",
        "NMF",
        "GensimLDA",
        "TomotopyLDA",
        "TomotopyCT",
        "BERTopic",
    ]
    # models_names = ["NMF"]
    models: List[BaseModel] = []
    times = {}
    for m_name in models_names:
        logger.info(f"Model: {m_name}")
        t0 = time.time()
        # Mallet
        if m_name == "Mallet":
            mallet_model = MalletLDAModel(
                model_dir=dir_output_models.joinpath("Mallet"),
                # stop_words=stop_words,
                word_min_len=4,
                mallet_path=dir_mallet,
                logger=logger,
            )
            mallet_model.train(
                texts_train,
                num_topics=num_topics,
                alpha=5,
                optimize_interval=10,
                num_threads=4,
                num_iterations=1000,
            )
            models.append(mallet_model)

        # NMF
        elif m_name == "NMF":
            nmf_model = NMFModel(
                model_dir=dir_output_models.joinpath("NMF"),
                # stop_words=stop_words,
                word_min_len=4,
                logger=logger,
            )
            nmf_model.train(texts_train, num_topics=num_topics)
            models.append(nmf_model)

        # GensimLDA
        elif m_name == "GensimLDA":
            lda_gensim_model = GensimLDAModel(
                model_dir=dir_output_models.joinpath("Gensim"),
                # stop_words=stop_words,
                word_min_len=4,
                logger=logger,
            )
            lda_gensim_model.train(texts_train, num_topics=num_topics, iterations=1000)
            models.append(lda_gensim_model)

        # TomotopyLDA
        elif m_name == "TomotopyLDA":
            tomotopy_lda_model = TomotopyLDAModel(
                model_dir=dir_output_models.joinpath("TomotopyLDA"),
                # stop_words=stop_words,
                word_min_len=4,
                logger=logger,
            )
            tomotopy_lda_model.train(
                texts_train, num_topics=num_topics, iterations=1000
            )
            models.append(tomotopy_lda_model)

        # TomotopyLDA
        elif m_name == "TomotopyCT":
            tomotopy_lda_model = TomotopyCTModel(
                model_dir=dir_output_models.joinpath("TomotopyCT"),
                # stop_words=stop_words,
                word_min_len=4,
                logger=logger,
            )
            tomotopy_lda_model.train(
                texts_train, num_topics=num_topics, iterations=1000
            )
            models.append(tomotopy_lda_model)

        # BERTopic
        elif m_name == "BERTopic":
            # Word patterns
            min_len = 4
            word_pattern = (
                f"(?<![a-zA-Z\u00C0-\u024F\d\-\_])"
                f"[a-zA-Z\u00C0-\u024F]"
                f"(?:[a-zA-Z\u00C0-\u024F]|(?!\d{{4}})[\d]|[\-\_\·\.'](?![\-\_\·\.'])){{{min_len - 1},}}"
                f"(?<![\-\_\·\.'])[a-zA-Z\u00C0-\u024F\d]?"
                f"(?![a-zA-Z\u00C0-\u024F\d])"
            )
            # Step 1 - Extract embeddings
            sentence_model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
            # Step 2 - Reduce dimensionality
            umap_model = UMAP(
                n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine"
            )
            # Step 3 - Cluster reduced embeddings
            hdbscan_model = HDBSCAN(
                min_cluster_size=500,
                min_samples=2,
                metric="euclidean",
                prediction_data=True,
            )
            # Step 4 - Tokenize topics
            vectorizer_model = CountVectorizer(
                token_pattern=word_pattern,
                stop_words=stop_words,
                max_df=0.8,
                min_df=1,
            )
            # Step 5 - Create topic representation
            ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
            # Step 6 - (Optional) Fine-tune topic representations with
            # a `bertopic.representation` model
            representation_model = MaximalMarginalRelevance(
                diversity=0.3, top_n_words=20
            )

            bertopic_model = BERTopicModel(
                model_dir=dir_output_models.joinpath("BERTopic"),
                # stop_words=stop_words,
                word_min_len=4,
                logger=logger,
            )
            bertopic_model.train(
                texts_train,
                num_topics=num_topics,
                embedding_model=sentence_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
                representation_model=representation_model,
                verbose=True,
            )
            models.append(bertopic_model)

        else:
            logger.warning(f"Model '{m_name}' not in available models. Skipping.")
            models_names.remove(m_name)

        t1 = time.time()
        print("-" * 100)
        times[f"train_{m_name}"] = t1 - t0

    # Comparison
    query = "contrato de servicio de instalación de un gestor de datos reutilizables así como el apoyo técnico relacionado con el mismo"

    info_headers = [
        "Topic diversity",
        "Avg. topic PMI",
    ]
    pred_topics_headers = [
        "TP words predicted",
        "TP words by appearance",
        "TP words by embedding",
    ]
    info = [[m] for m in models_names]
    pred_topics = [[m] for m in models_names]

    close_docs_headers = [
        "doc",
    ]
    closest_docs = [[m] for m in models_names]

    for n, model in enumerate(models):
        info[n].append(f"{model.get_topics_diversity(20):.3f}")
        info[n].append(f"{np.mean(model.get_topics_pmi()):.3f}")

        #####################################
        t0 = time.time()
        topTP_top = model.find_close_topics(query=query, top_n=5)
        pred_topics[n].append(
            "\n".join(
                f"  {f'{k}:':<5}{v}"
                for k, v in {
                    t: ",".join(model.get_topic_words(topic=t, n_words=5))
                    for t in topTP_top.keys()
                }.items()
            )
        )
        logger.info(f"{models_names[n]} topTP_top")

        topTP_app = model.find_close_topics_by_appearance(query=query, top_n=5)
        pred_topics[n].append(
            "\n".join(
                f"  {f'{k}:':<5}{v}"
                for k, v in {
                    t: ",".join(model.get_topic_words(topic=t, n_words=5))
                    for t in topTP_app.keys()
                }.items()
            )
        )
        logger.info(f"{models_names[n]} topTP_app")

        topTP_emb = model.find_close_topics_by_embeddings(query=query, top_n=5)
        pred_topics[n].append(
            "\n".join(
                f"  {f'{k}:':<5}{v}"
                for k, v in {
                    t: ",".join(model.get_topic_words(topic=t, n_words=5))
                    for t in topTP_emb.keys()
                }.items()
            )
        )
        logger.info(f"{models_names[n]} topTP_emb")

        close_docs = model.find_close_docs(query)
        closest_docs[n].append("\n".join(close_docs))
        logger.info(f"{models_names[n]} close_docs")

        t1 = time.time()
        print("-" * 100)
        times[f"pred_{models_names[n]}"] = t1 - t0
        # pred_topics[n].append(model.predict(texts_test))

    logger.info(f"\n{tabulate(info, headers=info_headers, tablefmt='mixed_grid')}")
    logger.info(
        f"\n{tabulate(pred_topics, headers=pred_topics_headers, tablefmt='mixed_grid')}"
    )
    logger.info(
        "Times:\n" + "\n".join(f"  {f'{k}:':<20}{v:>10.3f}" for k, v in times.items())
    )

    logger.info(
        f"Closest docs to '{query}':\n{tabulate(closest_docs, headers=close_docs_headers, tablefmt='mixed_grid')}"
    )
