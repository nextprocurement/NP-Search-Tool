import logging
import shutil
from pathlib import Path
from subprocess import check_output
from typing import List, Union

import numpy as np
from tqdm import trange

from .BaseModel import BaseModel


class MalletLDAModel(BaseModel):
    """
    Wrapper for the Mallet Topic Model Training. Implements the
    following functionalities
    - Import of the corpus to the mallet internal training format
    - Training of the model
    - Creation and persistence of the TMmodel object for tm curation
    - Execution of some other time consuming tasks (pyLDAvis, ..)

    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        stop_words: list = [],
        word_min_len: int = 2,
        mallet_path="/app/Mallet/bin/mallet",
        logger: logging.Logger = None,
    ):
        """
        Initilization Method

        Parameters
        ----------
        mallet_path: str
            Full path to mallet binary
        """

        super().__init__(model_dir, stop_words, word_min_len, logger)
        self._mallet_path = Path(mallet_path)

        return

    def _create_corpus_mallet(self, texts: List[str], predict: bool = False):
        """
        Generate .txt and .mallet files to train LDA model.
        Returns path to generated .mallet file.
        """
        # Define directories
        # texts_txt_path = name.joinpath("corpus.txt")
        # texts_mallet_path = name.joinpath("corpus.mallet")
        if predict:
            texts_txt_path = self._temp_dir.joinpath("corpus_predict.txt")
            texts_mallet_path = self._temp_dir.joinpath("corpus_predict.mallet")
        else:
            texts_txt_path = self._temp_dir.joinpath("corpus_train.txt")
            texts_mallet_path = self._temp_dir.joinpath("corpus_train.mallet")

        # Create corpus.txt
        self.logger.info("Creating corpus.txt...")
        with texts_txt_path.open("w", encoding="utf8") as fout:
            for i, t in enumerate(texts):
                fout.write(f"{i} 0 {t}\n")
        self.logger.info(f"corpus.txt created")

        # Convert corpus.txt to corpus.mallet
        self.logger.info("Creating corpus.mallet...")
        cmd = (
            f"{self._mallet_path} import-file "
            f"--input {texts_txt_path} "
            f"--output {texts_mallet_path} "
            f"--keep-sequence "
            # f"--remove-stopwords "
        )
        if predict:
            cmd += f"--use-pipe-from {self._temp_dir.joinpath('corpus_train.mallet')}"
        check_output(args=cmd, shell=True)
        self.logger.info(f"corpus.mallet created")

        # Move info to desired location
        if predict:
            name = self._infer_data_dir
        else:
            name = self._train_data_dir
        texts_txt_path = shutil.copy(texts_txt_path, name.joinpath("corpus.txt"))
        texts_mallet_path = shutil.copy(
            texts_mallet_path, name.joinpath("corpus.mallet")
        )
        # texts_txt_path = shutil.move(texts_txt_path, name.joinpath("corpus.txt"))
        # texts_mallet_path = shutil.move(texts_mallet_path, name.joinpath("corpus.mallet"))
        # texts_txt_path = name.joinpath("corpus.txt")
        # texts_mallet_path = name.joinpath("corpus.mallet")
        return Path(texts_mallet_path)

    def train(
        self,
        texts: List[str],
        num_topics: int,
        alpha: float = 5.0,
        optimize_interval: int = 10,
        num_threads: int = 4,
        num_iterations: int = 1000,
        doc_topic_thr: float = 0.0,
    ):
        """
        Train LDA model

        Parameters:
        -----------
        texts: list(str)
            List of texts to train on.
        alpha: float
            Parameter for the Dirichlet prior on doc distribution
        optimize_interval: int
            Number of steps betweeen parameter reestimation
        num_threads: int
            Number of threads for the optimization
        num_iterations: int
            Number of iterations for the mallet training
        doc_topic_thr: float
            Min value for topic proportions during mallet training
        thetas_thr: float
            Min value for sparsification of topic proportions after training
        """
        # Set num topics
        self.num_topics = num_topics

        # Define directories
        config_file = self._train_data_dir.joinpath("train.config")

        # Generate corpus.mallet
        texts_mallet_path = self._create_corpus_mallet(texts, predict=False)

        # Generate train config
        with config_file.open("w", encoding="utf8") as fout:
            fout.write(f"input = {texts_mallet_path}\n")
            fout.write(f"num-topics = {self.num_topics}\n")
            fout.write(f"alpha = {alpha}\n")
            fout.write(f"optimize-interval = {optimize_interval}\n")
            fout.write(f"num-threads = {num_threads}\n")
            fout.write(f"num-iterations = {num_iterations}\n")
            fout.write(f"doc-topics-threshold = {doc_topic_thr}\n")

            fout.write(
                f"output-doc-topics = {self._model_data_dir.joinpath('doc-topics.txt')}\n"
            )
            fout.write(
                f"word-topic-counts-file = {self._model_data_dir.joinpath('word-topic-counts.txt')}\n"
            )
            fout.write(
                f"diagnostics-file = {self._model_data_dir.joinpath('diagnostics.xml')}\n"
            )
            fout.write(
                f"xml-topic-report = {self._model_data_dir.joinpath('topic-report.xml')}\n"
            )
            fout.write(
                f"output-topic-keys = {self._model_data_dir.joinpath('topickeys.txt')}\n"
            )
            fout.write(
                f"inferencer-filename = {self._model_data_dir.joinpath('inferencer.mallet')}\n"
            )

            # fout.write(f"topic-word-weights-file = {self._model_data_dir.joinpath('word-weights.txt')}\n")
            # fout.write(f"output-topic-docs = {self._model_data_dir.joinpath('topic-docs.txt')}\n")

        cmd = f"{self._mallet_path} train-topics "
        # cmd += f"--config {config_file}"
        with open("_test_models/Mallet/train_data/train.config", "r") as f:
            cmd += " ".join([f"--{l.strip()}" for l in f.readlines()]).replace("=", "")
        # print(cmd)
        # print()
        check_output(args=cmd, shell=True)
        self.logger.info("Finished training")

        return

    def predict(self, texts: List[str]):
        """
        Infer doctopic file of corpus using ntopics model

        Parameters:
        -----------
        corpus: list(str)
            Preprocessed corpus as a list of strings

        """
        # Define directories
        dir_inferencer = self._model_data_dir.joinpath("inferencer.mallet")
        predicted_doctopic = self._infer_data_dir.joinpath("predicted_topics.txt")

        # Generate corpus.mallet
        texts_mallet_infer_path = self._create_corpus_mallet(texts, predict=True)

        self.logger.info("Infer topics")
        # Infer topics
        cmd = (
            f"{self._mallet_path} infer-topics "
            f"--inferencer {dir_inferencer} "
            f"--input {texts_mallet_infer_path} "
            f"--output-doc-topics {self._temp_dir.joinpath('predicted_topics.txt')} "
        )
        check_output(args=cmd, shell=True)
        shutil.copy(
            f"{self._temp_dir.joinpath('predicted_topics.txt')}", predicted_doctopic
        )

        pred = np.loadtxt(predicted_doctopic, usecols=range(2, self.num_topics + 2))
        return pred

    def get_topics_words(self, n_words=10):
        # Define directories
        topickeys_dir = self._model_data_dir.joinpath("topickeys.txt")
        # Read from generated file
        with topickeys_dir.open("r", encoding="utf8") as f:
            topic_words = [
                l.strip().split("\t")[-1].split()[:n_words] for l in f.readlines()
            ]
        return topic_words
