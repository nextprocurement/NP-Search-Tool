import logging
import shutil
from pathlib import Path
from subprocess import check_output
from typing import List, Union

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from tqdm import trange

from .BaseModel import BaseModel
from src.TopicModeling.tm_utils.tm_model import TMmodel
from src.TopicModeling.tm_utils.utils import file_lines


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

        # Create temp dir for model data (improves speed)
        self._temp_mallet_dir = self._temp_dir.joinpath("mallet")
        self._temp_mallet_dir.mkdir(parents=True, exist_ok=True)

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
            texts_txt_path = self._temp_mallet_dir.joinpath(
                "corpus_predict.txt")
            texts_mallet_path = self._temp_mallet_dir.joinpath(
                "corpus_predict.mallet")
        else:
            texts_txt_path = self._temp_mallet_dir.joinpath("corpus_train.txt")
            texts_mallet_path = self._temp_mallet_dir.joinpath(
                "corpus_train.mallet")

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
            cmd += f"--use-pipe-from {self._temp_mallet_dir.joinpath('corpus_train.mallet')}"
        check_output(args=cmd, shell=True)
        self.logger.info(f"corpus.mallet created")

        # Move info to desired location
        if predict:
            name = self._infer_data_dir
        else:
            name = self._train_data_dir
        texts_txt_path = shutil.copy(
            texts_txt_path, name.joinpath("corpus.txt"))
        texts_mallet_path = shutil.copy(
            texts_mallet_path, name.joinpath("corpus.mallet")
        )
        # texts_txt_path = shutil.move(texts_txt_path, name.joinpath("corpus.txt"))
        # texts_mallet_path = shutil.move(texts_mallet_path, name.joinpath("corpus.mallet"))
        # texts_txt_path = name.joinpath("corpus.txt")
        # texts_mallet_path = name.joinpath("corpus.mallet")
        return Path(texts_mallet_path)

    def _model_train(
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
                f"output-doc-topics = {self._temp_mallet_dir.joinpath('doc-topics.txt')}\n"
            )
            fout.write(
                f"word-topic-counts-file = {self._temp_mallet_dir.joinpath('word-topic-counts.txt')}\n"
            )
            fout.write(
                f"diagnostics-file = {self._temp_mallet_dir.joinpath('diagnostics.xml')}\n"
            )
            fout.write(
                f"xml-topic-report = {self._temp_mallet_dir.joinpath('topic-report.xml')}\n"
            )
            fout.write(
                f"output-topic-keys = {self._temp_mallet_dir.joinpath('topic-keys.txt')}\n"
            )
            fout.write(
                f"inferencer-filename = {self._temp_mallet_dir.joinpath('inferencer.mallet')}\n"
            )
            # fout.write(f"topic-word-weights-file = {self._model_data_dir.joinpath('word-weights.txt')}\n")
            # fout.write(f"output-topic-docs = {self._model_data_dir.joinpath('topic-docs.txt')}\n")

        cmd = f"{self._mallet_path} train-topics "
        with config_file.open("r") as f:
            cmd += " ".join([f"--{l.strip()}" for l in f.readlines()]
                            ).replace("=", "")
        check_output(args=cmd, shell=True)
        self.logger.info("Finished training")

        shutil.copy(
            self._temp_mallet_dir.joinpath("doc-topics.txt"),
            self._model_data_dir.joinpath("doc-topics.txt"),
        )
        shutil.copy(
            self._temp_mallet_dir.joinpath("word-topic-counts.txt"),
            self._model_data_dir.joinpath("word-topic-counts.txt"),
        )
        shutil.copy(
            self._temp_mallet_dir.joinpath("diagnostics.xml"),
            self._model_data_dir.joinpath("diagnostics.xml"),
        )
        shutil.copy(
            self._temp_mallet_dir.joinpath("topic-report.xml"),
            self._model_data_dir.joinpath("topic-report.xml"),
        )
        shutil.copy(
            self._temp_mallet_dir.joinpath("topic-keys.txt"),
            self._model_data_dir.joinpath("topic-keys.txt"),
        )
        shutil.copy(
            self._temp_mallet_dir.joinpath("inferencer.mallet"),
            self._model_data_dir.joinpath("inferencer.mallet"),
        )

        pred = np.loadtxt(
            self._model_data_dir.joinpath("doc-topics.txt"),
            usecols=range(2, self.num_topics + 2),
        )
        with self._model_data_dir.joinpath("topic-keys.txt").open(
            "r", encoding="utf8"
        ) as f:
            topic_keys = [t.strip().split("\t") for t in f.readlines()]
        topic_keys = {t[0]: t[-1] for t in topic_keys}
        # topic_keys = np.loadtxt(self._model_data_dir.joinpath('topic-keys.txt'), usecols=range(2, self.num_topics + 2))
        
        # Create TMmodel
        self._createTMmodel()

        return pred, topic_keys

    def _model_predict(self, texts: List[str]):
        """
        Infer doctopic file of corpus using ntopics model

        Parameters:
        -----------
        corpus: list(str)
            Preprocessed corpus as a list of strings

        """
        # Define directories
        predicted_doctopic = self._infer_data_dir.joinpath(
            "predicted_topics.txt")
        # Get inferencer
        if self._temp_mallet_dir.joinpath("inferencer.mallet").exists():
            dir_inferencer = self._temp_mallet_dir.joinpath(
                "inferencer.mallet")
        elif self._model_data_dir.joinpath("inferencer.mallet").exists():
            dir_inferencer = self._model_data_dir.joinpath("inferencer.mallet")
        else:
            self.logger.warning("There is no inferencer. Train model first.")
            return

        # Generate corpus.mallet
        texts_mallet_infer_path = self._create_corpus_mallet(
            texts, predict=True)

        self.logger.info("Infer topics")
        # Infer topics
        cmd = (
            f"{self._mallet_path} infer-topics "
            f"--inferencer {dir_inferencer} "
            f"--input {texts_mallet_infer_path} "
            f"--output-doc-topics {self._temp_mallet_dir.joinpath('predicted_topics.txt')} "
        )
        check_output(args=cmd, shell=True)
        shutil.copy(
            f"{self._temp_mallet_dir.joinpath('predicted_topics.txt')}",
            predicted_doctopic,
        )

        pred = np.loadtxt(predicted_doctopic,
                          usecols=range(2, self.num_topics + 2))
        return pred

    def _createTMmodel(self):

        # Load thetas matrix for sparsification
        thetas_file = self._model_data_dir.joinpath("doc-topics.txt")
        cols = [k for k in np.arange(2, self.num_topics + 2)]
        thetas32 = np.loadtxt(thetas_file, delimiter='\t',
                              dtype=np.float32, usecols=cols)

        # Create figure to check thresholding is correct
        self._SaveThrFig(
            thetas32, self._model_data_dir.joinpath('thetasDist.pdf'))

        # Set to zeros all thetas below threshold, and renormalize
        thetas32[thetas32 < self._thetas_thr] = 0
        thetas32 = normalize(thetas32, axis=1, norm='l1')
        thetas32 = sparse.csr_matrix(thetas32, copy=True)

        # Recalculate topic weights to avoid errors due to sparsification
        alphas = np.asarray(np.mean(thetas32, axis=0)).ravel()

        # Create vocabulary files and calculate beta matrix
        # A vocabulary is available with words provided by the Count Vectorizer object, but the new files need the order used by mallet
        wtcFile = self._model_data_dir.joinpath('word-topic-counts.txt')
        vocab_size = file_lines(wtcFile)
        betas = np.zeros((self.num_topics, vocab_size))
        vocab = []
        term_freq = np.zeros((vocab_size,))
        with wtcFile.open('r', encoding='utf8') as fin:
            for i, line in enumerate(fin):
                elements = line.split()
                vocab.append(elements[1])
                for counts in elements[2:]:
                    tpc = int(counts.split(':')[0])
                    cnt = int(counts.split(':')[1])
                    betas[tpc, i] += cnt
                    term_freq[i] += cnt
        betas = normalize(betas, axis=1, norm='l1')

        # Save vocabulary and frequencies
        vocabfreq_file = self._model_data_dir.joinpath('vocab_freq.txt')
        with vocabfreq_file.open('w', encoding='utf8') as fout:
            [fout.write(el[0] + '\t' + str(int(el[1])) + '\n')
             for el in zip(vocab, term_freq)]

        tm = TMmodel(TMfolder=self._model_data_dir.joinpath('TMmodel'))
        tm.create(betas=betas, thetas=thetas32, alphas=alphas,
                  vocab=vocab)

        # Remove doc-topics file
        thetas_file.unlink()

        return tm
