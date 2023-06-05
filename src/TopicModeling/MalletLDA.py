import shutil
from pathlib import Path
from subprocess import check_output
from typing import List, Union

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
    ):
        """
        Initilization Method

        Parameters
        ----------
        mallet_path: str
            Full path to mallet binary
        """

        super().__init__(model_dir, stop_words, word_min_len)
        self._mallet_path = Path(mallet_path)

        # Create sub-directories
        self._model_data_dir = self.model_dir.joinpath("model_data")
        self._model_data_dir.mkdir(parents=True, exist_ok=True)
        self._train_data_dir = self.model_dir.joinpath("train_data")
        self._train_data_dir.mkdir(parents=True, exist_ok=True)
        self._infer_data_dir = self.model_dir.joinpath("infer_data")
        self._infer_data_dir.mkdir(parents=True, exist_ok=True)

        return

    def _create_corpus_mallet(
        self, texts: List[str], name: Path, predict: bool = False
    ):
        """
        Generate .txt and .mallet files to train LDA model.
        Returns path to generated .mallet file.
        """
        # Define directories
        # texts_txt_path = name.joinpath("corpus.txt")
        # texts_mallet_path = name.joinpath("corpus.mallet")
        texts_txt_path = Path("/tmp").joinpath("corpus.txt")
        texts_mallet_path = Path("/tmp").joinpath("corpus.mallet")

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
            cmd += f"--use-pipe-from corpus.mallet"
        check_output(args=cmd, shell=True)
        self.logger.info(f"corpus.mallet created")
        texts_txt_path = shutil.move(texts_txt_path, name.joinpath("corpus.txt"))
        texts_mallet_path = shutil.move(
            texts_mallet_path, name.joinpath("corpus.mallet")
        )
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
        file_name = self._train_data_dir
        config_file = self._train_data_dir.joinpath("train.config")

        # Generate corpus.mallet
        texts_mallet_path = self._create_corpus_mallet(
            texts, name=file_name, predict=False
        )

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

        cmd = f"{self._mallet_path} train-topics --config {config_file}"
        # print(cmd)
        # print()
        check_output(args=cmd, shell=True)

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
        file_name = self._infer_data_dir
        dir_inferencer = self._model_data_dir.joinpath("inferencer.mallet")
        predicted_doctopic = self._infer_data_dir.joinpath("predicted_topics.txt")

        # Generate corpus.mallet
        texts_mallet_infer_path = self._create_corpus_mallet(
            texts, name=file_name, predict=True
        )

        # Infer topics
        cmd = (
            f"{self._mallet_path} infer-topics "
            f"--inferencer {dir_inferencer} "
            f"--input {texts_mallet_infer_path} "
            f"--output-doc-topics {predicted_doctopic} "
        )
        check_output(args=cmd, shell=True)

        # return ftext_doctopic
        return

    def get_topics_words(self, n_words=10):
        # Define directories
        topickeys_dir = self._model_data_dir.joinpath("topickeys.txt")
        # Read from generated file
        with topickeys_dir.open("r", encoding="utf8") as f:
            topic_words = [
                l.strip().split("\t")[-1].split()[:n_words] for l in f.readlines()
            ]
        return topic_words
