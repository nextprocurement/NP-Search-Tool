import argparse
import logging
import random
from pathlib import Path
from typing import List, Union

import dask.dataframe as dd
import pandas as pd
import yaml
from tqdm import trange

from src.Preprocessor.LanguageDetector import LanguageDetector
from src.Preprocessor.TextProcessor import TextPreprocessor
from src.Preprocessor.utils import merge_data
from src.utils import load_item_list, load_vocabulary


def load_df(dir_df: Path, lang: Union[str, List[str]] = "all"):
    df = None

    df = pd.read_parquet(dir_df)
    # Choose languages
    if lang and "lang" in df.columns:
        if lang == "all":
            return df
        lang = list(lang)
        df = df[df["lang"].isin(lang)]

    return df


def save_df(df: pd.DataFrame, dir_df: Path):
    df.to_parquet(dir_df, engine="pyarrow")


def set_logger(console_log=True, file_log=True):
    # Set up the logger
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create console handler
    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Create file handler
    if file_log:
        file_handler = logging.FileHandler("app.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # logger.info("Log created.")
    return logger


#  Process text
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

    # Access options
    use_dask = options["use_dask"]
    subsample = options["subsample"]
    pipe = options["pipe"]
    merge_dfs = options["merge_dfs"]
    lang = options["lang"]

    # Define directories
    # Set default values if not provided in the YAML file
    dir_data = Path(options.get("dir_data", "data"))
    dir_metadata = Path(options.get("dir_metadata", f"{dir_data}/metadata"))
    dir_stopwords = Path(options.get("dir_stopwords", f"{dir_data}/stopwords"))
    dir_ngrams = Path(options.get("dir_ngrams", f"{dir_data}/ngrams"))
    # Files directories
    dir_vocabulary = Path(
        options.get("dir_vocabulary", f"{dir_data}/RAE/vocabulary_extended.json")
    )
    dir_text_metadata = Path(
        options.get("dir_text_metadata", f"{dir_metadata}/df_text.parquet")
    )
    dir_text_processed = Path(
        options.get("dir_text_processed", f"{dir_metadata}/df_processed_pd.parquet")
    )
    # List loading options
    use_stopwords = options.get("use_stopwords", "all")
    use_ngrams = options.get("use_stopwords", "all")

    # Load data
    stop_words = load_item_list(dir_stopwords, use_item_list=use_stopwords)
    ngrams = load_item_list(dir_ngrams, use_item_list=use_ngrams)
    vocabulary = load_vocabulary(dir_vocabulary)

    # Load data
    # case df_text is not (re)created
    if not "merge_data" in pipe:
        if any([p in ["lang_id", "normalization", "preprocess"] for p in pipe]):
            # dir_text_processed already exists and can be used
            if dir_text_processed.exists():
                df_processed = load_df(dir_text_processed)
            else:
                # df_text already exists?
                if not dir_text_metadata.exists():
                    print(
                        f"Error: '{dir_text_metadata}' does not exist. Create it first (call 'merge_data')."
                    )
                    exit()
                else:
                    df_text = load_df(dir_text_metadata)
                    df_text = df_text[df_text["text"].apply(lambda x: len(x) > 20)]
                    if subsample:
                        if subsample > len(df_text):
                            print(
                                f"Subsample of {subsample} is larger than population. Setting subsample to max value."
                            )
                            subsample = len(df_text)
                        df_processed = df_text.sample(n=subsample, random_state=42)
                    else:
                        df_processed = df_text
            df_processed_ids = df_processed.index

    # Number of processing steps
    n_proc = len(pipe) - 1
    # Process:
    for proc, p in enumerate(pipe):
        logger.info(f"Stage: '{p}'")
        # Merge multiple dataframes
        if p == "merge_data":
            merge_data(dir_metadata, dir_text_metadata, merge_dfs=merge_dfs)
            logger.info("New dataframe saved.")
            # load info if it's not the last processing step
            if not proc == n_proc:
                df_text = load_df(dir_text_metadata)
                if subsample:
                    if subsample > len(df_text):
                        logger.warning(
                            f"Subsample of {subsample} is larger than population. Setting subsample to max value."
                        )
                        subsample = len(df_text)
                    df_processed_ids = random.sample(list(df_text.index), subsample)
                    df_processed = df_text.loc[df_text.index.isin(df_processed_ids)]
                else:
                    df_processed_ids = df_text.index
                    df_processed = df_text
                save_df(df_processed, dir_text_processed)
                df_processed = load_df(dir_text_processed)

        # Language Identification
        elif p == "lang_id":
            lang_detector = LanguageDetector(
                library="fasttext", ft_model=str(Path("models/lid.176.ftz").absolute())
            )

            if use_dask:
                ids = df_processed.index.isin(df_processed_ids)
                aux = dd.from_pandas(df_processed.loc[ids][["text"]], npartitions=100)
                aux["lang"] = aux["text"].apply(
                    lang_detector.identify_language, meta=(None, "object")
                )
                aux = aux.compute()
                df_processed.loc[aux.index, "lang"] = aux["lang"]

            else:
                df_processed.loc[df_processed_ids, "lang"] = df_processed.loc[
                    df_processed_ids, "text"
                ].apply(lang_detector.identify_language)

            # Save
            save_df(df_processed, dir_text_processed)
            logger.info("Language identified.")
            # load info if it's not the last processing step
            if not proc == n_proc:
                df_processed = load_df(dir_text_processed)

        # Text normalization
        elif p == "normalization":
            preprocessor_normalizer = TextPreprocessor(
                methods=[
                    "lowercase",
                    "remove_urls",
                    ("clean_text", {"min_len": 1}),
                    # "convert_ngrams",
                ],
                # ngrams=ngrams,
                logger=logger,
            )

            # Compute and save iteratively
            step = 1000
            indices = range(len(df_processed_ids))

            # Skip columns already processed
            skip = 0
            if "normalized_text" in df_processed.columns:
                skip = df_processed["normalized_text"].dropna().size
            else:
                df_processed["normalized_text"] = None
            t = trange(skip, len(df_processed), step, desc="", leave=True)

            if use_dask:
                for i in t:
                    ids = df_processed.index.isin(df_processed_ids[i : i + step])
                    aux = dd.from_pandas(
                        df_processed.loc[ids][["text"]], npartitions=100
                    )
                    aux["normalized_text"] = aux["text"].apply(
                        preprocessor_normalizer.preprocess, meta=(None, "object")
                    )
                    aux = aux.compute()["normalized_text"]
                    df_processed.loc[aux.index, "normalized_text"] = aux
                    # Save
                    save_df(df_processed, dir_text_processed)
                    df_processed = load_df(dir_text_processed)

            else:
                for i in t:
                    df_processed.loc[
                        df_processed_ids[i : i + step], "normalized_text"
                    ] = df_processed.loc[df_processed_ids[i : i + step], "text"].apply(
                        preprocessor_normalizer.preprocess
                    )
                    # Save
                    save_df(df_processed, dir_text_processed)
                    df_processed = load_df(dir_text_processed)

            # Save
            save_df(df_processed, dir_text_processed)
            logger.info("Text normalized.")
            # load info if it's not the last processing step
            if not proc == n_proc:
                df_processed = load_df(dir_text_processed)

        # Full process text
        elif p == "preprocess":
            preprocessor_full = TextPreprocessor(
                methods=[
                    "lowercase",
                    "remove_urls",
                    "lemmatize_text",
                    ("clean_text", {"min_len": 1}),
                    "convert_ngrams",
                    ("clean_text", {"min_len": 2}),
                    "remove_stopwords",
                    # "tokenize_text",
                ],
                stopwords=stop_words,
                vocabulary=vocabulary,
                ngrams=ngrams,
                logger=logger,
            )

            # Compute and save iteratively
            step = 1000
            indices = range(len(df_processed_ids))

            # Skip columns already processed
            skip = 0
            if "preprocessed_text" in df_processed.columns:
                skip = df_processed["preprocessed_text"].dropna().size
            else:
                df_processed["preprocessed_text"] = None
            t = trange(skip, len(df_processed), step, desc="", leave=True)
            if use_dask:
                for i in t:
                    # t.set_description(f"")
                    # t.refresh()
                    ids = df_processed.index.isin(df_processed_ids[i : i + step])
                    # ids = df_processed.loc[
                    #     df_processed["preprocessed_text"].isna(), "preprocessed_text"
                    # ].index[i : i + step]
                    aux = dd.from_pandas(
                        df_processed.loc[ids][["text"]], npartitions=100
                    )
                    aux["preprocessed_text"] = aux["text"].apply(
                        preprocessor_full.preprocess, meta=(None, "object")
                    )
                    aux = aux.compute()["preprocessed_text"]
                    df_processed.loc[aux.index, "preprocessed_text"] = aux

                    # Save
                    save_df(df_processed, dir_text_processed)
                    df_processed = load_df(dir_text_processed)

            else:
                for i in t:
                    # t.set_description(f"")
                    # t.refresh()
                    # ids = df_processed_ids[i : i + step]
                    ids = df_processed.loc[
                        df_processed["preprocessed_text"].isna(), "preprocessed_text"
                    ].index[i : i + step]
                    df_processed.loc[ids, "preprocessed_text"] = df_processed.loc[
                        ids, "text"
                    ].apply(preprocessor_full.preprocess)

                    # Save
                    save_df(df_processed, dir_text_processed)
                    df_processed = load_df(dir_text_processed)

            # Save
            save_df(df_processed, dir_text_processed)
            logger.info("Text preprocessed.")
            # load info if it's not the last processing step
            if not proc == n_proc:
                df_processed = load_df(dir_text_processed)
        else:
            logger.warning(
                f"Invalid element: {p} not in ['merge_data', 'lang_id', 'normalization', 'preprocess']"
            )

        # logger.info("Finished")
