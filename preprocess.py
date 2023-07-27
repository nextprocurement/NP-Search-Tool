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
from src.utils import load_item_list, load_vocabulary, parallelize_function, set_logger


def load_df(dir_df: Path, lang: Union[str, List[str]] = "all"):
    df = None

    df = pd.read_parquet(dir_df)
    # Choose languages
    if lang and "lang" in df.columns:
        if lang == "all":
            return df
        if isinstance(lang, str):
            lang = [lang]
        df = df[df["lang"].isin(lang)]
    df_ids = df.index
    return df, df_ids


def save_df(df: pd.DataFrame, dir_df: Path):
    dir_df.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dir_df, engine="pyarrow")


#  Process text
if __name__ == "__main__":
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
    use_dask = options.get("use_dask", False)
    subsample = options.get("subsample", None)
    pipe = options.get("pipe", [])
    merge_dfs = options.get("merge_dfs", ["minors", "insiders", "outsiders"])
    lang = options.get("lang", "all")

    # Set logger
    dir_logger = Path(options.get("dir_logger", "app.log"))
    console_log = options.get("console_log", True)
    file_log = options.get("file_log", True)
    logger = set_logger(console_log=console_log, file_log=file_log, file_loc=dir_logger)

    # Define directories
    # Set default values if not provided in the YAML file
    dir_data = Path(options.get("dir_data", "data"))
    # Files directories
    dir_text_processed = Path(
        options.get("dir_text_processed", "df_processed_pd.parquet")
    )
    # List loading options
    use_vocabulary = options.get("use_vocabulary", False)
    use_stopwords = options.get("use_stopwords", False)
    use_ngrams = options.get("use_ngrams", False)

    #################################

    # Load data
    if isinstance(merge_dfs, str):
        merge_dfs = [merge_dfs]
    if use_vocabulary:
        vocabulary = load_item_list(
            dir_data, "vocabulary", use_item_list=use_vocabulary
        )
        vocabulary = dict([w.split(",") for w in vocabulary])
    else:
        vocabulary = {}
    if use_stopwords:
        stop_words = load_item_list(dir_data, "stopwords", use_item_list=use_stopwords)
    else:
        stop_words = []
    if use_ngrams:
        # ngrams = load_item_list(Path(dir_ngrams), use_item_list=use_ngrams)
        ngrams = load_item_list(dir_data, "ngrams", use_item_list=use_ngrams)
    else:
        ngrams = []

    logger.info(f"Loaded vocabulary: {len(vocabulary)} terms.")
    logger.info(f"Loaded stopwords: {len(stop_words)} terms.")
    logger.info(f"Loaded ngrams: {len(ngrams)} terms.")

    # Load data
    if not "merge_data" in pipe:
        if any([p in ["lang_id", "normalization", "preprocess"] for p in pipe]):
            # dir_text_processed already exists and can be used
            if not dir_text_processed.exists():
                logger.error(
                    f"Error: '{dir_text_processed}' does not exist. Create it first (call 'merge_data')."
                )
                exit()
            df_processed, df_processed_ids = load_df(dir_text_processed, lang=lang)
            df_processed = df_processed[
                df_processed["text"].apply(lambda x: len(x) > 20)
            ]
            if subsample:
                if subsample > len(df_processed):
                    logger.warning(
                        f"Subsample of {subsample} is larger than population. Setting subsample to max value ({len(df_processed)} samples)."
                    )
                    subsample = len(df_processed)
                logger.info(f"Sampling {subsample} elements.")
                df_processed = df_processed.sample(n=subsample, random_state=42)
            df_processed_ids = df_processed.index

    # Number of processing steps
    n_proc = len(pipe) - 1
    # Process:
    for proc, p in enumerate(pipe):
        logger.info(f"Stage: '{p}'")
        # Merge multiple dataframes
        if p == "merge_data":
            merge_data(dir_data, dir_text_processed, merge_dfs=merge_dfs, logger=logger)
            logger.info("New dataframe saved.")
            # load info if it's not the last processing step
            if not proc == n_proc:
                df_processed, df_processed_ids_ = load_df(dir_text_processed, lang=lang)
                if subsample:
                    if subsample > len(df_processed):
                        logger.warning(
                            f"Subsample of {subsample} is larger than population. Setting subsample to max value ({len(df_processed)} samples)."
                        )
                        subsample = len(df_processed)
                    logger.info(f"Sampling {subsample} elements.")
                    df_processed = df_processed.sample(n=subsample, random_state=42)
                save_df(df_processed, dir_text_processed)
                df_processed, df_processed_ids = load_df(dir_text_processed, lang=lang)

        # Language Identification
        elif p == "lang_id":
            lang_detector = LanguageDetector(
                library="fasttext", ft_model=str(Path("models/lid.176.ftz").absolute())
            )

            if use_dask:
                aux = dd.from_pandas(df_processed[["text"]], npartitions=100)
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
                df_processed, df_processed_ids = load_df(dir_text_processed, lang=lang)

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
            step = min(1000, len(df_processed_ids))
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
                    ids = df_processed.loc[
                        df_processed["normalized_text"].isna(), "normalized_text"
                    ].index[:step]
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
                    df_processed, df_processed_ids = load_df(
                        dir_text_processed, lang=lang
                    )

            else:
                for i in t:
                    ids = df_processed.loc[
                        df_processed["normalized_text"].isna(), "normalized_text"
                    ].index[:step]
                    df_processed.loc[ids, "normalized_text"] = df_processed.loc[
                        ids, "text"
                    ].apply(preprocessor_normalizer.preprocess)
                    # Save
                    save_df(df_processed, dir_text_processed)
                    df_processed, df_processed_ids = load_df(
                        dir_text_processed, lang=lang
                    )

            # Save
            save_df(df_processed, dir_text_processed)
            logger.info("Text normalized.")
            # load info if it's not the last processing step
            if not proc == n_proc:
                df_processed, df_processed_ids = load_df(dir_text_processed, lang=lang)

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
            step = min(1000, len(df_processed_ids))
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
                    ids = df_processed.loc[
                        df_processed["preprocessed_text"].isna(), "preprocessed_text"
                    ].index[:step]
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
                    df_processed, df_processed_ids = load_df(
                        dir_text_processed, lang=lang
                    )

            else:
                workers = 5
                for i in t:
                    ids = df_processed.loc[
                        df_processed["preprocessed_text"].isna(), "preprocessed_text"
                    ].index[:step]
                    # df_processed.loc[ids, "preprocessed_text"] = df_processed.loc[
                    #     ids, "text"
                    # ].apply(preprocessor_full.preprocess)
                    df_processed.loc[ids, "preprocessed_text"] = parallelize_function(
                        preprocessor_full.preprocess,
                        df_processed.loc[ids, "text"],
                        workers=workers,
                        prefer="threads",
                    )

                    # Save
                    save_df(df_processed, dir_text_processed)
                    df_processed, df_processed_ids = load_df(
                        dir_text_processed, lang=lang
                    )

            # Save
            save_df(df_processed, dir_text_processed)
            logger.info("Text preprocessed.")
            # load info if it's not the last processing step
            if not proc == n_proc:
                df_processed, df_processed_ids = load_df(dir_text_processed, lang=lang)
        else:
            logger.warning(
                f"Invalid element: {p} not in ['merge_data', 'lang_id', 'normalization', 'preprocess']"
            )

        # logger.info("Finished")
