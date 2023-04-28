import random
from pathlib import Path
from typing import List, Union

import dask.dataframe as dd
import pandas as pd
from tqdm import trange

from src.Preprocessor.LanguageDetector import LanguageDetector
from src.Preprocessor.TextProcessor import TextPreprocessor
from src.Preprocessor.utils import merge_data
from src.Utils.utils import load_stopwords, load_vocabulary

random.seed(42)


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


#  Process text
if __name__ == "__main__":
    # Options
    use_dask = True
    subsample = 500_000
    # pipe = ["merge_data", "lang_id", "normalization", "preprocess"]
    # pipe = ["preprocess"]
    # pipe = ["lang_id", "normalization", "preprocess"]
    pipe = ["preprocess"]
    merge_dfs = ["minors", "insiders", "outsiders"]
    lang = ["es"]

    # Define directories
    dir_data = Path("data")
    dir_metadata = dir_data.joinpath("metadata")
    dir_stopwords = dir_data.joinpath("stopwords")
    dir_ngrams = dir_data.joinpath("ngrams")
    dir_vocabulary = dir_data.joinpath("RAE/vocabulary_extended.json")

    dir_text_metadata = dir_metadata.joinpath("df_text.parquet")
    dir_text_processed = dir_metadata.joinpath("df_processed_pd.parquet")

    # Load data
    stop_words = load_stopwords(dir_stopwords, use_stopwords="all")
    vocabulary = load_vocabulary(dir_vocabulary)
    ngrams = load_stopwords(dir_ngrams, use_stopwords="all")

    # Load data
    # case df_text is not (re)created
    if not "merge_data" in pipe:
        if any([p in ["lang_id", "normalization", "preprocess"] for p in pipe]):
            # dir_text_processed already exists and can be used
            if dir_text_processed.exists():
                df_processed = load_df(dir_text_processed)

                if subsample:
                    if dir_text_metadata.exists():
                        df_text = load_df(dir_text_metadata)
                        df_processed_ids = random.sample(list(df_text.index), subsample)
                    else:
                        df_processed_ids = df_processed.index
                else:
                    df_processed_ids = df_processed.index

            else:
                # df_text already exists?
                if not dir_text_metadata.exists():
                    print(
                        f"Error: '{dir_text_metadata}' does not exist. Create it first."
                    )
                    exit()
                else:
                    df_text = load_df(dir_text_metadata)
                    df_text = df_text[df_text["text"].apply(lambda x: len(x) > 20)]
                    if subsample:
                        df_processed_ids = random.sample(list(df_text.index), subsample)
                        df_processed = df_text.loc[df_text.index.isin(df_processed_ids)]
                    else:
                        df_processed_ids = df_text.index
                        df_processed = df_text

    # Number of processing steps
    n_proc = len(pipe) - 1
    # Process:
    for proc, p in enumerate(pipe):
        print(p)
        # Merge multiple dataframes
        if p == "merge_data":
            merge_data(dir_metadata, dir_text_metadata, merge_dfs=merge_dfs)
            # load info if it's not the last processing step
            if not proc == n_proc:
                df_text = load_df(dir_text_metadata)
                if subsample:
                    df_processed_ids = random.sample(list(df_text.index), subsample)
                    df_processed = df_text.loc[df_text.index.isin(df_processed_ids)]
                else:
                    df_processed_ids = df_text.index
                    df_processed = df_text

        # Language Identification
        if p == "lang_id":
            lang_detector = LanguageDetector(
                library="fasttext", ft_model=str(Path("models/lid.176.bin").absolute())
            )

            if use_dask:
                ids = df_processed.index.isin(df_processed_ids)
                aux = dd.from_pandas(df_processed.loc[ids][["text"]], npartitions=100)
                aux["lang"] = aux["text"].apply(
                    lang_detector.identify_language, meta=(None, "object")
                )
                aux = aux.compute()["lang"]
                df_processed.loc[aux.index, "lang"] = aux

            else:
                df_processed.loc[df_processed_ids, "lang"] = df_processed.loc[
                    df_processed_ids, "text"
                ].apply(lang_detector.identify_language)

            # Save
            save_df(df_processed, dir_text_processed)
            # load info if it's not the last processing step
            if not proc == n_proc:
                df_processed = load_df(dir_text_processed)

        # Text normalization
        if p == "normalization":
            preprocessor_normalizer = TextPreprocessor(
                methods=[
                    "lowercase",
                    "remove_urls",
                    ("clean_text", {"min_len": 1}),
                    "convert_ngrams",
                ],
                ngrams=ngrams,
            )

            if use_dask:
                ids = df_processed.index.isin(df_processed_ids)
                aux = dd.from_pandas(df_processed.loc[ids][["text"]], npartitions=100)
                aux["normalized_text"] = aux["text"].apply(
                    preprocessor_normalizer.preprocess, meta=(None, "object")
                )
                aux = aux.compute()["normalized_text"]
                df_processed.loc[aux.index, "normalized_text"] = aux

            else:
                df_processed.loc[
                    df_processed_ids, "normalized_text"
                ] = df_processed.loc[df_processed_ids, "text"].apply(
                    preprocessor_normalizer.preprocess
                )

            # Save
            save_df(df_processed, dir_text_processed)
            # load info if it's not the last processing step
            if not proc == n_proc:
                df_processed = load_df(dir_text_processed)

        # Full process text
        if p == "preprocess":
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
            )

            # Compute and save iteratively
            step = 10_000
            indices = range(len(df_processed_ids))

            # Skip columns already processed
            skip = 0
            if "preprocessed_text" in df_processed.columns:
                skip = len(df_processed) - len(
                    df_processed["preprocessed_text"].dropna().index
                )
            else:
                df_processed["preprocessed_text"] = None
            t = trange(0, skip, step, desc="", leave=True)
            if use_dask:
                for i in t:
                    # t.set_description(f"")
                    # t.refresh()
                    # ids = df_processed.index.isin(df_processed_ids[i : i + step])
                    ids = df_processed.loc[
                        df_processed["preprocessed_text"].isna(), "preprocessed_text"
                    ].index[i : i + step]
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
            # load info if it's not the last processing step
            if not proc == n_proc:
                df_processed = load_df(dir_text_processed)

        print("- finished")


# # Test texts
# text = "Hola, mi nombre s /202 es Juana3w29, de Castilla-La Mancha-999iu. ca-g-k-8--l1 fs-12sas -ejemplo C.I.N.E. km/h otro- Aquí laboratorio añado algùn que otro 329854. Yo 8923jk3292 ( (34) 9348hnkj2j398421h9 vi\ve en Mancha-iu993 Madriz y 9876trabajo 543como profecional en el -201 ámbito de la marketin. Me (gusta) mucho viajar, conozer nuebas u-235 u235 utf8 utf-8 culturas y aprendel cosas nuebas. También me encanta leer, ver pelis y hacer deporte. Mi favorito es el tenis y mi jugadora preferida es Rafal Nadal. Espero conocel gente nueva y hacer amigos por íaquí."
# # text = "Hi mi name is Juana. Here I add some 329854. I lives in Madriz and work as..."
# text = """
# P-2023-78-asdasdf-12-23
# REF-234-123
# REF-2334-123
# C6H12O6

# Hola, mi nombre s /202 es Juana3w29, de Castilla-La Mancha-999iu. ca-g-k-8--l1 fs-12sas -ejemplo C.I.N.E. km/h otro- Aquí laboratorio añado algùn que otro 329854. Yo 8923jk3292 ( (34) 9348hnkj2j398421h9 vi\ve en Mancha-iu993 Madriz y 9876trabajo 543como profecional en el -201 ámbito de la marketin. Me (gusta) mucho viajar, conozer nuebas u-235 u235 utf8 utf-8 culturas y aprendel cosas nuebas. También me encanta leer, ver pelis y hacer deporte. Mi favorito es el tenis y mi jugadora preferida es Rafal Nadal. Espero conocel gente nueva y hacer amigos por íaquí.
# """
