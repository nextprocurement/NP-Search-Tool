import re
import os
import pathlib
from src.TopicModeling.solr_backend_utils.utils import create_trainconfig
from src.utils import load_item_list, set_logger, train_test_split
import argparse
import yaml
from pathlib import Path
import pandas as pd
import sys
import time
from src.TopicModeling import (
    BERTopicModel,
    GensimLDAModel,
    MalletLDAModel,
    NMFModel,
    TomotopyCTModel,
    TomotopyLDAModel,
)

########################
# AUXILIARY FUNCTIONS  #
########################


def create_model(model_name, **kwargs):
    # Map model names to corresponding classes
    model_mapping = {
        'BERTtopic': BERTopicModel,
        'Gensim': GensimLDAModel,
        'Mallet': MalletLDAModel,
        'NMF': NMFModel,
        'TomotopyCTModel': TomotopyCTModel,
        'TomotopyLDAModel': TomotopyLDAModel,
    }

    # Retrieve the class based on the model name
    model_class = model_mapping.get(model_name)

    # Check if the model name is valid
    if model_class is None:
        raise ValueError(f"Invalid model name: {model_name}")

    # Create an instance of the model class
    model_instance = model_class(**kwargs)

    return model_instance


if __name__ == "__main__":

    # Set logger
    logger = set_logger(console_log=True, file_log=True)

    # Parse args
    parser = argparse.ArgumentParser(
        description="Train options for topic modeling")
    parser.add_argument(
        "--options",
        default="config/options_env.yaml",
        help="Path to options YAML file"
    )
    parser.add_argument(
        "--trainer",
        default="Mallet",
        help="Trainer to use for topic modeling"
    )
    parser.add_argument(
        "--test_split",
        default=0.0,
        type=float,
        help="Size of test split for training the model. If set to 0.0, the model will be trained on the entire dataset and no test set will be created."
    )
    args = parser.parse_args()

    with open(args.options, "r") as f:
        options = dict(yaml.safe_load(f))

    # Access options
    merge_dfs = options.get("merge_dfs", ["minors", "insiders", "outsiders"])

    # Set logger
    dir_logger = Path(options.get("dir_logger", "app.log"))
    console_log = options.get("console_log", True)
    file_log = options.get("file_log", True)
    logger = set_logger(
        console_log=console_log,
        file_log=file_log,
        file_loc=dir_logger
    )

    ##########
    # Config #
    ##########

    # Number of topics for training the model and word min len
    num_topics = options.get('training_params', {}).get('num_topics', '50')
    try:
        num_topics = [int(k) for k in num_topics.split(",")]
    except:
        num_topics = [int(num_topics)]
    word_min_len = options.get('training_params', {}).get('word_min_len', 4)

    # File directories
    dir_data = Path(options.get("dir_data", "data"))
    dir_text_processed = Path(
        options.get("dir_text_processed", "df_processed_pd.parquet")) /\
        ("_".join(merge_dfs) + ".parquet")
    dir_logical = Path(
        options.get("dir_logical", "data/logical_dtsets")) /\
        (dir_text_processed.stem + ".json")
    dir_output_models = Path(options.get("dir_output_models", "output_models"))
    dir_mallet = Path(options.get("dir_mallet"))

    # List loading options
    use_stopwords = options.get("use_stopwords", False)

    #############
    # Load data #
    #############
    if use_stopwords:
        stop_words = load_item_list(
            dir_data, "stopwords", use_item_list=use_stopwords)
    else:
        stop_words = []

    ##########
    # Train  #
    ##########
    # Get training parameters
    tr_params = options.get('training_params', {}).get(
        f"{args.trainer}_params", {})
    if not bool(tr_params):
        logger.error(
            "--- Training parameters not found in options.yaml. Quitting script....")
        sys.exit(1)
    model_init_params = {
        "word_min_len": word_min_len,
        "stop_words": stop_words,
        "logger": logger,
    }
    if args.trainer == "Mallet":
        model_init_params["mallet_path"] = dir_mallet

    ################
    # Paths to data
    ################
    path_parquets = pathlib.Path(
        "/export/usuarios_ml4ds/cggamella/NP-Search-Tool/sample_data/all_processed")
    path_place_without_lote = path_parquets / \
        "minors_insiders_outsiders_origen_sin_lot_info.parquet"
    path_place_esp = path_parquets / "df_esp_langid.parquet"
    path_manual_stops = "/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/stopwords_sin_duplicados"
    path_eq = "/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/eq.txt"

    ################
    # Read data
    ################
    print(f"-- -- Reading data from {path_place_esp} and {path_place_without_lote}")
    processed = pd.read_parquet(path_place_esp)
    cols = processed.columns.values.tolist()
    print(f"-- -- Data read from {path_place_esp}: {len(processed)} rows.")
    # set identifier as column so we dont loose it
    processed['identifier'] = processed.index
    print(f"-- -- Columns: {cols}")
    place_without_lote = pd.read_parquet(path_place_without_lote)
    print(f"-- -- Data read from {path_place_without_lote}: {len(place_without_lote)} rows.")
    
    #########################
    # Get additional metadata
    #########################
    # Merge 'processed' with 'place_without_lote' to get info about the source of the tender (minors, insiders, outsiders)
    processed = pd.merge(processed, place_without_lote, how='left', on='id_tm')
    processed.set_index('identifier', inplace=True)  # Preserved index
    processed = processed[cols + ["origen"]]  #  Keep necessary columns
    print(f"-- -- Data merged: {len(processed)} rows.")
    print(f"-- -- Sample: {processed.head()}")
    
    #####################
    # Filter stops /eqs #
    #####################
    # Filter stops
    stopwords = set()
    # Lista para registrar los nombres de los archivos procesados
    archivos_procesados = []
    # Iterar sobre cada archivo en el directorio especificado
    for archivo in os.listdir(path_manual_stops):
        if archivo.endswith('.txt'):
            ruta_completa = os.path.join(path_manual_stops, archivo)
            with open(ruta_completa, 'r', encoding='utf-8') as f:
                stopwords.update(f.read().splitlines())
            # Registrar el archivo procesado
            archivos_procesados.append(archivo)

    def eliminar_stopwords(fila):
        return ' '.join([palabra for palabra in fila.split() if palabra not in stopwords])
    start = time.time()
    processed['lemmas'] = processed['lemmas'].apply(eliminar_stopwords)
    print(f"-- -- Stops filtered in {time.time() - start}")

    # Filter eqs
    start = time.time()
    pares_diccionario = {}
    compiled_regexes = {}
    with open(path_eq, 'r') as archivo:
        for linea in archivo:
            linea = linea.strip()
            palabras = linea.split(':')
            if len(palabras) < 2:
                print(f"Línea omitida o incompleta: '{linea}'")
                continue
            pares_diccionario[palabras[0]] = palabras[1]
    pares_diccionario = \
        dict(sorted(pares_diccionario.items(), key=lambda x: x[0]))
    print("-- -- Eq dict constructed in :", time.time() - start)

    def replace_keywords(lst, keyword_dict):
        return " ".join([keyword_dict.get(word, word) for word in lst])

    start = time.time()
    processed["lemmas_split"] = processed['lemmas'].apply(lambda x: x.split())
    processed['lemmas'] = processed['lemmas_split'].apply(
        lambda x: replace_keywords(x, pares_diccionario))
    processed = processed.drop(columns=['lemmas_split'])
    print("-- -- Eq substituted in:", time.time() - start)

    ############################
    # Filter by lemmas min len #
    ############################
    min_lemmas = 2
    processed['len'] = processed['lemmas'].apply(lambda x: len(x.split()))
    processed = processed[processed['len'] > min_lemmas]

    ############################
    # Generate training dfs
    ############################
    only_all = False
    dfs_to_train = []  # List to store dataframes to train as tuples (name, df)
    all = processed.copy()
    #dfs_to_train.append(("all", all))
    if not only_all:
        # Generate only minors, insiders and outsiders
        minors = all[all.origen == "minors"]
        outsiders = all[all.origen == "outsiders"]
        insiders = all[all.origen == "insiders"]
        #dfs_to_train.append(("all", all))
        #dfs_to_train.append(("minors", minors))
        #dfs_to_train.append(("outsiders", outsiders))
        dfs_to_train.append(("insiders", insiders))
    
    # Train models
    lang = "es"
    models_train = []
    for k in num_topics:
        logger.info(f"{'='*10} Training models with {k} topics. {'='*10}")
        for name, df in dfs_to_train:
            logger.info(
                f"{'='*5} Training {args.trainer} model with {k} topics on {name} {'='*5}")

            # Store df in training parquet file
            logger.info(f"-- -- Number of documents: {len(df)}")

            texts_train, texts_test = train_test_split(df, args.test_split)
            logger.info("Data loaded.")
            logger.info(
                f"Train: {len(texts_train)} docs. Test: {len(texts_test)}.")

            # Update model parameters
            model_init_params["model_dir"] = (dir_output_models.joinpath(
                args.trainer)).joinpath(f"{lang}_{args.trainer}_{name}_{k}_topics_FINAL")

            # Create model
            models_train.append(model_init_params["model_dir"])
            model = create_model(args.trainer, **model_init_params)
            
            start = time.time()

            # Train model
            model.train(
                texts_train.lemmas,
                texts_train.id_tm,
                num_topics=k,
                **tr_params,
                texts_test=texts_test.lemmas,
                ids_test=texts_test.id_tm
            )

            # Saave model
            model.save_model(
                path=model_init_params["model_dir"] /
                "model_data" / "model.pickle"
            )
            
            print("-- -- Model trained in:", time.time() - start)

            print(f"-- -- Model saved in {model_init_params['model_dir']}")
            
            # Create trainconfig
            create_trainconfig(
                modeldir=model_init_params["model_dir"],
                model_name=f"{args.trainer}_{k}_topics",
                model_desc=f"{args.trainer}_{k}_topics model trained with {k} on {dir_text_processed.stem}",
                trainer=args.trainer,
                TrDtSet=dir_text_processed.as_posix(),
                TMparam=tr_params
            )
