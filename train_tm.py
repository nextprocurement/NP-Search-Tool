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
        default=0.1,
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

    """
    subsample = int(options.get("subsample", 0))
    df_processed = pd.read_parquet(dir_text_processed).dropna()
    df_sample = df_processed.loc[df_processed["preprocessed_text"].apply(lambda x: len(x.split()) > 5), ["preprocessed_text", "id_tm"]]
    if subsample:
        if subsample > len(df_sample):
            logger.warning(
                f"Subsample of {subsample} is larger than population. Setting subsample to max value ({len(df_processed)} samples)."
            )
            subsample = len(df_sample)
        df_sample = df_sample.sample(n=subsample, random_state=42)
    texts_train, texts_test = train_test_split(df_sample, args.test_split)
    logger.info("Data loaded.")
    logger.info(
        f"Train: {len(texts_train)} documents. Test: {len(texts_test)}.")
    """
        
        
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
    
    # Paths to data
    path_parquets = pathlib.Path("/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/all_processed")
    path_place_with_lote = path_parquets / "minors_insiders_outsiders_origen_con_lot_info.parquet"
    path_place_without_lote = path_parquets / "minors_insiders_outsiders_origen_sin_lot_info.parquet"
    # This is PLACE WITH LOTE processed with Spacy md model
    path_place_processed_1 = path_parquets / "md.parquet"
    # This is PLACE WITH LOTE processed with Spacy md model + filtering
    # stopwords (/export/usuarios_ml4ds/lbartolome/NextProcurement/data/stw_lists/es)
    path_place_processed_2 = path_parquets / "md2.parquet"
    # This is PLACE WITH LOTE processed with Spacy trf model
    path_place_processed_3 = path_parquets / "trf.parquet"
    # This is PLACE WITHOUT LOTE processed with Spacy md model + filtering
    # stopwords (/export/usuarios_ml4ds/lbartolome/NextProcurement/data/stw_lists/es)
    path_place_processed_no_lote = path_parquets / "trf_lote_es.parquet" #"md_sin_lote_es.parquet" #"md_sin_lote.parquet"
    path_save = pathlib.Path(
        "/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/processed/minors_insiders_outsiders.parquet")
    path_manual_stops = "sample_data/stopwords"
    path_eq = "sample_data/eq.txt"
    # Read PLACE files. This files are used to merge with the processed data while keeping the information about the origin of the text (minors, insiders, outsiders)
    place_with_lote = pd.read_parquet(path_place_with_lote)
    place_without_lote = pd.read_parquet(path_place_without_lote)

    # Processed data
    processed = [path_place_processed_1, path_place_processed_2, path_place_processed_3]
    
    filter_stops = True
    
    dfs_to_train = []
    """
    for path_processed in processed:
        processed = pd.read_parquet(path_processed)
        processed_name = path_processed.stem
        if filter_stops:
            stopwords = set()
            for archivo in os.listdir(path_manual_stops):
                if archivo.endswith('.txt'):
                    ruta_completa = os.path.join(path_manual_stops, archivo)
                    with open(ruta_completa, 'r', encoding='utf-8') as f:
                        stopwords.update(f.read().splitlines())

            # Vectorizar el proceso de eliminación de stopwords
            def eliminar_stopwords(fila):
                return ' '.join([palabra for palabra in fila.split() if palabra not in stopwords])

            # Aplicar la función de processed vectorizada
            processed['lemmas'] = processed['lemmas'].apply(eliminar_stopwords)
            
        # Merge processed data with PLACE data
        place_with_lote = pd.merge(processed,place_with_lote,how='left', on='id_tm')
        dfs_to_train.append((f"place_with_lote_{processed_name}", place_with_lote))
        
        # Generate only minors, insiders and outsiders
        place_with_lote_minors = place_with_lote[place_with_lote.origin == "minors"]
        dfs_to_train.append((f"place_with_lote_minors_{processed_name}", place_with_lote_minors))
        place_with_lote_outsiders = place_with_lote[place_with_lote.origin == "outsiders"]
        dfs_to_train.append((f"place_with_lote_outsiders_{processed_name}", place_with_lote_outsiders))
        place_with_lote_insiders = place_with_lote[place_with_lote.origin == "insiders"]
        dfs_to_train.append((f"place_with_lote_insiders_{processed_name}", place_with_lote_insiders))
    """
    
    # Merge processed data without lote with PLACE datas
    processed = pd.read_parquet(path_place_processed_no_lote)
    processed = processed.sample(frac=0.1).reset_index(drop=True)
    if filter_stops:
            stopwords = set()
            for archivo in os.listdir(path_manual_stops):
                if archivo.endswith('.txt'):
                    ruta_completa = os.path.join(path_manual_stops, archivo)
                    with open(ruta_completa, 'r', encoding='utf-8') as f:
                        stopwords.update(f.read().splitlines())

            # Vectorizar el proceso de eliminación de stopwords
            def eliminar_stopwords(fila):
                return ' '.join([palabra for palabra in fila.split() if palabra not in stopwords])

            # Aplicar la función de processed vectorizada
            processed['lemmas'] = processed['lemmas'].apply(eliminar_stopwords)
    
    # Reemplazar equivalencias
    pares_diccionario = {}
    with open(path_eq, 'r') as archivo:
        for linea in archivo:
            linea = linea.strip()
            palabras = linea.split(':')
            #if det(palabras[0]) == "es":
            #    pares_diccionario[palabras[0]] = palabras[1]
            #else:
            #    print(palabras[0], det(palabras[0]))
            patron = r'\b{}\b'.format(re.escape(palabras[0]))
            pares_diccionario[patron] = palabras[1]
    pares_diccionario = \
        dict(sorted(pares_diccionario.items(), key=lambda x: x[0]))
    
    def reemplazar_palabras(texto, diccionario):
        for palabra_original, palabra_nueva in diccionario.items():
            #patron = r'\b{}\b'.format(re.escape(palabra_original))
            #texto = texto.replace(palabra_original, palabra_nueva)
            texto = re.sub(palabra_original, palabra_nueva, texto)
        return texto
    
    processed['lemmas'] = processed['lemmas'].apply(lambda x: reemplazar_palabras(x, pares_diccionario))

    place_without_lote = pd.merge(processed,place_without_lote,how='left', on='id_tm')
    dfs_to_train.append(("place_without_lote", place_without_lote))
    
    place_without_lote_minors = place_without_lote[place_without_lote.origen == "minors"]
    dfs_to_train.append(("place_without_lote_minors", place_without_lote_minors))
    place_without_lote_outsiders = place_without_lote[place_without_lote.origen == "outsiders"]
    dfs_to_train.append(("place_without_lote_outsiders", place_without_lote_outsiders))
    place_without_lote_insiders = place_without_lote[place_without_lote.origen == "insiders"]
    dfs_to_train.append(("place_without_lote_insiders", place_without_lote_insiders))
   
    # Train models
    lang = "es"
    models_train = []
    for k in num_topics:
        logger.info(f"{'='*10} Training models with {k} topics. {'='*10}")
        for name, df in dfs_to_train:
            logger.info(f"{'='*5} Training {args.trainer} model with {k} topics on {name} {'='*5}")
            
            # Store df in training parquet file
            logger.info(f"-- -- Number of documents: {len(df)}")
            # df.to_parquet(path_save)
            
            texts_train, texts_test = train_test_split(df, args.test_split)
            logger.info("Data loaded.")
            logger.info(
                f"Train: {len(texts_train)} docs. Test: {len(texts_test)}.")
            
            # Update model parameters
            model_init_params["model_dir"] = (dir_output_models.joinpath(args.trainer)).joinpath(f"{lang}_{args.trainer}_{name}_{k}_topics")
            
            # Create model
            models_train.append(model_init_params["model_dir"])
            model = create_model(args.trainer, **model_init_params)

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
                path=model_init_params["model_dir"] / "model_data" / "model.pickle"
            )
            
            # Create trainconfig
            create_trainconfig(
                modeldir=model_init_params["model_dir"],
                model_name=f"{args.trainer}_{k}_topics",
                model_desc=f"{args.trainer}_{k}_topics model trained with {k} on {dir_text_processed.stem}",
                trainer=args.trainer,
                TrDtSet=dir_text_processed.as_posix(),
                TMparam=tr_params
            )    
        
