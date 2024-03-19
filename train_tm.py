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
        default="config/options.yaml",
        help="Path to options YAML file"
    )
    parser.add_argument(
        "--trainer",
        default="Mallet",
        help="Trainer to use for topic modeling"
    )
    parser.add_argument(
        "--test_split",
        default=0.3,
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
    num_topics = [int(k) for k in num_topics.split(",")]
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
    texts_train, texts_test = train_test_split(df_sample, args.test_split)
    logger.info("Data loaded.")
    logger.info(
        f"Train: {len(texts_train)} documents. Test: {len(texts_test)}.")

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

    models_train = []
    for k in num_topics:

        logger.info(f"Training {args.trainer} model with {k} topics.")

        model_init_params["model_dir"] = (dir_output_models.joinpath(args.trainer)).joinpath(f"{args.trainer}_{k}_topics")
        models_train.append(model_init_params["model_dir"])
        model = create_model(args.trainer, **model_init_params)

        # Train model
        model.train(
            texts_train,
            num_topics=k,
            **tr_params,
            texts_test=texts_test,
        )
        
        # Saave model
        model.save_model(
            path=model_init_params["model_dir"].joinpath("model.pickle")
        )
        
        # Create trainconfig
        create_trainconfig(
            modeldir=model_init_params["model_dir"],
            model_name=f"{args.trainer}_{k}_topics",
            model_desc=f"{args.trainer}_{k}_topics model trained with {num_topics} on {dir_text_processed.stem}",
            trainer=args.trainer,
            TrDtSet=dir_text_processed.as_posix(),
            TMparam=tr_params
        )    
        
