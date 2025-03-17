from collections import defaultdict
import pathlib
from subprocess import check_output
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import yaml
from src.TopicModeling.tm_utils.tm_model import TMmodel
from src.Utils.utils import set_logger
from src.TopicModeling.tm_utils.utils import tkz_clean_str, preprocBOW

path_models = pathlib.Path("/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/final_cpv_models_with_labels_corrected")
path_infer_data = pathlib.Path("/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/data/train_data/processed_objectives_with_lemmas_emb_label.parquet")
path_new_embeddings = pathlib.Path("/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/data/train_data/to_process/all_objectives_embeddings.parquet")
min_lemmas=1

def calculate_s3(corpusFile, thetas, betas, vocab_w2id):
    """Given the path to a TMmodel, it calculates the similarities between documents and saves them in a sparse matrix.
    """
    
    t_start = time.perf_counter()
    
    with corpusFile.open("r", encoding="utf-8") as f:
        lines = f.readlines()  
        f.seek(0)
        try:
            documents_texts = [line.rsplit(" 0 ")[1].strip().split() for line in lines]
        except:
            documents_texts = [line.rsplit("\t0\t")[1].strip().split() for line in lines]
    
    D = len(thetas.toarray())
    K = len(betas)
    S3 = np.zeros((D, K))

    for doc in range(D):
        for topic in range(K):
            wd_ids = [
                vocab_w2id[word] 
                for word in documents_texts[doc] 
                if word in vocab_w2id
            ]
            S3[doc, topic] = np.sum(betas[topic, wd_ids])
            
    sparse_S3 = sparse.csr_matrix(S3)
    sparse.save_npz(corpusFile.parent.joinpath('s3.npz'), sparse_S3)

    t_end = time.perf_counter()
    t_total = (t_end - t_start)/60
    print(f"Total computation time: {t_total}")
    
    return S3
    

def infer_thetas(path_model, num_topics, docs, mallet_path):
    num_iterations = 1000
    doc_topic_thr = 0.0
    holdout_corpus = path_model / "infer_data" / "corpus.txt"
    with holdout_corpus.open("w", encoding="utf8") as fout:
        for _, (identifier, text) in enumerate(docs):
            clean_text = f"{text}".replace('\n', ' ').replace('\r', '')
            fout.write(f"{identifier} 0 {clean_text}\n")
        print(f"-- -- Mallet corpus.txt for inference created.")

    # Get inferencer
    inferencer = path_model / "model_data" / "inferencer.mallet"

    # Files to be generated thorough Mallet
    corpus_mallet_inf = path_model / "infer_data" / "corpus_inf.mallet"
    doc_topics_file = path_model / "infer_data" / "doc-topics-inf.txt"

    # Extract pipe
    # Get corpus file
    path_corpus = path_model / "train_data" / "corpus.mallet"
    if not path_corpus.is_file():
        print(f"-- Pipe extraction: Could not locate corpus file")

    # Create auxiliary file with only first line from the original corpus file
    path_txt = path_model / "train_data" / "corpus.txt"
    with path_txt.open('r', encoding='utf8') as f:
        first_line = f.readline()
    path_aux = path_model / "train_data" / "corpus_aux.txt"
    with path_aux.open('w', encoding='utf8') as fout:
        fout.write(first_line + '\n')

    # We perform the import with the only goal to keep a small file containing the pipe
    print(f"-- Extracting pipeline")
    path_pipe = path_model / "train_data" / "import.pipe"

    cmd = mallet_path.as_posix() + \
        ' import-file --use-pipe-from %s --input %s --output %s'
    cmd = cmd % (path_corpus, path_aux, path_pipe)

    try:
        print(f'-- Running command {cmd}')
        check_output(args=cmd, shell=True)
    except:
        print('-- Failed to extract pipeline. Revise command')

    # Import data to mallet
    print('-- Inference: Mallet Data Import')

    #
    cmd = mallet_path.as_posix() + \
        ' import-file --use-pipe-from %s --input %s --output %s'
    cmd = cmd % (path_pipe, holdout_corpus, corpus_mallet_inf)

    try:
        print(f'-- Running command {cmd}')
        check_output(args=cmd, shell=True)
    except Exception as e:
        print(e)
        print('-- Mallet failed to import data. Revise command')

    # Get topic proportions
    print('-- Inference: Inferring Topic Proportions')

    cmd = mallet_path.as_posix() + \
        ' infer-topics --inferencer %s --input %s --output-doc-topics %s ' + \
        ' --doc-topics-threshold ' + str(doc_topic_thr) + \
        ' --num-iterations ' + str(num_iterations)
    cmd = cmd % (inferencer, corpus_mallet_inf, doc_topics_file)

    try:
        print(f'-- Running command {cmd}')
        check_output(args=cmd, shell=True)
    except:
        print('-- Mallet inference failed. Revise command')

    cols = [k for k in np.arange(2, num_topics + 2)]
    thetas32 = np.loadtxt(doc_topics_file, delimiter='\t', dtype=np.float32, usecols=cols)
    thetas32[thetas32 < 3e-3] = 0
    thetas32 = normalize(thetas32, axis=1, norm='l1')
    thetas32 = sparse.csr_matrix(thetas32, copy=True)
    
    path_save = path_model / "infer_data" / "thetas.npz"
    sparse.save_npz(path_save, thetas32)
    
    return thetas32, holdout_corpus

def get_info_doc(thetas, id_row):
    this_info_doc = {}
    for k in range(thetas.shape[1]):
        if thetas[id_row, k] > 0:
            this_info_doc[k] = str(thetas[id_row, k])
    return this_info_doc
    
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
    parser.add_argument(
        "--data_path",
        default="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/data/train_data/training_dfs_per_cpv",
        help="Path to data"
    )
    parser.add_argument(
        "--num_topics",
        default="5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50",
        #default="5,10,15",
        help="Number of topics to train the model on"
    )
    parser.add_argument(
        "--path_stops",
        default="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/stopwords",
        help="Path to stopwords file"
    )
    parser.add_argument(
        "--path_eq",
        default="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/equivalences/manual_equivalences.txt",
        help="Path to equivalence dictionary file"
    )
    parser.add_argument(
        "--final_models",
        default="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/final_cpv_models_with_labels_corrected",
        help="Path to final models"
    )
    parser.add_argument(
        "--extra",
        type=bool,
        default=False,
        help="Extra parameter for the model"
    )
    parser.add_argument(
        "--path_data",
        default="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/data/train_data/processed_objectives_with_lemmas_emb_label.parquet",
        help="Path to data"
    )
    parser.add_argument(
        "--cpv",
        default="0.45",
        type=str,
        help="CPV model to train"
    )
    args = parser.parse_args()
    
    
    with open(args.options, "r") as f:
        options = dict(yaml.safe_load(f))

    # Set logger
    dir_logger = pathlib.Path(options.get("dir_logger", "app.log"))
    console_log = options.get("console_log", True)
    file_log = options.get("file_log", True)
    logger = set_logger(
        console_log=console_log,
        file_log=file_log,
        file_loc=dir_logger
    )
    dir_mallet = pathlib.Path(options.get("dir_mallet"))
    
    #####################
    # Load stops /eqs #
    #####################
    print(f"Loading stopwords")
    start_time = time.time()
    stopwords = set()
    for file in os.listdir(args.path_stops):
        if file.endswith('.txt'):
            with open(os.path.join(args.path_stops, file), 'r', encoding='utf-8') as f:
                stopwords.update(f.read().splitlines())            
    print(f"-- -- {len(stopwords)} stopwords load in {time.time() - start_time:.2f} seconds")

    print(f"Loading equivalences")
    start_time = time.time()
    equivalents = {}
    with open(args.path_eq, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            words = line.split(':')
            if len(words) < 2:
                print(f"Skipped or incomplete line: '{line}'")
                continue
            equivalents[words[0]] = words[1]
    print(f"-- -- {len(equivalents)} equivalence loaded in: {time.time() - start_time:.2f} seconds")

############################################
# Main
############################################
version = "small"

df = pd.read_parquet(path_infer_data)
df_embeddings = pd.read_parquet(path_new_embeddings)
# drop column 'embeddings' in df
df = df.drop(columns=['embeddings'])
# get column 'embeddings' from df_embeddings
df = pd.merge(df, df_embeddings[['id', 'embeddings']], on='id', how='inner')

df = df.drop_duplicates(subset=['place_id'])
df = df.explode("two_cpv")
df_all = df.copy()

df_save = df.copy() # this df will be saved with the new information
df["kept_in_training"] = False
df["kept_in_inference"] = False
df["avg_s3"] = 0.0 
df["kept_by_s3"] = False
df["topic_info_small"] = None
df["topic_info_large"] = None

# keep rows that were not predicted as 1
df = df[df["predicted_label"] != 1]

for version in ["small", "large"]:

    count_per_cpv = defaultdict(int)
    for directory in path_models.iterdir():
        if not directory.is_dir():
            continue

        cpv = directory.name.split("_")[-1] 
        print(f"Processing CPV: {cpv}")

        topics_pos = []
        topics_files = []

        for file in directory.iterdir():
            if file.is_dir():
                try:
                    n_topics = int(file.name.split("_")[0])
                    topics_pos.append(n_topics)
                    topics_files.append(file)
                except ValueError:
                    continue

        if not topics_pos:
            print(f"No valid topis for CPV {cpv}")
            continue
        
        n_topics_graph = min(topics_pos) if version == "small" else max(topics_pos)
        tmmodel_path = topics_files[topics_pos.index(n_topics_graph)]
        tmmodel = TMmodel(
            tmmodel_path.joinpath("model_data/TMmodel"))

        tmmodel._load_s3()
        s3 = tmmodel._s3.toarray()
        tmmodel._load_vocab()
        tmmodel._load_betas()
        tmmodel._load_thetas()
        tmmodel._load_vocab_dicts()
        vocabulary = tmmodel._vocab
        betas = tmmodel._betas
        vocab_w2id = tmmodel._vocab_w2id
        thetas = tmmodel._thetas
        
        # Load training corpus to see which documents were kept in the training data
        corpusFile = tmmodel_path.joinpath('train_data/corpus.txt')
        with corpusFile.open("r", encoding="utf-8") as f:
            lines = f.readlines()  
            f.seek(0)
            try:
                documents_ids = [line.rsplit(" 0 ")[0].strip() for line in lines]
                documents_texts = [line.rsplit(" 0 ")[1].strip().split() for line in lines]
            except:
                documents_ids = [line.rsplit("\t0\t")[0].strip() for line in lines]
                documents_texts = [line.rsplit("\t0\t")[1].strip().split() for line in lines]
        df_corpus_train = pd.DataFrame({'id': documents_ids, 'text': documents_texts})
        df_corpus_train["id_int"] = range(df_corpus_train.shape[0])
        df_corpus_train[f"topic_info_{version}"] = df_corpus_train.apply(lambda x: get_info_doc(thetas, x.id_int), axis=1)
        # filter df_all by CPV
        df_all_ = df_all[df_all["two_cpv"] == cpv]
        df_corpus_train = pd.merge(
            df_corpus_train, 
            df_all_[['place_id', 'procurement_id', 'raw_text']], 
            left_on='id', 
            right_on="place_id", 
            how='left'
        )

        # append count if the cpv is not in count_per_cpv already
        if cpv not in count_per_cpv:
            count_per_cpv[cpv] = df_corpus_train.shape[0]
        
        # update in the df_save the rows that were kept in the training data
        df_save.loc[df_save['place_id'].isin(df_corpus_train['place_id']) & (df_save.two_cpv == cpv), 'kept_in_training'] = True
        
        # get thetas representation from the training data
        df_save.loc[df_save['place_id'].isin(df_corpus_train['place_id']) & (df_save.two_cpv == cpv), f"topic_info_{version}"] = df_corpus_train[f"topic_info_{version}"]
        
        # keep rows whose "two_cpv" is the same as the current cpv
        df_cpv = df[df["two_cpv"] == cpv]
        
        # Carry out the same preprocessing as in the training data
        print(f"Data shape: {df_cpv.shape}")
        print(f"Average lemmas per document (before processing): {df_cpv['lemmas'].apply(lambda x: len(x.split())).mean()}")
        
        # Clean text
        df_cpv['lemmas'] = df_cpv['lemmas'].apply(lambda x: tkz_clean_str(x, stopwords, equivalents))
        print(f"-- -- Text cleaned in {time.time() - start_time:.2f} seconds")

        # remove words that are not in the vocabulary
        df_cpv['lemmas'] = df_cpv['lemmas'].apply(lambda x: ' '.join([word for word in x.split() if word in vocabulary]))

        # remove rows with less than min_lemmas lemmas
        df_cpv = df_cpv[df_cpv['lemmas'].apply(lambda x: len(x.split())) >= min_lemmas]
        print(f"Data shape after filtering: {df_cpv.shape}")
        print(f"Average lemmas per document (after processing): {df_cpv['lemmas'].apply(lambda x: len(x.split())).mean()}")
        
        # check that there is no empty lemmas
        print(f"Empty lemmas: {df_cpv['lemmas'].apply(lambda x: len(x)).sum() == 0}")
        
        # make inference
        docs = df_cpv[["place_id", "lemmas"]].values
        thetas_infer, infer_corpus_path = infer_thetas(tmmodel_path, n_topics_graph, docs, dir_mallet)
        
        s3_infer = calculate_s3(infer_corpus_path, thetas_infer, betas, vocab_w2id)
        # calculate average per document
        s3_infer_avg = s3_infer.mean(axis=1)
        
        # get the ids of the documents that were kept in the inference data
        with infer_corpus_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()  
            f.seek(0)
            try:
                documents_ids = [line.rsplit(" 0 ")[0].strip() for line in lines]
                documents_texts = [line.rsplit(" 0 ")[1].strip().split() for line in lines]
            except:
                documents_ids = [line.rsplit("\t0\t")[0].strip() for line in lines]
                documents_texts = [line.rsplit("\t0\t")[1].strip().split() for line in lines]
        df_corpus_infer = pd.DataFrame({'id': documents_ids, 'text': documents_texts})
        df_corpus_infer["id_int"] = range(df_corpus_infer.shape[0])
        # get topic info for the inference data
        df_corpus_infer[f"topic_info_{version}"] = df_corpus_infer.apply(lambda x: get_info_doc(thetas_infer, x.id_int), axis=1)
        df_corpus_infer = pd.merge(
            df_corpus_infer, 
            df_all_[['place_id', 'procurement_id', 'raw_text']], 
            left_on='id', 
            right_on="place_id", 
            how='left'
        )
        
        # update in the df_save the rows that were kept in the infer data
        # the conditions to be met are that the place_id is in the df_corpus_infer and the "two_cpv" is the same as the current cpv
        df_save.loc[df_save['place_id'].isin(df_corpus_infer['place_id']) & (df_save.two_cpv == cpv), 'kept_in_inference'] = True
        
        # get thetas representation from the inference data
        df_save.loc[df_save['place_id'].isin(df_corpus_infer['place_id']) & (df_save.two_cpv == cpv), f"topic_info_{version}"] = df_corpus_infer[f"topic_info_{version}"]
        
        # update the average s3 in the df_save
        df_save.loc[df_save['place_id'].isin(df_corpus_infer['place_id']) & (df_save.two_cpv == cpv), "avg_s3"] = s3_infer_avg

        # if the average is higher than s3_mean keep the document
        thr = s3.mean()
        df_save.loc[df_save['place_id'].isin(df_corpus_infer['place_id']) & (df_save.two_cpv == cpv) & (df_save['avg_s3'] >= thr), "kept_by_s3"] = True


    # set to default values the rows that were not processed
    df_save.loc[df_save['kept_in_training'] != True, 'kept_in_training'] = False
    df_save.loc[df_save['kept_in_inference'] != True, 'kept_in_inference'] = False
    df_save.loc[df_save['avg_s3'].isna(), 'avg_s3'] = 0.0
    df_save.loc[df_save['kept_by_s3'] != True, 'kept_by_s3'] = False
    df_save[f"topic_info_{version}"] = df_save[f"topic_info_{version}"].apply(lambda x: str(x))

df_count_per_cpv = pd.DataFrame(count_per_cpv.items(), columns=['cpv', 'count'])
df_count_per_cpv.to_csv("/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/data/train_data/count_per_cpv.csv", index=False)

df_save.to_parquet("/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/data/train_data/processed_objectives_with_lemmas_emb_label_tm_augmented.parquet")