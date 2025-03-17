import pathlib
import numpy as np
import pandas as pd
from src.TopicModeling.tm_utils.tm_model import TMmodel

path_data = pathlib.Path("/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/data/train_data/training_dfs_per_cpv")
path_data_all = pathlib.Path("/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/data/train_data/processed_objectives_with_lemmas_emb_label.parquet")
path_models = pathlib.Path("/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/zaragoza_models")
#pathlib.Path("/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/final_cpv_models_with_labels_corrected")
path_save = pathlib.Path("/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/users_to_eval_zaragoza")
path_save.mkdir(parents=True, exist_ok=True)

# get number of documents per CPV
cpv_n_docs = {}
for cpv_file in path_data.iterdir():
    
    cpv = cpv_file.name.split("_")[-1].split(".")[0]
    df = pd.read_parquet(cpv_file)
    cpv_n_docs[cpv] = len(df)

# keep the CPVs that:
# - have the closest to the 80 % of the total number of documents
# - have the closest to the 60 % of the total number of documents
# - have the closest to the 10 % of the total number of documents
total_docs = sum(cpv_n_docs.values())
cpv_80 = min(cpv_n_docs, key=lambda x: abs(cpv_n_docs[x] - 0.8 * total_docs))
cpv_10 = min(cpv_n_docs, key=lambda x: abs(cpv_n_docs[x] - 0.1 * total_docs))
cpv_005 = min(cpv_n_docs, key=lambda x: abs(cpv_n_docs[x] - 0.05 * total_docs))

df_all = pd.read_parquet(path_data_all)
    
for directory in path_models.iterdir():
    if not directory.is_dir():
        continue
    
    """
    cpv = directory.name.split("_")[-1] 
    print(f"Processing CPV: {cpv}")
    
    if cpv not in [cpv_80, cpv_10, cpv_005]:
        continue
    """
    path_save_user = path_save / directory.name
    path_save_user.mkdir(parents=True, exist_ok=True)
    
    #for file in directory.iterdir():
        #if file.is_dir():
    file = directory
    print(f"Processing model: {file.name}")
    #import pdb; pdb.set_trace()
    tmmodel = TMmodel(file.joinpath("model_data/TMmodel"))
    tmmodel._load_s3()
    tmmodel._load_thetas()
    tmmodel._load_alphas()
    tmmodel._load_ndocs_active()
    
    # Load topic-keys
    with (file / "model_data/TMmodel" / "tpc_descriptions.txt").open('r', encoding='utf8') as fin:
        topics_keys = [el.strip() for el in fin.readlines()]

    # Load topic labels
    with (file / "model_data/TMmodel" / "tpc_labels.txt").open('r', encoding='utf8') as fin:
        topics_labels = [el.strip() for el in fin.readlines()]
    topics_labels = [label for label in topics_labels if len(label) > 1]

    # Load alphas and number of active documents
    alphas = np.round(tmmodel._alphas * 100, 2)
    ndocs_active = tmmodel._ndocs_active

    # Load docs
    corpusFile = file.joinpath('train_data/corpus.txt')
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
    df_corpus_train = pd.merge(df_corpus_train, df_all[['place_id', 'procurement_id','raw_text']], left_on='id', right_on = "place_id", how='inner')

    s3 = tmmodel._s3.toarray()
    thetas = tmmodel._thetas.toarray()
    #s3_thetas = s3 + thetas
    top_docs_per_topic = tmmodel.get_most_representative_per_tpc(s3, topn=3)
    
    # get text and summary for each top doc
    top_docs_per_topic_text = []
    for topic_docs in top_docs_per_topic:
        docs = []
        for doc in topic_docs:
            doc_text = df_corpus_train[df_corpus_train.id == documents_ids[doc]].raw_text.values[0]
            # if doc_text does not end in a dot, cut the text in the last dot and remove the rest
            #if doc_text[-1] != ".":
            #    doc_text = doc_text.rsplit(".", 1)[0] + "."
            docs.append(doc_text)
        top_docs_per_topic_text.append(docs)

    top_docs_0 = [docs[0] for docs in top_docs_per_topic_text]
    top_docs_1 = [docs[1] for docs in top_docs_per_topic_text]
    top_docs_2 = [docs[2] for docs in top_docs_per_topic_text]

    df = pd.DataFrame(
        {
            "ID del tópico": range(len(topics_keys)),
            "Etiqueta del tópico": topics_labels,
            "Tamaño del tópico (%)": alphas,
            "Nº documentos activos": ndocs_active,
            "Palabras más representativas": topics_keys,
            "Documento más significativo 1": top_docs_0,
            "Documento más significativo 2": top_docs_1,
            "Documento más significativo 3": top_docs_2,
        }
    )
    path_save_excel = path_save_user / f"{file.name}.xlsx"
    df.to_excel(path_save_excel)