from collections import defaultdict
import pandas as pd
import pathlib
import json
from src.TopicModeling.tm_utils.tm_model import TMmodel
from src.Utils.utils import set_logger
from src.TopicModeling.tm_utils.utils import tkz_clean_str, preprocBOW

path_models = pathlib.Path("/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/final_cpv_models_with_labels_corrected")
path_data = "/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/data/train_data/processed_objectives_with_lemmas_emb_label_tm_augmented.parquet"
path_predicted_cpv = "/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/data/cpv_predicted"

# read jsons with cpv; path_predicted_cpv is a directory with jsons; concat all json but the one with "zaragoza" in the name, which we keep apart
cpv_jsons = []
for file in pathlib.Path(path_predicted_cpv).iterdir():
    if "zaragoza" in file.name:
        df_zaragoza = pd.read_json(file)
    else:
        cpv_jsons.append(pd.read_json(file))
df_cpv = pd.concat(cpv_jsons)


df_data = pd.read_parquet(path_data)
# keep only the predicted objectives that were kept according to the criteria
#df_data = df_data[df_data["title_origin"] == "augmented"]
df_filtered = df_data[
    (df_data["kept_in_training"]) |
    (
        df_data["kept_in_inference"] & 
        df_data[["kept_by_s3_small", "kept_by_s3_large"]].any(axis=1)
    )
]
df_grouped = df_filtered.groupby("place_id").agg({
    "id": "first",  # Keeps the first occurrence (assuming it's unique)
    "text": "first",
    "id_int": "first",
    "procurement_id": "first",
    "raw_text": "first",
    "topic_info_small": list,  # Converts into lists
    "topic_info_large": list,
    "two_cpv": list
}).reset_index()


# añadir información de cpv a df_data
# meter modelo grande y pequeño
# renombar modelos o finegrined and whathever other word.

info_tpcs = defaultdict(dict)
for version in ["small", "large"]:
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
        selected_model_path = topics_files[topics_pos.index(n_topics_graph)]
                
        print(f"Processing model: {selected_model_path.name}")
        tmmodel = TMmodel(selected_model_path.joinpath("model_data/TMmodel"))
        tmmodel._load_s3()
        tmmodel._load_thetas()
        tmmodel._load_alphas()
        tmmodel._load_ndocs_active()
        
        tpc_labels_path = selected_model_path / "model_data/TMmodel/tpc_labels.txt"
        tpc_words_path = selected_model_path / "model_data/TMmodel/tpc_descriptions.txt"

        try:
            with open(tpc_labels_path, "r") as f:
                tpc_labels = [label.strip() for label in f.readlines()]
            with open(tpc_words_path, "r") as f:
                tpc_words = [words.strip() for words in f.readlines()]
        except FileNotFoundError:
            print(f"File not found for CPV {cpv}")
            tpc_labels = []
            tpc_words = []
            
        this_info_tpcs = {tpc_id:tpc_labels[tpc_id].strip() for tpc_id in range(len(tpc_labels))}
        info_tpcs[cpv][version] = this_info_tpcs

print("Processing of info_docs starts...")
info_docs = {}
for id_int, df_row in df_grouped.iterrows():
    id_p = df_row['place_id']
    
    this_doc_topic_dict = {}
    for cpv, topic_info_small, topic_info_large in zip(df_row["two_cpv"], df_row["topic_info_small"], df_row["topic_info_large"]):
        this_doc_topic_dict[cpv] = {
            "small": topic_info_small,
            "large": topic_info_large
        }
    
    augmented_cpv = df_cpv[df_cpv["procurement_id"] == df_row["procurement_id"]]["cpv_predicted_objective"].values.tolist() if df_row["procurement_id"] in df_cpv["procurement_id"].values else df_zaragoza[df_zaragoza["expediente"] == id_p]["cpv_predicted_objective"].values.tolist()
    
    this_doc_dict = {
        "augmented_objective": df_row["raw_text"],
        "augmented_cpv": augmented_cpv
    }
    info_docs[id_p] = this_doc_dict
    
json_all = {
    "info_topics": info_tpcs,
    "info_docs": info_docs
}
try:
    with open('/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/final_objectives_tm_info_for_bsc.json', 'w') as json_file:
        json.dump(json_all, json_file, indent=4, ensure_ascii=False)
except Exception as e:
    import pdb; pdb.set_trace()