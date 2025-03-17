import pathlib
import numpy as np
import plotly.express as px
import pandas as pd
from src.TopicModeling.tm_utils.cpv_codes import CPV_CODES

path_models = pathlib.Path("/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/final_cpv_models_with_labels_corrected")

for version in ["small", "large"]:
    output_html = pathlib.Path(f"sample_data/to_eval/cpv_topic_hierarchy_treemap_{version}.html")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    
    cpv_list = []  
    topic_list = []  
    words_list = []
    size_list = []  
    ALPHA_SCALE_FACTOR = 5000  

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
        tpc_labels_path = selected_model_path / "model_data/TMmodel/tpc_labels.txt"
        tpc_words_path = selected_model_path / "model_data/TMmodel/tpc_descriptions.txt"
        corpusFile = selected_model_path / "train_data/corpus.txt"
        alphas_path = selected_model_path / "model_data/TMmodel/alphas.npy"

        with open(corpusFile, "r") as f:
            n_docs = len(f.readlines())

        alphas = np.load(alphas_path)

        try:
            with open(tpc_labels_path, "r") as f:
                tpc_labels = [label.strip() for label in f.readlines()]
            with open(tpc_words_path, "r") as f:
                tpc_words = [words.strip() for words in f.readlines()]
        except FileNotFoundError:
            print(f"File not found for CPV {cpv}")
            tpc_labels = []
            tpc_words = []

        alphas_scaled = np.log1p(alphas * n_docs) if len(alphas) > 0 else []
        
        cpv_name = f"CPV {cpv} - {CPV_CODES[cpv]}"
        for topic, alpha, keywords in zip(tpc_labels, alphas_scaled, tpc_words):
            keywords_split = keywords.split(", ")
            for word in keywords_split:
                cpv_list.append(cpv_name)
                topic_list.append(topic)
                words_list.append(word)
                size_list.append(alpha)
            #import pdb; pdb.set_trace()
    df = pd.DataFrame({
        "CPV": cpv_list,  
        "Topic": topic_list,  
        "Words": words_list,
        "Size": size_list,
    })

    # put a minimum value for the size when it is 0
    df["Size"] = df["Size"].apply(lambda x: max(x, 1))
    df["Treemap Size"] = df["Size"]
    df["Is_Top_Level"] = df["CPV"] == df["CPV"]  # True for CPV-level

    fig = px.treemap(
        df,
        path=["CPV", "Topic", "Words"],  
        values="Treemap Size",
        color="Treemap Size",
        color_continuous_scale="Blues",
        hover_data={"CPV": False, "Topic": False, "Words": True, "Treemap Size": False}  # Hide unwanted fields
    )
    
    # Update hover template with conditional logic
    for trace in fig.data:
        trace.customdata = [
            ["" if label.startswith("CPV") else f"Parent: {parent}"]  # Hide parent for top-level nodes
            for label, parent in zip(trace.labels, trace.parents)
        ]

    # Update hover template
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>%{customdata[0]}",
        maxdepth=3
    )
    
    fig.write_html(output_html)
        
