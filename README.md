# NP-Search-Tools

NP-Search-Tools is a modular toolkit for preprocessing text, generating embeddings, training and running inference on topic models, and building semantic graphs. The output is structured for integration with the [NP-Backend-Dockers](https://github.com/nextprocurement/NP-Backend-Dockers) backend for indexing and further analysis.

## Main Modules

All core functionalities are located in the [``src``](``src``) directory and are organized into the following modules:

### Preprocessor

Cleans and tokenizes raw text. The default pipeline detects language, normalizes text, and applies a customizable sequence of transformations.

#### Main Components:
- **Lemmatizer**  
  You can choose between a custom lemmatizer for Spanish (rule-based) or faster alternatives like those provided by SpaCy. By default, SpaCy is used (`options.yaml/vocabulary: false`).

- **NgramProcessor**  
  Provides functions to search, filter, replace, or validate n-grams.

- **TextProcessor**  
  Builds and executes the preprocessing pipeline. It integrates all previously mentioned tools and supports custom stopwords, ngrams, and vocabularies.  
  Supported steps:
  - `lowercase`
  - `remove_tildes`
  - `remove_extra_spaces`
  - `remove_punctuation`
  - `remove_urls`
  - `lemmatize_text`
  - `pos_tagging`
  - `clean_text`: keeps words with at least `min_len` characters and filters out malformed tokens.
  - `convert_ngrams`
  - `remove_stopwords`
  - `correct_spelling` *(needs improvement)*
  - `tokenize_text`

  Example pipeline:
  ```python
  methods = [
      {"method": "lowercase"},
      {"method": "remove_urls"},
      {"method": "lemmatize_text"},
      {"method": "clean_text", "args": {"min_len": 1}},
      {"method": "remove_stopwords"}
  ]
  ```

The module is executed via the `preprocess.py` script, which loads data, applies preprocessing, and saves the results:

```bash
python preprocess.py --options config/options.yaml
```

It merges input data from various sources (insiders, outsiders, minors) using `merge_data`, and processes the `_title_` and `_summary_` columns (concatenated). Update the code if input parquet formats change. The pipeline uses parallelization when possible (configurable via `use_dask`).

#### Configuration (`options.yaml`)
- `use_dask`: Enable parallel processing.
- `subsample`: Limit dataset size for faster runs or debugging.
- `pipe`: List of preprocessing operations to apply (as shown above).
- `dir_*`: Paths to data, metadata, stopwords, ngrams, vocabulary, and output files.
- `use_stopwords`, `use_ngrams`: Specify specific files or `"all"` to load all available.

See the commented `options.yaml` file for detailed descriptions of each setting.

##### Data Directory
Configured via `options.yaml`, defaults to `/app/data`, or `/app/data/data.zip`.
```
/app/data o /app/data/data.zip
├───ngrams
│    ├───ngrams.txt
│    └───proposed_ngrams.txt
├───stopwords
│    ├───common_stopwords.txt
│    ├───administración.txt
│    └───municipios.txt
├───vocabulary
│   └───vocabulary.txt
└───metadata
    ├───insiders.parquet
    ├───outsiders.parquet
    └───minors.parquet
```

### Topic models
Contains a series of topic model implementations, wrapped under the common class ``BaseModel``.  It can be adapted by changing the data source and the models to use.

#### Models:
- **BaseModel**\
    A base model for all others. You need to specify where to save it, number of topics, etc. There are three functions that need to be implemented in each of the extending models, which are `_model_train`, `_model_predict`, and `load_model`.
    It also contains methods to save the documents, save and load topic-keys, doc-topics, etc.
- **BERTopic**\
    The training and prediction follow the scheme from https://maartengr.github.io/BERTopic/algorithm/algorithm.html
    The model used for Spanish, Catalan, Galician, and Basque is `paraphrase-multilingual-MiniLM-L12-v2`.
- **Mallet**\
    It's a wrapper for Mallet, you just need to provide the bin/mallet location. The rest of the setup is the same as the regular Mallet.
- **Tomotopy**\
    Uses https://bab2min.github.io/tomotopy/v/en/
    There are two versions: **tomotopyLDA** and **tomotopyCT**. The implementation is similar.
- **NMF**\
    It's a simple implementation of NMF that uses a TF-IDF vectorizer or CountVectorizer.
- **Gensim**\
    The basic Gensim model that I use in the test to have a baseline.

- **topic_models_test.py**
Main script. You need to pass the --options parameter with the yaml where all the settings are. First, all models are executed and then the model quality information is printed (execution times, PMI, etc.)

### Embeddings
Provides the Embedder class to compute embeddings using Word2Vec or BERT. These embeddings are used internally for topic labeling and semantic representations of topics.

### Graph
Builds and visualizes a semantic graph using Battacharyya distances between topic distributions. Community detection is performed using the Louvain algorithm, and node layout is computed with ForceAtlas2. Graphs include visual overlays of communities and key metrics.

> **Note:** The Graph module requires a separate Docker container with Python 3.8, as ForceAtlas2 is not compatible with Python 3.9 and above.
> This is distinct from the Docker environment used for preprocessing and training topic models.

#### Build and Run (from `Graph` directory):

```bash
# Build Docker image
docker build -t np_graphs .
```

```bash
# Run graph construction
docker run --rm \
    -v /absolute/path/to/models:/data/source \
    np_graphs \
    python3 build_graph.py \
    --path_model /data/source/<model_directory_name>
```

### Utils
Here you will find functions with various purposes, such as parallelizing or loading data.

## Output Structure

After running all modules, the following output directory structure is generated:

```python
model
model
├── graphs
│   ├── draw_graph_communities_communities.csv # CSV file containing community assignments for each node in the graph.
│   ├── draw_graph_communities.png             # PNG image visualizing the graph with community overlays.
│   ├── draw_graph_forceatlas2.png             # PNG image visualizing the graph layout using the ForceAtlas2 algorithm.
│   ├── draw_graph_lcc.png                     # PNG image visualizing the largest connected component of the graph.
│   ├── graph_edges.csv                        # CSV file containing the edges of the graph with source and target nodes.
│   └── graph_nodes.csv                        # CSV file containing the nodes of the graph with their attributes.
├── infer_data                     # Contains data and outputs related to the inference process
│   ├── corpus.txt                 # Raw text data used for inference (documents not included in the training corpus)
│   ├── corpus_inf.mallet          # Processed input for inference in Mallet format
│   ├── doc-topics-inf.txt         # Document-topic distribution after inference
│   ├── s3.npz                     # NumPy file storing S3 metric data 
│   └── thetas.npz                 # NumPy file storing inferred topic distributions
├── model_data                     # Stores model-related data and outputs
│   ├── TMmodel                    # Directory for various model-related files
│   │   ├── alphas.npy             # NumPy array storing alpha parameters (topics' size)
│   │   ├── alphas_orig.npy        # Original alpha parameters (in case modifications to the alphas file are made)
│   │   ├── betas.npy              # NumPy array storing beta parameters
│   │   ├── betas_ds.npy           # Downsampled beta parameters
│   │   ├── betas_orig.npy         # Original beta parameters
│   │   ├── distances.npz          # NumPy file storing the sparse matrix of document similarities.
│   │   ├── distances.txt          # Text file containing the string representation of the top similarities between each pair of documents.
│   │   ├── edits.txt              # Text file for documenting edits
│   │   ├── ndocs_active.npy       # NumPy array storing the number of active documents
│   │   ├── pyLDAvis.html          # HTML file for visualizing topic models (PyLDAvis)
│   │   ├── s3.npz
│   │   ├── thetas.npz             # NumPy file storing theta parameters (document-topic distribution)
│   │   ├── thetas_orig.npz        # Original theta parameters
│   │   ├── topic_coherence.npy    # NumPy array storing topic coherence scores
│   │   ├── tpc_coords.txt         # Text file storing topic coordinates
│   │   ├── tpc_descriptions.txt   # Text file for topic descriptions
│   │   ├── tpc_embeddings.npy     # NumPy array storing the Word2Vec embeddings of the topic descriptions.
│   │   ├── tpc_labels.txt         # Text file storing curated topic labels
│   │   └── vocab.txt              # Text file storing vocabulary
│   ├── corpus_train.mallet        # Training corpus in Mallet format
│   ├── corpus_train.txt           # Training corpus in text format
│   ├── diagnostics.xml            # XML file for model diagnostics obtained from Mallet
│   ├── dictionary.gensim          # Gensim dictionary file
│   ├── doc-topics.txt             # Document-topic distribution after training
│   ├── inferencer.mallet          # Mallet inferencer file
│   ├── model.pickle               # Pickle file storing the trained model
│   ├── topic-keys.json            # JSON file storing topic keys
│   ├── topic-keys.txt             # Text file storing topic keys
│   ├── topic-report.xml           # XML file for topic report
│   ├── vocab_freq.txt             # Text file storing vocabulary frequency
│   ├── vocabulary.txt             # Text file storing vocabulary
│   └── word-topic-counts.txt      # Text file storing word-topic counts
├── train_data                     # Contains data used for training the model
│   ├── corpus.mallet              # Corpus in Mallet format for training
│   ├── corpus.txt                 # Raw text data for training
│   ├── corpus_aux.txt             # Auxiliary text file for the corpus
│   ├── import.pipe                # Pipe file for importing inference data
│   └── train.config               # Configuration file for training
└── trainconfig.json               # JSON file containing the training configuration
```

## Docker
Preprocessing and topic modeling can be run in a Docker container with Python 3.11 and CUDA support.

### Build Image:
```
docker build -t image_name .
```

### Run:
```
docker run --gpus all --rm -it -v /path/to/data:/app/data image_name
```
In this command, the data directory of the host machine is mounted into the Docker container at the /app/data location.

## Main scripts
Utility scripts used to generate final topic models, graphs, metadata, and evaluations:

- `generate_graphs.py`: Graphs for all models in a directory  
- `generate_hierarchy.py`: Treemap of topic hierarchy per CPV code  
- `generate_users_eval.py`: Excel sheets for topic evaluation  
- `get_info_index_bsc.py`: Generates topic & objective JSON for BSC API  
- `predict_objective_with_s3.py`: Predicts missing objectives using the S3 metric  
- `train_tm.py`: Trains topic models per CPV or globally

## Acknowledgements

This work has received funding from the NextProcurement European Action (grant agreement INEA/CEF/ICT/A2020/2373713-Action 2020-ES-IA-0255).

<p align="center">
  <img src="static/Images/eu-logo.svg" alt="EU Logo" width="200" style="margin-right: 20px;">
  <img src="static/Images/nextprocurement-logo.png" alt="Next Procurement Logo" width="200">
</p>