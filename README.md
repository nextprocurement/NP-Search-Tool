# NextProcurement

# Functionality
This project provides a preprocessing script to load texts from metadata available in parquet format. The texts are preprocessed for all subsequent activities (such as topic modeling and others).

# Execution
To execute the script, simply use the following command:
```
python preprocess.py [--options config/options.yaml]
```

By default, the configuration file `config/options.yaml` is loaded. This file contains all the necessary options and configurations for the execution of the script.

# Configuration
The configuration of the preprocessing script is done through the _options.yaml_ file. In this file, various options can be set such as:

- use_dask: if you want to use Dask for parallel processing.
- subsample: size of the subsample to use in preprocessing.
- pipe: processing operations to apply.
- dir_*: directories for data, metadata, stopwords, ngrams, vocabulary, and output files.
- use_stopwords and use_ngrams: specify specific files to load or "all" to load all files from the corresponding directory.

For more details on each option, you can refer to the _options.yaml_ file where explanatory comments are found.

# File Structure
## Data:
The directory defined in the configuration file. By default it is: **_/app/data_**, but this can be a zip (**_/app/data/data.zip_**). Within that directory, you will find other subdirectories (stopwords, vocabulary, ngrams) or a `.zip` file with the same structure. For example:
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
## app/src
Contains various functionalities of the application, divided into Preprocessor and TopicModels.

### Preprocessor:
In src/Preprocessor you will find all the classes that are used, and `preprocess.py` is the script to process the parquets using the pipeline. The default pipeline identifies the language of the text, normalizes it, and preprocesses it.
Some considerations:
- Lemmatizer\
    You can use a custom one for Spanish (that uses the language's rules) or the ones from Spacy and similar (somewhat faster). By default, Spacy ones are used (*options.yaml/vocabulary:false*).

- NgramProcessor\
    Contains various functions to search, filter, replace ngrams, or check their validity.

- TextProcessor\
    Generates a preprocessing pipeline for texts. It uses everything described in the previous points. You can pass the ngrams, stopwords, or vocabulary. The available pipeline elements are:
    - lowercase
    - remove_tildes
    - remove_extra_spaces
    - remove_punctuation
    - remove_urls
    - lemmatize_text
    - pos_tagging
    - clean_text: selects words that have at least min_len characters. Filters out words that don't start with a letter, don't end with a letter or number, have multiple special characters…
    - convert_ngrams
    - remove_stopwords
    - correct_spelling: (this needs improvement)
    - tokenize_text

To create a TextProcessor, you must do it using a list of dictionaries because, in some cases, parameters are needed. For example:
    ```
    methods=[
        {"method": "lowercase"},
        {"method": "remove_urls"},
        {"method": "lemmatize_text"},
        {"method": "clean_text", "args": {"min_len": 1}},
        {"method": "remove_stopwords"}
    ]
    ```
- **preprocess.py**\
    Main script. You must pass the –options parameter with the yaml where all the settings are. The `merge_data` method used at the beginning is used to unify the data from different sources (insiders, outsiders, and minors). The default text columns used are [_title_, _summary_] and they concatenate into text. If the format of the parquets changes, this will need to be modified.\
    Parallelization methods are used as much as possible, although this should be adjusted based on the availability of the equipment.

## Topic models
Contains the elements used in `topic_models_test.py` which would be a comparator of all models with a subset. It can be adapted by changing the data source and the models to use.

### Models:
- BaseModel
    A base model for all others. You need to specify where to save it, number of topics, etc. There are three functions that need to be implemented in each of the extending models, which are `_model_train`, `_model_predict`, and `load_model`.
    It also contains methods to save the documents, save and load topic-keys, doc-topics, etc.
- BERTopic\
    The training and prediction follow the scheme from https://maartengr.github.io/BERTopic/algorithm/algorithm.html
    The model used for Spanish, Catalan, Galician, and Basque is `paraphrase-multilingual-MiniLM-L12-v2`.
- Mallet\
    It's a wrapper for Mallet, you just need to provide the bin/mallet location. The rest of the setup is the same as the regular Mallet.
- Tomotopy\
    Uses https://bab2min.github.io/tomotopy/v/en/
    There are two versions: **tomotopyLDA** and **tomotopyCT**. The implementation is similar.
- NMF\
    It's a simple implementation of NMF that uses a TF-IDF vectorizer or CountVectorizer.
- Gensim\
    The basic Gensim model that I use in the test to have a baseline.

- **topic_models_test.py**
Main script. You need to pass the --options parameter with the yaml where all the settings are. First, all models are executed and then the model quality information is printed (execution times, PMI, etc.)

## Utils
Here you will find functions with various purposes, such as parallelizing or loading data.

# Docker
The Dockerfile allows to build a Docker image based on an Ubuntu image with CUDA. In this image Python 3.11 and all necessary dependencies are installed, including SentenceTransformer and other language models.

To build the Docker image, the following command can be used:
```
docker build -t image_name .
```

Once the image is built, a Docker container can be created to run the preprocessing script. The following command shows how to do it:
```
docker run --gpus all --rm -it -v /path/to/data:/app/data image_name
```
In this command, the data directory of the host machine is mounted into the Docker container at the /app/data location.


## Acknowledgements

This work has received funding from the NextProcurement European Action (grant agreement INEA/CEF/ICT/A2020/2373713-Action 2020-ES-IA-0255).

<p align="center">
  <img src="static/Images/eu-logo.svg" alt="EU Logo" width="200" style="margin-right: 20px;">
  <img src="static/Images/nextprocurement-logo.png" alt="Next Procurement Logo" width="200">
</p>