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

[![](https://img.shields.io/badge/lang-es-red)](README.md)
