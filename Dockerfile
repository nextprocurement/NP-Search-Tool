# Use official ubuntu image with CUDA support
FROM nvidia/cuda:11.6.2-base-ubuntu20.04

# Set environment variables
ENV TZ=Europe/Madrid
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_HOME=/torch/
# ENV MALLET_HOME /app/Mallet

# Install Vim, Git, Java and Ant
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y vim git openjdk-8-jdk ant

# Install build dependencies for Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libbz2-dev \
    libenchant-dev \
    libffi-dev \
    libgdbm-dev \
    libncurses5-dev \
    libnss3-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    tzdata \
    zlib1g-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install Python 3.11.1
RUN wget https://www.python.org/ftp/python/3.11.1/Python-3.11.1.tgz && \
    tar xzf Python-3.11.1.tgz && \
    cd Python-3.11.1 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    ln -s /usr/local/bin/python3.11 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.11 /usr/local/bin/pip && \
    cd .. && \
    rm -rf Python-3.11.1.tgz Python-3.11.1

# Install other necessary dependencies
RUN apt-get update && apt-get install hunspell-es

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install additional dependencies for HDBSCAN
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install HDBSCAN
RUN pip install hdbscan

# Install pytorch
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install torch

# Install requirements
RUN pip install wheel setuptools pip --upgrade
RUN pip install -r requirements.txt
RUN pip install "dask[complete]"
#RUN python -m pip install "dask[dataframe]" --upgrade
RUN python -m spacy download es_dep_news_trf
RUN python -m spacy download es_core_news_lg

# Clone the Mallet repository
RUN git clone https://github.com/mimno/Mallet.git
# Change into the Mallet directory and build the Mallet project
RUN cd /app/Mallet && ant

# Download and cache the sentence transformer model
ARG MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"

# Download fasttext model
RUN mkdir models
RUN wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz -O /app/models/lid.176.ftz

# Copy the config/ directory
#COPY config/ config/
# Copy the src/ directory
#COPY src/ src/
# Copy all .py files
#COPY *.py ./
COPY . .
RUN python -c "import sys; print('\n'.join(sys.path))"

# Set the entrypoint command
#CMD ["python", "preprocess.py"]
CMD ["/bin/bash"]


# docker build -t next_proc .

# docker run \
#     --gpus all \
#     --name some_name \
#     --rm \
#     -it
#     -v /path/to/your/local/data:/app/data \
#     next_proc

# docker run \
#     --gpus all \
#     --name preprocess \
#     --rm \
#     -v ./data/:/app/data/ \
#     next_proc
