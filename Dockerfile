# Use official Python 3.8 image with CUDA support
FROM nvidia/cuda:11.6.2-base-ubuntu20.04

# Set environment variables
ENV TZ=Europe/Madrid
ENV DEBIAN_FRONTEND=noninteractive

# Install git
RUN apt-get -y update
RUN apt-get -y install git

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

# Install additional dependencies for HDBSCAN
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Install HDBSCAN
# RUN pip install --no-cache-dir hdbscan
RUN pip install hdbscan

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the requirements
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# RUN echo "cache buster: $(date)" > cache_buster
RUN pip install -r requirements.txt
RUN python -m pip install "dask[dataframe]" --upgrade
RUN python -m spacy download es_dep_news_trf
RUN python -m spacy download es_core_news_lg

# Copy the config/ directory
COPY config/ config/
# Copy the src/ directory
COPY src/ src/
# Copy all .py files
COPY *.py .

# Set the entrypoint command
# CMD ["python", "preprocess.py"]
CMD ["/bin/bash"]


# docker build -t next_proc .

# docker run \
#     --gpus all \
#     --name preprocess \
#     --rm \
#     -v /path/to/your/local/data:/app/data \
#     next_proc

# docker run \
#     --gpus all \
#     --name preprocess \
#     --rm \
#     -v ./data/:/app/data/ \
#     next_proc
