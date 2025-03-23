FROM python:3.8

WORKDIR /graphs

COPY . .

RUN mkdir -p /data/source

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "build_graph.py"]