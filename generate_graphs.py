import subprocess
import time
import pathlib
from tqdm import tqdm

path_models = pathlib.Path("/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Backend-Dockers/data/source/cpv_models")

for directory in tqdm(path_models.iterdir()):
    if not directory.is_dir():
        continue

    t_start = time.perf_counter()
    print(f"Processing {directory.name}...")

    subprocess.run([
        "docker", "run", "--rm",
        "-v", f"{str(path_models)}:/data/source",
        "np_graphs",
        "python3", "build_graph.py",
        "--path_model", f"/data/source/{directory.name}"
    ])

    t_end = time.perf_counter()
    print(f"Finished {directory.name} in {t_end - t_start:.2f} seconds\n")