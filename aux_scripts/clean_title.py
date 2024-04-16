import pathlib
import pandas as pd
import re
import bs4
import ctypes
import requests
import time
from bs4 import BeautifulSoup
from unidecode import unidecode
import time

host = 'http://dle.rae.es'
referer = 'http://www.rae.es/'
parser = 'lxml'
s = requests.Session()

path_processed = pathlib.Path(
    "/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/all_processed/md_sin_lote_es.parquet")
path_raw = pathlib.Path(
    "/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/all_processed/minors_insiders_outsiders_origen_sin_lot_info.parquet")
path_save = pathlib.Path(
    "/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/all_processed/minors_insiders_outsiders_origen_sin_lot_info_title_clean.parquet")

df_processed = pd.read_parquet(path_processed)
df_raw = pd.read_parquet(path_raw)
df = pd.merge(df_processed, df_raw, how='inner', on='id_tm')

# Remove Id licitación:.*?Órgano de Contratación
pattern = (
    r'|;\s*Número de\s+Estado: RES\s*'
    r'|;\s*Número de\s+Estado: EV\s*'
    r'|;\s*Número de\s*'
    r'|;\s*Id licitación:.*?Estado: RES\s*'
    r'|;\s*Id licitación:.*?Estado: En Plazo\s*'
    r'|;\s*Id licitación:.*?Estado: ADJ\s*'
    r'|;\s*Id licitación:.*?Estado: PUB\s*'
    r'|;\s*Id licitación:.*?Estado: EV\s*'
    r'|;\s*Id licitación:.*?Estado: ANUL\s*'
    r'|;\s*Id licitación:.*?Estado: PRE\s*'
    r'|;\s*Id licitación:.*?Estado: CERR\s*'
    r'|;\s*Id licitación:.*?Órgano de Contratación:.*?;\s*'
    r'|Id licitación:.*?Órgano de Contratación:.*?;'
    r'|;\s*Id licitación:.*?Importe.*?€;\s*'
    r'|;\s*Id licitación:.*?Importe.*?,\s*'
    r'|;\s*Órgano de Contratación:.*?Estado: PRE\s*'
    r'|;\s*Importe:.*?Estado: CERR\s*'
    r'|;\s*Importe:.*?Estado: BORR\s*'
    r'|;\s*Importe:.*?EUR;\s*'
    r'|;\s*Expediente:.*?Resuelta\s*'
    r'|;\s*Expediente:.*?En Plazo\s*'
    r'|;\s*Expediente:.*?En prazo\s*'
    r'|;\s*Expediente:.*?Adj\s*'
    r'|;\s*Expediente:.*?Órgano de Contratación:.*?;\s*'
    r'|Estado: PENDIENTE'
    r'|Estado: ABIERTA'
    r'|Estado:EN PLAZO'
    r'|;\s*Estado.*?ADJUDICACION\s*'
    r'|;\s*Estado.*?ADJUDICACIÓN\s*'
    r'|;\s*Estado: 1Abierta\s*'
    r'|;\s*Estado: 2Entransicion\s*'
    r'|;\s*Estado: 3Cerrada\s*'
    r'|;\s*Estado: En prazo\s*'
    r'|;\s*Estado: Anuncio Previo\s*'
    r'|;\s*Estado: Anul\s*'
    r'|;\s*Estado:\s+RES\s*'
    r'|;\s*Estado: Formalización de contratos\s*'
    r'|Estado: Apertura documentación técnica'
    r'|Estado: No formalizado'
    r'|Estado: Clasificación de ofertas'
    r'|Estado: Empresas seleccions'
    r'|;\s*Estado: ABIERTO\s*'
    r'|Estado:ADJ'
    r'|Estado:ANUL'
    r'|;\s*ALUACION\s*'
    r'|UELTA'
    r'|udicada'
    r'|aluación'
    r'|udicación'
    r'|de presentación'
    r'|ada'
    r'|uelto'
    r'|Pleno'
    r'|;\s*Importe:.*?;\s*'
    r'|licación en Perfil del Contratante'
    r'|Id licitación:\s*\+?\d+(\.\d+)?[A-Z\d\-]+[\s\n]*; Órgano de contratación: Servicio Andaluz de Salud;'
    r'|Órgano de Contratación:\s*[\w\s\',.\-\(\)àèéíòóúçÁÉÍÓÚÀÈÇ]+\s*;'
    r'|Id licitación:\s*\+?[A-Z\d\-]+[\s\n]*;.*?'
    r'|Expediente:\s*\d+\/\d+\(.*?\),\s*Entidad:.*?,\s*Importe:\s*\d+\.\d+\s*,'
    r'|Estado: Apertura documentación administrativa'
)

df['summary'] = df['summary'].str.replace(
    pattern, '', regex=True, flags=re.IGNORECASE)


non_empty_rows = df['summary'].str.strip() != ''
non_empty_df = df[non_empty_rows]
print(f"There are {len(non_empty_df)} / {len(df)} after removing 'Id licitación:.*?Órgano de Contratación:'")

print(f"-- Proceeding to keep only words in dictionary...")


def search(word):
    s.headers.update({'Upgrade-Insecure-Requests': '1'})
    s.headers.update(
        {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'})
    s.headers.update({'Referer': referer})
    s.headers.update({'Accept-Language': 'es-ES,es;q=0.8'})
    s.headers.update(
        {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.81 Safari/537.36'})
    s.headers.update({'Accept-Encoding': 'gzip, deflate, sdch'})

    s.cookies = requests.utils.cookiejar_from_dict({'cookies_rae': 'aceptada'})

    url1 = host + '/?w=' + word
    # print(url1)
    url2 = host + '/srv/search?w=' + word

    response = s.get(url1).text
    soup = BeautifulSoup(response, parser)
    text = soup.get_text()
    is_in_dict = f"La palabra {word} no está en el Diccionario"
    if is_in_dict in text:
        return False
    else:
        return True


time_start = time.time()
df["clean_title"] = df["title"].apply(
    lambda x: " ".join([x for x in x.split() if search(x)]))

print(f"-- -- Time elapsed cleaning title: {time.time() - time_start}")

df[['id_tm', 'origen', 'title', 'clean_title']].to_parquet(path_save)
