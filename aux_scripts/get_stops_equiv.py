import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import bs4
import ctypes
import requests
import time
from bs4 import BeautifulSoup
from Levenshtein import distance as levenshtein_distance
from itertools import combinations

###############################################################################
# Global variables
###############################################################################
host = 'http://dle.rae.es'
referer = 'http://www.rae.es/'
parser = 'lxml'
s = requests.Session()

###############################################################################
# Funciones
###############################################################################
def print_first_k_elements(my_dict,k):
    count = 0
    for key, value in my_dict.items():
        print(key, value)
        count += 1
        if count >= k:
            break
        
def agrupar_palabras(palabras):
    # Agrupar por las tres primeras letras
    agrupados_por_prefijo = {}
    # Descomponer palabras compuestas y agrupar por los prefijos de cada sub-palabra
    for palabra in palabras:
        sub_palabras = palabra.split('_')  # Divide la palabra en sub-palabras basadas en '_'
        for sub_palabra in sub_palabras:
            prefijo = sub_palabra[:3] if len(sub_palabra) >= 3 else sub_palabra
            if prefijo not in agrupados_por_prefijo:
                agrupados_por_prefijo[prefijo] = []
            agrupados_por_prefijo[prefijo].append(palabra)

    # Usamos numpy para crear un array de las palabras para iteraciones más rápidas
    palabras_array = np.array(palabras)
    n = len(palabras)
    agrupados_por_similitud = []

    # Matriz de distancia para optimizar las comparaciones
    """
    for i in range(n):
        for j in range(i + 1, n):
            if levenshtein_distance(palabras_array[i], palabras_array[j]) <= 2:
                agrupados_por_similitud.append((palabras_array[i], palabras_array[j]))
    """
    agrupados_por_similitud = [(word1, word2) for word1, word2 in combinations(palabras_array, 2) if levenshtein_distance(word1, word2) <= 2]


    return agrupados_por_prefijo, agrupados_por_similitud

def palabras_que_empiezan(lista_palabras):
    palabras_san = [palabra for palabra in lista_palabras if palabra.startswith('san_') or palabra.startswith('pº') or 'nº' in palabra]
    return palabras_san
        
def diferir_en_una_letra(palabras):
    from collections import defaultdict

    def palabras_diferentes_en_una(palabra1, palabra2):
        """ Retorna True si las palabras difieren en exactamente una letra. """
        diferencias = 0
        for c1, c2 in zip(palabra1, palabra2):
            if c1 != c2:
                diferencias += 1
                if diferencias > 1:
                    return False
        return diferencias == 1

    # Agrupamos palabras por longitud
    palabras_por_longitud = defaultdict(list)
    for palabra in palabras:
        palabras_por_longitud[len(palabra)].append(palabra)

    pares = []

    # Comparamos solo palabras de la misma longitud
    for grupo in palabras_por_longitud.values():
        n = len(grupo)
        for i in range(n):
            for j in range(i + 1, n):
                if palabras_diferentes_en_una(grupo[i], grupo[j]):
                    pares.append((grupo[i], grupo[j]))

    return pares

def search(word):
    s.headers.update({'Upgrade-Insecure-Requests': '1'})
    s.headers.update({'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'})
    s.headers.update({'Referer': referer})
    s.headers.update({'Accept-Language': 'es-ES,es;q=0.8'})
    s.headers.update({'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.81 Safari/537.36'})
    s.headers.update({'Accept-Encoding': 'gzip, deflate, sdch'})

    s.cookies = requests.utils.cookiejar_from_dict({'cookies_rae': 'aceptada'})

    url1 = host + '/?w=' + word
    url2 = host + '/srv/search?w=' + word
    #print(url1)

    response = s.get(url1).text
    soup = BeautifulSoup(response, parser)
    text = soup.get_text()
    is_in_dict = f"La palabra {word} no está en el Diccionario"
    if is_in_dict in text:
        return False
    else:
        return True
        
###############################################################################
# Leer el archivo con el vocabulario 
###############################################################################
print("Leyendo archivo con el vocabulario...")
path = "/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/models/Mallet/es_Mallet_all_150_topics/model_data/TMmodel/vocab.txt"

# Lista para almacenar las palabras del archivo
palabras_archivo = []

# Leer el archivo .txt y guardar las palabras en la lista
with open(path, 'r', encoding='utf-8') as f:
    palabras_archivo = f.read().split()
    
# Mostrar las palabras del archivo
print("Palabras del archivo:", len(palabras_archivo))


###############################################################################
# Agrupar las palabras por prefijo y similitud
###############################################################################
print("Agrupando palabras...")
agrupados_por_prefijo, agrupados_por_similitud = agrupar_palabras(palabras_archivo[:1000])

k=20
print("Agrupados por prefijo:")
print_first_k_elements(agrupados_por_prefijo, k=k)
print("######################")
print("Agrupados por similitud:", agrupados_por_similitud[:k])

agrupados_por_prefijo, agrupados_por_similitud = agrupar_palabras(palabras_archivo)

k=100
print("Agrupados por prefijo:")
print_first_k_elements(agrupados_por_prefijo, k=k)
print("######################")
print("Agrupados por similitud:", agrupados_por_similitud[:k])


###############################################################################
# Encontrar palabras que empiecen por san_, pº, nº
###############################################################################
print("Encontrando palabras que empiezan por san_, pº, nº...")
palabras_san = palabras_que_empiezan(palabras_archivo)
print(palabras_san)

###############################################################################
# Encontrar palabras que difieran en una letra
###############################################################################
print("Encontrando palabras que difieren en una letra...")
resultado = diferir_en_una_letra(palabras_archivo)

###############################################################################
# Procesar los resultados
###############################################################################
# Lista para almacenar las filas del DataFrame
data = []
# Lista para almacenar posibles stopwords
posibles_stopwords = []

# Procesar cada tupla en resultado
for palabra1, palabra2 in resultado:
    #print("La palabra1 es:\n", palabra1)
    #print("La palabra2 es:\n", palabra2)
    existe_1 = search(palabra1)
    #print("La palabra1 existe?", existe_1)
    existe_2 = search(palabra2)
    #print("La palabra2 existe?", existe_2)
  
    # Añadir al DataFrame si una existe y la otra no
    if (existe_1 and not existe_2) or (not existe_1 and existe_2):
        equivalencia = f"{palabra1}:{palabra2}" if not existe_1 else f"{palabra2}:{palabra1}"
        data.append({
            'columna1': palabra1,
            'columna2': palabra2,
            'existe_palabra1': existe_1,
            'existe_palabra2': existe_2,
            'equivalencias': equivalencia
        })
    # Si ninguna de las dos palabras existe, las añadimos a posibles_stopwords
    elif not existe_1 and not existe_2:
        posibles_stopwords.append(palabra1)
        posibles_stopwords.append(palabra2)

# Crear DataFrame
df = pd.DataFrame(data, columns=['columna1', 'columna2', 'existe_palabra1', 'existe_palabra2', 'equivalencias'])
# Añadir la columna de posibles_stopwords
df['posibles_stopwords'] = ', '.join(posibles_stopwords)

# Crear lista de equivalencias
lista_equivalencias = df['equivalencias'].tolist()

ruta_archivo_txt = '/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/rae_lista_equivalencias.txt'

# Abre el archivo de texto en modo escritura
with open(ruta_archivo_txt, 'w', encoding='utf-8') as archivo_txt:
    # Itera sobre cada elemento de la lista
    for palabra in lista_equivalencias:
        # Escribe la palabra en el archivo de texto con un salto de línea
        archivo_txt.write(palabra + '\n')
        
# Especifica la ruta donde quieres guardar el archivo Parquet
ruta = '/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/posible_lista_equivalencias.parquet'

# Guarda el DataFrame como un archivo Parquet en la ruta especificada
df.to_parquet(ruta)
