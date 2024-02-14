# NextProcurement

## Funcionalidad

Este proyecto proporciona un script de preprocesamiento para cargar textos a partir de metadatos disponibles en formato parquet. Los textos son preprocesados para todas las actividades siguientes (como el modelado de tópicos y otros).

## Ejecución

Para ejecutar el script, basta con usar el comando siguiente:

```bash
python preprocess.py [--options config/options.yaml]
```

Por defecto, se carga el fichero de configuración `config/options.yaml`. Este fichero contiene todas las opciones y configuraciones necesarias para la ejecución del script.

## Configuración

La configuración del script de preprocesamiento se realiza a través del fichero _options.yaml_. En este fichero se pueden establecer diversas opciones como:

- use_dask: si se desea usar Dask para el procesamiento en paralelo.
- subsample: tamaño de la submuestra a utilizar en el preprocesamiento.
- pipe: operaciones de procesamiento a aplicar.
- dir_*: directorios para datos, metadatos, stopwords, ngrams, vocabulario y ficheros de salida.
- use_stopwords y use_ngrams: especificar ficheros específicos a cargar o "all" para cargar todos los ficheros del directorio correspondiente.

Para más detalles sobre cada opción, se puede consultar el fichero _options.yaml_ donde se encuentran comentarios explicativos.

## Estructura de ficheros

### Datos

El directorio definido en el fichero de configuración. Por defecto es: **_/app/data_**, aunque este puede ser un zip (**_/app/data/data.zip_**). Dentro de ese directorio se encontrarán otros subdirectorios (stopwords, vocabulary, ngrams) o un fichero `.zip` con la misma estructura. Por ejemplo:

```bash
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

### app/src

Contiene las diversas funcionalidades de la aplicación, divididas en Preprocessor y TopicModels.

#### Preprocessor

En src/Preprocessor están todas las clases que se utilizan, y el `preprocess.py` es el script para procesar los parquets usando el pipeline. El pipeline por defecto identifica el lenguaje del texto, lo normaliza y preprocesa.
Algunas consideraciones:

- Lemmatizer\
    Se puede utilizar uno propio para castellano (que usa las reglas del idioma) o los de Spacy y similares (algo más rápidos). Por defecto se usan los de Spacy (_options.yaml/vocabulary:false_).

- NgramProcessor\
    Contiene diversas funciones para buscar, filtrar, sustituir ngrams o comprobar la validez de los mismos.

- TextProcessor\
    Genera un pipeline de preprocesamiento para los textos. Utiliza todo lo descrito en los puntos anteriores. Se pueden pasar los ngrams, stopwords o vocabulario. Los elementos del pipeline disponible son:
  - lowercase
  - remove_tildes
  - remove_extra_spaces
  - remove_punctuation
  - remove_urls
  - lemmatize_text
  - pos_tagging
  - clean_text: selecciona palabras que tienen al menos min_len caracteres. Filtra palabras que no empiezan con una letra, no terminan con una letra o número, tengan múltiples caracteres especiales…
  - convert_ngrams
  - remove_stopwords
  - correct_spelling: (este habría que mejorarlo)
  - tokenize_text

    Para crear un TextProcessor hay que hacerlo usando una lista de diccionarios, porque en algunos casos es necesario usar parámetros. Por ejemplo:

    ```python
    methods=[
        {"method": "lowercase"},
        {"method": "remove_urls"},
        {"method": "lemmatize_text"},
        {"method": "clean_text", "args": {"min_len": 1}},
        {"method": "remove_stopwords"}
    ]
    ```

- **preprocess.py**\
    Script principal. Hay que pasar el parámetro –options con el yaml donde está toda la configuración. El método `merge_data` utilizado al principio se utiliza para unificar los datos de las distintas fuentes (insiders, outsiders y minors). Las columnas utilizadas son las de texto por defecto ([_title_, _summary_]) y las concatena en _text_. En caso de que cambie el formato de los parquets, habrá que cambiar esto.\
    Se utilizan métodos de paralelización todo lo posible, aunque esto debe ajustarse en función de la disponibilidad de los equipos.

### Topic models

Contiene los elementos usados en `topic_models_test.py` que sería un comparador de todos los modelos con un subset. Se puede adaptar cambiando el origen de los datos y los modelos a usar.

#### Modelos

- BaseModel\
    Modelo que sirve de base para todos los demás. Hay que especificar la ubicación para guardarlo, número de tópicos, etc. Hay tres funciones que hay que implementar en cada uno de los modelos que lo extienden, que son `_model_train`, `_model_predict` y `load_model`.
    Contiene, además, métodos para guardar los documentos, guardar y cargar topic-keys, doc-topics, etc.
- BERTopic\
    El entrenamiento y el predict siguen el esquema de <https://maartengr.github.io/BERTopic/algorithm/algorithm.html>
    El modelo usado para castellano, catalán, gallego y vasco es `paraphrase-multilingual-MiniLM-L12-v2`.
- Mallet\
    Es un wrapper para Mallet, simplemente hay que pasarle la ubicación del bin/mallet. El resto de la configuración es igual que el Mallet normal.
- Tomotopy\
    Usa <https://bab2min.github.io/tomotopy/v/en/>
    Hay dos versiones: **tomotopyLDA** y **tomotopyCT**. La implementación es similar.
- NMF\
    Es una implementación sencilla de NMF que usa vectorizador de TF-IDF o CountVectorizer.
- Gensim\
    El modelo básico de gensim que uso en el test para tener un baseline.

- **topic_models_test.py**\
Script principal. Hay que pasar el parámetro --options con el yaml donde está toda la configuración. Primero se ejecutan todos los modelos y luego se imprime la información de la calidad de los modelos (tiempos de ejecución, PMI, etc.)

### Utils

Aquí se encuentran funciones con diversas finalidades, como paralelizar o cargar datos.

## Docker

El fichero Dockerfile permite construir una imagen de Docker basada en una imagen de Ubuntu con CUDA. En esta imagen se instala Python 3.11 y todas las dependencias necesarias, incluyendo SentenceTransformer y otros modelos de lenguaje.

Para construir la imagen de Docker, se puede utilizar el comando siguiente:

```bash
docker build -t image_name .
```

Una vez construida la imagen, se puede crear un contenedor de Docker para ejecutar el script de preprocesamiento. El comando siguiente muestra cómo hacerlo:

```bash
docker run --gpus all --rm -it -v /path/to/data:/app/data image_name
```

Para que funcione el topiclabeller, el servicio se tiene que ejecutar de la siguiente manera:

donde ``.venv`` es un fichero dónde está definda una clave de OpenAI, tal que ``OPENAI_API_KEY=XXXX``.

En este comando, se monta el directorio de datos de la máquina host en el contenedor de Docker en la ubicación /app/data.

[![](https://img.shields.io/badge/lang-en-red)](README.en.md)
