# NextProcurement

# Funcionalidad
Este proyecto proporciona un script de preprocesamiento para cargar textos a partir de metadatos disponibles en formato parquet. Los textos son preprocesados para todas las actividades siguientes (como el modelado de tópicos y otros).

# Ejecución
Para ejecutar el script, basta con usar el comando siguiente:
```
python preprocess.py [--options config/options.yaml]
```

Por defecto, se carga el fichero de configuración `config/options.yaml`. Este fichero contiene todas las opciones y configuraciones necesarias para la ejecución del script.

# Configuración
La configuración del script de preprocesamiento se realiza a través del fichero _options.yaml_. En este fichero se pueden establecer diversas opciones como:

- use_dask: si se desea usar Dask para el procesamiento en paralelo.
- subsample: tamaño de la submuestra a utilizar en el preprocesamiento.
- pipe: operaciones de procesamiento a aplicar.
- dir_*: directorios para datos, metadatos, stopwords, ngrams, vocabulario y ficheros de salida.
- use_stopwords y use_ngrams: especificar ficheros específicos a cargar o "all" para cargar todos los ficheros del directorio correspondiente.

Para más detalles sobre cada opción, se puede consultar el fichero _options.yaml_ donde se encuentran comentarios explicativos.

# Docker
El fichero Dockerfile permite construir una imagen de Docker basada en una imagen de Ubuntu con CUDA. En esta imagen se instala Python 3.11 y todas las dependencias necesarias, incluyendo SentenceTransformer y otros modelos de lenguaje.

Para construir la imagen de Docker, se puede utilizar el comando siguiente:
```
docker build -t image_name .
```

Una vez construida la imagen, se puede crear un contenedor de Docker para ejecutar el script de preprocesamiento. El comando siguiente muestra cómo hacerlo:
```
docker run --gpus all --rm -it -v /path/to/data:/app/data image_name
```
En este comando, se monta el directorio de datos de la máquina host en el contenedor de Docker en la ubicación /app/data.

[![](https://img.shields.io/badge/lang-en-red)](README.en.md)
