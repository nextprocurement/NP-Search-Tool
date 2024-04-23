import pandas as pd
from langdetect import detect

def det(x: str) -> str:
    """
    Detects the language of a given text

    Parameters
    ----------
    x : str
        Text whose language is to be detected

    Returns
    -------
    lang : str
        Language of the text
    """

    try:
        lang = detect(x)
    except:
        lang = 'Other'
    return lang

def main():
    # Leer el archivo parquet
    path = '/export/usuarios_ml4ds/lbartolome/NextProcurement/data/processed_10_abr/md2.parquet'
    df_preprocesado = pd.read_parquet(path)

    # Agregar columna de idioma al DataFrame
    df_preprocesado['lang'] = df_preprocesado['raw_text'].apply(det)

    # Filtrar por idioma 'es'
    df_es = df_preprocesado[df_preprocesado['lang'] == 'es']
    print(len(df_es))
    # Guardar el df filtrado como parquet
    df_es.to_parquet('/export/usuarios_ml4ds/cggamella/NP-Search-Tool/sample_data/md2_es.parquet', index=True)

    # Filtrar por idioma 'ca'
    df_cat = df_preprocesado[df_preprocesado['lang'] == 'ca']
    print(len(df_cat))
    # Guardar el df filtrado como parquet
    df_cat.to_parquet('/export/usuarios_ml4ds/cggamella/NP-Search-Tool/sample_data/md2_cat.parquet', index=True)

    print("Archivos guardados correctamente.")

if __name__ == "__main__":
    main()
