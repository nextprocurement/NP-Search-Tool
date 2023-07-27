import io
import zipfile
from collections import Counter
from logging import Logger
from pathlib import Path
from typing import List, Union

import pandas as pd
import regex


def fill_na(cell, fill=[]):
    """
    Fill elements in pd.DataFrame with `fill`.
    """
    if hasattr(cell, "__iter__"):
        if isinstance(cell, str):
            if cell == "nan":
                return fill
            return cell
        nas = []
        for el in cell:
            if el == "nan" or pd.isna(el):
                nas.append(True)
            else:
                nas.append(False)
        if all(nas):
            return fill
        return cell
    if pd.isna(cell):
        return fill
    return cell


def merge_data(
    dir_data: Union[str, Path],
    dir_text_metadata: Union[str, Path],
    merge_dfs: List[str] = ["minors", "insiders", "outsiders"],
    logger: Logger = None,
):
    """
    Merge original data parquet files into single dataframe
    """
    dir_data = Path(dir_data)
    dfs = []

    if dir_data.suffix == ".zip":
        # If it's a zip file, create a zipfile.Path and find the specific folder within it.
        with zipfile.ZipFile(dir_data, "r") as zip_ref:
            all_files = zip_ref.namelist()
            for d in merge_dfs:
                file_path = f"metadata/{d}.parquet"
                if file_path in all_files:
                    pq_file = io.BytesIO(zip_ref.read(file_path))
                    dfs.append(pd.read_parquet(pq_file))
                elif logger:
                    logger.warning(
                        f"File {d}.parquet does not exist in the zip file, skipping."
                    )
                else:
                    continue
    else:
        for d in merge_dfs:
            file_path = dir_data.joinpath(f"metadata/{d}.parquet")
            if file_path.exists():
                dfs.append(pd.read_parquet(file_path))
            elif logger:
                logger.warning(f"File {d}.parquet does not exist, skipping.")
            else:
                continue
    if not dfs:
        if logger:
            logger.error("No dataframes to merge.")
        return

    # Unify texts from all sources
    dfs_text = []
    for df in dfs:
        # Reset index and rename to common identifier
        index_names = df.index.names
        orig_cols = df.columns
        df.reset_index(inplace=True)
        df["identifier"] = df[index_names].astype(str).agg("/".join, axis=1)
        # df.drop(index_names, inplace=True, axis=1)
        df.set_index("identifier", inplace=True)
        df = df[orig_cols]

        # Select text columns and rename them
        join_str = lambda x: ".".join([el for el in x if el])
        joint_cnames = {join_str(c): c for c in df.columns}
        reverse_joint_cnames = {v: k for k, v in joint_cnames.items()}

        text_cols = sorted(
            [v for k, v in joint_cnames.items() if "summary" in k or "title" in k]
            # or "TenderResult.Description" in k
            # or "TenderingProcess.TenderSubmissionDeadlinePeriod.Description" in k
        )

        df_text = df.loc[:, text_cols]
        use_cols = [reverse_joint_cnames[c].split(".", 1)[-1] for c in text_cols]
        df_text.columns = use_cols
        df_text["text"] = (
            df_text[["title", "summary"]]
            .applymap(fill_na, fill=None)
            .agg(lambda x: ". ".join([el for el in x if el]), axis=1)
        )

        dfs_text.append(df_text)

    # Concatenate and save as unique DataFrame
    df_text = pd.concat(dfs_text)[["title", "summary", "text"]]
    df_text.to_parquet(dir_text_metadata, engine="pyarrow")


def filter_common_rare_words(
    corpus: List[str], topn: int = 0, min_count: int = 1, max_df: float = 0.7
):
    """
    Filters very common and very rare words from a corpus.

    Parameters
    ----------
        corpus (list):
            List of lists of strings, where each element represents a document in the corpus.
        topn (int):
            Number of top words in entire corpus to remove.
        min_count (int):
            Minimum number of occurrences required for a word to be included.
        max_df (float):
            Maximum frequency of a word required for it to be included.

    Returns
    -------
        filtered_corpus (list):
            List of list of strings, where each elements represents a filtered document.
    """
    word_counts = Counter()
    corpus_word_appearance = Counter()

    for c in corpus:
        word_counts.update(Counter(c))
        corpus_word_appearance.update(Counter(set(c)))

    # Remove top n
    [word_counts.pop(w[0]) for w in word_counts.most_common(topn)]

    # Remove words that don't appear in at least min_df
    for w, c in corpus_word_appearance.most_common()[::-1]:
        if c > min_count:
            break
        else:
            corpus_word_appearance.pop(w)

    # Remove words that appear in more than max_df documents
    for w, c in corpus_word_appearance.most_common():
        if c <= len(corpus) * max_df:
            break
        else:
            corpus_word_appearance.pop(w)

    # Words to use
    vocabulary = word_counts.keys() & corpus_word_appearance.keys()

    # Filter corpus
    filtered_corpus = []
    for c in corpus:
        filtered_corpus.append([w for w in c if w in vocabulary])

    return filtered_corpus, vocabulary


def find_sub_strings(text: str, str_len: int = 3):
    """
    Get a list of all combinations of `str_len` chars that appear in a text.
    """
    pattern = r"(?=([a-zA-Z\u00C0-\u024F\-\·\.\']{" rf"{str_len}" r"}))"
    return regex.findall(pattern, text)


# Conjugation rules
rules_irreg = [
    # Irreg
    (
        r"^(estuviéramos|estuviéremos|estuviésemos|estuvieseis|estuvierais|estuvisteis|estuviereis|estaríamos|estuvieron|estuvieras|estuvieses|estuviesen|estuvieran|estuvieres|estuvieren|estábamos|estuviera|estuvimos|estuviere|estuviese|estaríais|estaremos|estuviste|estarías|estaréis|estabais|estarían|estabas|estarás|estemos|estaban|estando|estarán|estamos|estaría|estuvo|estaré|estéis|estado|estaba|estáis|estará|estuve|estad|estoy|estás|estar|estés|estén|están|está|esté)$",
        "estar",
    ),
    (
        r"^(hubiésemos|hubiéremos|hubiéramos|hubisteis|hubiereis|habríamos|hubierais|hubieseis|hubieran|hubieras|hubiesen|hubieren|habremos|hubieses|habiendo|habríais|hubieron|hubieres|habíamos|hubiese|habrían|habréis|hayamos|habrías|hubiste|habíais|hubiere|hubiera|hubimos|habría|habido|habían|hayáis|habrás|habías|habrán|habéis|hayan|hayas|habré|había|habrá|haber|hemos|hube|hubo|haya|has|hay|han|he|ha)$",
        "haber",
    ),
    (
        r"^(hiciésemos|hiciéramos|hiciéremos|hiciereis|hicierais|hicisteis|hicieseis|hicieses|hacíamos|haríamos|hicieres|hicieren|hicieron|hiciesen|hicieras|hicieran|haciendo|hacemos|haríais|hacíais|hiciera|hiciste|haremos|hiciese|hagamos|hiciere|hicimos|hacías|hacían|hagáis|hacéis|harías|haréis|harían|hagas|hacía|hacen|hagan|haced|haces|harás|hecho|haría|harán|hacer|haga|hace|hará|hice|hizo|hago|haré|haz)$",
        "hacer",
    ),
    (
        r"^(fuéramos|fuisteis|seríamos|fuéremos|fuésemos|fueseis|fuerais|seremos|seríais|fuereis|seamos|seréis|fuesen|fuimos|fueses|serías|siendo|fueren|fueras|fueres|serían|fueran|éramos|fuiste|fueron|erais|fuera|somos|fuese|serás|fuere|serán|seáis|sería|seas|sean|seré|eras|será|sido|eran|eres|sois|soy|ser|sed|sea|son|fue|era|fui|es|sé)$",
        "ser",
    ),
    (
        r"^(fuéramos|fuisteis|fuéremos|fuésemos|fueseis|iríamos|vayamos|fuerais|fuereis|iremos|íbamos|fuesen|fuimos|fueses|fueren|fueras|fueres|vayáis|fueran|fuiste|iríais|fueron|fuera|yendo|vayan|fuese|irías|fuere|vamos|irían|ibais|iréis|vayas|vaya|irás|iría|iban|ibas|vais|irán|iba|van|vas|iré|irá|voy|ido|fue|fui|ve|ir|id|va)$",
        "ir",
    ),
    (
        r"^(dijéremos|dijisteis|dijéramos|dijésemos|decíamos|dijereis|dijerais|diciendo|diríamos|dijeseis|dijimos|dijeses|dijeren|dijeran|decíais|diremos|dijesen|dijiste|digamos|dijeron|decimos|dijeres|dijeras|diríais|digáis|dirían|dijese|diréis|decían|dijera|decías|dirías|dijere|digas|digan|decid|decía|dirán|dicho|diría|decís|dirás|dices|dicen|dirá|dice|digo|dijo|dije|diga|diré|di)$",
        "decir",
    ),
    (
        r"(isiéramos|isiéremos|isiésemos|isisteis|isiereis|erríamos|isierais|isieseis|erríais|erremos|isieras|isieses|isiesen|isieran|isieron|isieren|isieres|errías|isimos|erréis|isiste|errían|isiera|isiese|isiere|errás|ieres|ieren|ieran|erría|errán|ieras|errá|erré|iera|iere|ise|iso)$",
        "erer",
    ),
    (r"^(quepamos|quepáis|quepan|quepas|quepa|quepo)$", "caber"),
    (r"^sé$", "saber"),
    (
        r"(riésemos|riéremos|riéramos|riereis|rierais|rieseis|riendo|rieses|riesen|rieres|rieren|rieran|riamos|rieron|rieras|riese|riere|riáis|riera|ríes|ríen|rías|rían|río|rió|ríe|ría)$",
        "reir",
    ),
]

rules_unique = [
    # Valid unique
    (
        r"(?<=.+)(aríamos|aremos|aríais|ábamos|ásemos|áramos|asteis|áremos|aseis|arías|arais|arían|abais|aréis|areis|abas|aras|aran|aría|aren|ando|aste|asen|aban|arán|arás|ases|ares|aron|aré|aba|ara|ará|ado|are|ase|ar|ad)$",
        "ar",
    ),
    (r"[úü](emos|éis|en|es|an|as|o|é|a|e)$", "uar"),
    (r"u(amos|áis|an|as|ó|a|o|)$", "uar"),
    (r"[uú]s(amos|emos|éis|áis|en|as|es|an|e|o|a|é|ó)$", "usar"),
    (
        r"(?<!u)(eríamos|eríais|eremos|eréis|erían|erías|erán|ería|erás|eré|erá|ed)$",
        "er",
    ),
    (r"eí(amos|ais|as|an|a|)$", "er"),
    (
        r"b(iéramos|iéremos|iésemos|iereis|ierais|ieseis|isteis|ríamos|ieras|ríais|ieran|ieses|ieren|ieres|remos|íamos|iesen|iendo|ieron|iese|rían|iera|emos|iste|rías|íais|iere|amos|imos|réis|ría|ías|áis|ían|rás|rán|éis|ido|en|ed|as|an|ré|ía|es|rá|a|o|e)$",
        "ber",
    ),
    (
        r"r(eiríamos|eiríais|eísteis|eiremos|eíamos|eiréis|eirías|eirían|eímos|eíais|eirás|eíste|eiría|eirán|eiré|eirá|eído|eías|eían|eía|eír|eís|eíd|eí)$",
        "reir",
    ),
    (r"ir(íamos|emos|íais|éis|ías|ían|án|ás|ía|é|á)$", "ir"),
    (r"íd$", "ir"),
]

rules_non_unique = [
    (r"(ían|ías|ía|ió)$", "iar"),
    (r"u(emos|éis|é)$", "uar"),
    (r"tiendo$", "tender"),
]

rules_other = [
    # cer/cir
    (r"z(gamos|camos|gáis|cáis|cas|gas|gan|can|co|ca|ga|go|)$", "cer"),
    (r"z(gamos|camos|gáis|cáis|cas|gas|gan|can|co|ca|ga|go|)$", "cir"),
    # ar/ener
    (
        r"uv(iéramos|iésemos|iéremos|ieseis|isteis|ierais|iereis|ieres|ieses|ieron|ieran|iesen|ieren|ieras|iese|iere|iera|iste|imos|e|o)$",
        "ar",
    ),
    (
        r"uv(iéramos|iésemos|iéremos|ieseis|isteis|ierais|iereis|ieres|ieses|ieron|ieran|iesen|ieren|ieras|iese|iere|iera|iste|imos|e|o)$",
        "ener",
    ),
    # guer/guir
    (r"(?<!r)[gií]+(amos|áis|ais|ues|uen|as|an|a|o|)$", "er"),  # [guir]
    (r"(?<!r)[gií]+(amos|áis|ais|ues|uen|as|an|a|o|)$", "ir"),  # [guir]
    (r"g(amos|uen|áis|ues|as|an|a|o)$", "guir"),
    # er/ir
    (
        r"y(ésemos|éremos|éramos|ereis|eseis|erais|eses|eres|esen|amos|eras|eron|eran|eren|endo|era|ese|ere|áis|an|es|en|as|o|ó|a|e)$",
        "er",
    ),
    (
        r"y(ésemos|éremos|éramos|ereis|eseis|erais|eses|eres|esen|amos|eras|eron|eran|eren|endo|era|ese|ere|áis|an|es|en|as|o|ó|a|e)$",
        "ir",
    ),
    # cer/cir
    (
        r"c(iéremos|iésemos|iéramos|isteis|iereis|ieseis|ierais|iesen|ieron|ieres|ieses|ieren|ieran|ieras|emos|iese|iera|iste|iere|imos|éis|io|es|en|ió|ie|ís|e|o)$",
        "cer",
    ),
    (
        r"c(iéremos|iésemos|iéramos|isteis|iereis|ieseis|ierais|iesen|ieron|ieres|ieses|ieren|ieran|ieras|emos|iese|iera|iste|iere|imos|éis|io|es|en|ió|ie|ís|e|o)$",
        "cir",
    ),
    # ar/er/ir
    (
        r"(iéramos|iésemos|iéremos|ésemos|iereis|ierais|ísteis|isteis|ieseis|éremos|éramos|ereis|iesen|eseis|iendo|ieses|ieren|ieran|erais|ieron|ieres|uemos|eamos|ieras|íamos|eras|eran|eren|eres|esen|amos|iere|imos|emos|eses|ímos|íste|uéis|eron|iera|endo|eáis|iese|iste|íais|uen|ían|igo|áis|eis|ese|ean|ido|ere|eas|ues|éis|ais|ías|era|ué|ió|ea|an|ía|as|io|es|en|ís|id|ó|a|é|e|í|o)$",
        "ar",
    ),
    (
        r"(iéramos|iésemos|iéremos|ésemos|iereis|ierais|ísteis|isteis|ieseis|éremos|éramos|ereis|iesen|eseis|iendo|ieses|ieren|ieran|erais|ieron|ieres|uemos|eamos|ieras|íamos|eras|eran|eren|eres|esen|amos|iere|imos|emos|eses|ímos|íste|uéis|eron|iera|endo|eáis|iese|iste|íais|uen|ían|igo|áis|eis|ese|ean|ido|ere|eas|ues|éis|ais|ías|era|ué|ió|ea|an|ía|as|io|es|en|ís|id|ó|a|é|e|í|o)$",
        "er",
    ),
    (
        r"(iéramos|iésemos|iéremos|ésemos|iereis|ierais|ísteis|isteis|ieseis|éremos|éramos|ereis|iesen|eseis|iendo|ieses|ieren|ieran|erais|ieron|ieres|uemos|eamos|ieras|íamos|eras|eran|eren|eres|esen|amos|iere|imos|emos|eses|ímos|íste|uéis|eron|iera|endo|eáis|iese|iste|íais|uen|ían|igo|áis|eis|ese|ean|ido|ere|eas|ues|éis|ais|ías|era|ué|ió|ea|an|ía|as|io|es|en|ís|id|ó|a|é|e|í|o)$",
        "ir",
    ),
    # er/ir/der
    (r"dr(íamos|emos|íais|ías|éis|ían|ás|ía|án|é|á)$", "er"),
    (r"dr(íamos|emos|íais|ías|éis|ían|ás|ía|án|é|á)$", "ir"),
    (r"dr(íamos|emos|íais|ías|éis|ían|ás|ía|án|é|á)$", "der"),
]

# """
# Regex pattern that identifies words that may have the following characters:
# - letters: latin letters (including diacritics)
# - numbers
# - special characters: dash, forward and backward slash and vertical bar

# Word consist of a minimum of `min_len` (at least 2) characters with these restrictions:
# - starts with a letter
# - ends with letter or number
# - up to 3 consecutive numbers before another non-numeric character appears
# - words can have multiple special characters, but not consecutive
# """
# pattern = (
#     f"(?<![a-zA-Z\u00C0-\u024F\d\-])"
#     f"[a-zA-Z\u00C0-\u024F]"
#     f"(?:[a-zA-Z\u00C0-\u024F]|(?!\d{{4}})[\d]|[-](?![-])){{{min_len - 1},}}"
#     f"(?<![-])[a-zA-Z\u00C0-\u024F\d]?"
#     f"(?![a-zA-Z\u00C0-\u024F\d])"
# )

# (?<![a-zA-Z\u00C0-\u024F\d\-])[a-zA-Z\u00C0-\u024F](?:[a-zA-Z\u00C0-\u024F]|(?!\d{4})[\d]|[-](?![-])){2,}(?<![-])[a-zA-Z\u00C0-\u024F\d]?(?![a-zA-Z\u00C0-\u024F\d])
