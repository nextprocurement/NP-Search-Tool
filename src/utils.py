import json
import logging
from pathlib import Path
from typing import Dict, List, Union

import regex
from pandas import Series


# Load item_list
def load_item_list(
    dir_item_list: Path,
    use_item_list: Union[str, List[str]] = [
        "administración",
        "municipios",
        "common_item_list",
    ],
) -> List[str]:
    item_list = []
    if not use_item_list:
        return item_list

    dir_item_list = Path(dir_item_list)

    if isinstance(use_item_list, str) or "all" in use_item_list:
        if use_item_list == "all" or "all" in use_item_list:
            use_item_list = [el.stem for el in dir_item_list.iterdir() if el.is_file()]
        else:
            use_item_list = [use_item_list]

    for el in use_item_list:
        with dir_item_list.joinpath(f"{el}.txt").open("r", encoding="utf-8") as f:
            # item_list = {*item_list, *set([w.strip() for w in f.readlines()])}
            item_list.extend([w.strip() for w in f.readlines()])
    item_list = set(item_list)

    # item_list = list(
    #     set(
    #         list(item_list)
    #         + [w.lower() for w in item_list]
    #         + [w.upper() for w in item_list]
    #         + [" ".join([el.capitalize() for el in w.split()]) for w in item_list]
    #     )
    # )
    item_list.update(set([w.lower() for w in item_list]))
    item_list.update(set([w.replace(" ", "-") for w in item_list]))
    item_list = list(item_list)

    item_list = sorted(item_list, key=len, reverse=True)
    return item_list


def load_vocabulary(dir_vocabulary: Path) -> Dict[str, str]:
    with dir_vocabulary.open("r", encoding="utf8") as f:
        vocabulary = json.load(f)
    return vocabulary


def train_test_split(df: Series, frac=0.2):
    """
    Split a pd.Series into train and test samples.
    """
    test = df.sample(frac=frac, axis=0)
    train = df.drop(index=test.index)
    return train, test


def set_logger(console_log=True, file_log=True, file_loc: Union[Path, str] = "app.log"):
    # Set up the logger
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create console handler
    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Create file handler
    if file_log:
        file_loc = Path(file_loc)
        file_loc.parents[0].mkdir(parents=True, exist_ok=True)
        file_loc.touch()
        file_handler = logging.FileHandler(file_loc, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # logger.info("Log created.")
    return logger


c_type_beg = r"[\s\.,-]*(?<![\b\w\.])"
c_type_end = r"[\s\.,-]*(?=[\s\.,-]*)(?![\w\.])"
soc_merc = {
    r"s(oc(iedad)?)?.l(im(itada)?)?": "s.l",
    r"s(oc(iedad)?)?.l(im(itada)?)?.l(ab(oral)?)?": "s.l.l",
    r"s(oc(iedad)?)?.l(im(itada)?)?.u(nipersonal)?": "s.l.u",
    r"s(oc(iedad)?)?.l(im(itada)?)?.p(rof(esional)?)?": "s.l.p",
    r"s(oc(iedad)?)?.l(im(itada)?)?.n(ueva)?.e(mp(resa)?)?": "s.l.n.e",
    r"s(oc(iedad)?)?.(de )?r(es(p(onsabilidad)?)?)?.l(im(itada)?)?": "s.r.l",
    r"s(oc(iedad)?)?.a(n(o|ó)nima)?": "s.a",
    r"s(oc(iedad)?)?.a(n(o|ó)nima)?.l(ab(oral)?)?": "s.a.l",
    r"s(oc(iedad)?)?.a(n(o|ó)nima)?.u(nipersonal)?": "s.a.u",
    r"s(oc(iedad)?)?.a(n(o|ó)nima)?.d(ep(ortiva)?)?": "s.a.d",
    r"s(oc(iedad)?)?.a(n(o|ó)nima)?.e(s(p(añola)?)?)?": "s.a.e",
    r"s(oc(iedad)?)?.c(iv(il)?)?": "s.c",
    r"s(oc(iedad)?)?.c(iv(il)?)?.p(riv(ada)?)?": "s.c.p",
    r"s(oc(iedad)?)?.c(iv(il)?)?.p(art(icular)?)?": "s.c.p",
    # r"s(oc(iedad)?)?.c(oop(erativa)?)?":'c.o.o.p',
    r"(s(oc(iedad)?)?)?.c.o.o.p(erativa)?": "c.o.o.p",
    r"s(oc(iedad)?)?.c(oop(erativa)?)?.a(ndaluza)?": "s.c.a",
    r"s(oc(iedad)?)?.c(oop(erativa)?)?.l(im(itada)?)?": "s.c.l",
    r"s(oc(iedad)?)?.c(oop(erativa)?)?.i(nd(ustrial)?)?": "s.c.i",
    r"s(oc(iedad)?)?.a(graria)?.(de )?t(ransformaci(o|ó)n)?": "s.a.t",
    r"s(oc(iedad)?)?.(de )?g(arant(i|í)a)?.r(ec(i|í)proca)?": "s.g.r",
    r"s(oc(iedad)?)?.i(rr(eg(ular)?)?)?": "s.i",
    r"s(oc(iedad)?)?.r(eg(ular)?)?.c(ol(ectiva)?)?": "s.r.c",
    r"s(uc(ursal)?)?.(en )?e(s(p(aña)?)?)?": "s.e.e",
    r"s(oc(iedad)?)?.(en )?c(om(andita)?)?": "s.e.n.c",
    r"c(om(unidad)?)?.(de )?b(ienes)?": "c.b",
    r"a(grupaci(o|ó)n)?.(de )?i(nt(er(e|é)s)?)?.e(con(o|ó)mico)?": "a.i.e",
    r"a(grupaci(o|ó)n)?.e(uropea)?.(de )?i(nt(er(e|é)s)?)?.e(con(o|ó)mico)?": "a.e.i.e",
    r"u(ni(o|ó)?n)?.t(emp(oral)?)?.(de )?e(mp(resas)?)?": "u.t.e",
    r"inc(orporated)?.": "inc",
    r"l(imi)?t(e)?d.": "ltd",
    r"co": "co",
    # r"llc": "llc",
}


def replace_company_types(text, remove_type=False):
    """
    Replace the company type if present in a text in any given format
    (e.g.: "s.l.", "sl", "s. l.") into a standard form ("s.l.")
    or remove it if `remove_type`=`True`.
    """
    for pat, soc in soc_merc.items():
        pat = pat.replace(".", "[\s\.]*")
        pattern = rf"{c_type_beg}{pat}{c_type_end}"
        if remove_type:
            soc = " "
        else:
            soc = " " + soc + ". "
        text = regex.sub(pattern, soc, text)
    text = regex.sub(r"\s+", " ", text).strip()
    return text
