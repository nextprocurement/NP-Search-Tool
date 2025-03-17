import contextlib
from datetime import datetime
import json
import logging
import pathlib
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

import joblib
import pandas as pd
import regex
from joblib import Parallel, delayed
from pandas import Series
from tqdm.auto import tqdm


def parallelize_function(
    func,
    data: Union[pd.Series, List],
    workers=-1,
    prefer="processes",
    output: str = "list",
    show_progress: bool = True,
    desc: str = "",
    leave: bool = False,
    position: int = 0,
    *args,
    **kwargs,
):
    """
    Parallelizes function execution

    Parameters:
    -----------
    func:
        Function to be parallelized
    data:
        Data to be processed
    workers: int
        Number of worker processes. Default -1 (all cores)
    prefer: str
        Preferred backend for parallelization ("processes" or "threads")
    output:
        Type of output ("series" or "list")
    show_progress: bool
        If `True` a progress bar is shown.
    desc: str
        Description for the progress bar.
    leave: bool
        If `True` keeps all traces of the progress bar.
    position: int
        Specify the line offset to print this bar (starting from 0)
    args, kwargs:
        Additional arguments for the function

    Returns:
    --------
    Processed results as a Pandas Series or List
    """

    if show_progress:
        with tqdm_joblib(
            tqdm(desc=desc, total=len(data), leave=leave, position=position)
        ) as progress_bar:
            results = Parallel(n_jobs=workers, prefer=prefer, verbose=0)(
                delayed(func)(x, *args, **kwargs) for x in data
            )
    else:
        results = Parallel(n_jobs=workers, prefer=prefer, verbose=0)(
            delayed(func)(x, *args, **kwargs) for x in data
        )

    results = list(results)
    if output == "series" and isinstance(data, pd.Series):
        results = pd.Series(results, index=data.index)
        return results
    return results


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._tqdm_object = tqdm_object

        def __call__(self, *args, **kwargs):
            # Check if the callback has `batch_size` attribute
            if hasattr(self, "batch_size"):
                self._tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# Load item_list
def load_item_list(
    dir_data: Union[str, Path],
    folder_name: str,
    use_item_list: Union[str, List[str]] = "all",
) -> List[str]:
    """
    Parameters
    ----------
    dir_data: str
        Data location. Can be either directory or zip file.
    folder_name: str
        Inner directory to read from.
    use_item_list: str|list(str)
        Which elements to use from folder name.
        If "all": get all elements
        If list of strings: read only those specific elements.
    """
    item_list = []
    if not use_item_list:
        return item_list
    if isinstance(use_item_list, str):
        use_item_list = [use_item_list]

    main_path = Path(dir_data)

    if main_path.suffix == ".zip":
        # If it's a zip file, create a zipfile.Path and find the specific folder within it.
        with zipfile.ZipFile(main_path, "r") as zip_ref:
            zip_main_path = zipfile.Path(zip_ref)
            specific_folder = zip_main_path / folder_name
            if not specific_folder.exists():
                return item_list
            if "all" in use_item_list:
                use_item_list = [
                    el.name.split(".")[0]
                    for el in specific_folder.iterdir()
                    if el.is_file()
                ]
            for el in use_item_list:
                if specific_folder.joinpath(f"{el}.txt").is_file():
                    item_list.extend(
                        [
                            w.strip()
                            for w in specific_folder.joinpath(f"{el}.txt")
                            .read_text(encoding="utf-8")
                            .split("\n")
                        ]
                    )
    else:
        # If it's a directory, find the specific folder and list all txt files in it.
        specific_folder = main_path / folder_name
        if not specific_folder.exists():
            return item_list
        if "all" in use_item_list:
            use_item_list = [
                el.name.split(".")[0]
                for el in specific_folder.iterdir()
                if el.is_file()
            ]
        for el in use_item_list:
            if specific_folder.joinpath(f"{el}.txt").is_file():
                with specific_folder.joinpath(f"{el}.txt").open(encoding="utf-8") as f:
                    item_list.extend([w.strip() for w in f.readlines()])

    item_list = set(item_list)
    item_list.update(set([w.lower() for w in item_list]))
    item_list.update(set([w.replace(" ", "-") for w in item_list]))
    item_list = list(item_list)

    item_list = sorted(item_list, key=len, reverse=True)
    return item_list


def load_equivalences(
    dir_data: Union[str, Path],
    use_item_list: Union[str, List[str]] = "all"
):  
    
    equivalent = {}
    if not use_item_list:
        return equivalent
    if isinstance(use_item_list, str):
        use_item_list = [use_item_list]

    main_path = Path(dir_data)
    
    # Find the 'equivalences' folder and list all txt files in it
    specific_folder = main_path / 'equivalences'
    
    if not specific_folder.exists():
        return equivalent
    if "all" in use_item_list:
        use_item_list = [
            el.stem
            for el in specific_folder.iterdir()
            if el.is_file()
        ]
    for el in use_item_list:
        if specific_folder.joinpath(f"{el}.txt").is_file():
            with specific_folder.joinpath(f"{el}.txt").open(encoding="utf-8") as f:
                newEq = [x.split(':') for x in f.readlines()]
                newEq = [x for x in newEq if len(x) == 2]
                newEq = dict(newEq)
                equivalent = {**equivalent, **newEq}

    return equivalent

def train_test_split(df: Series, frac=0.2):
    """
    Split a pd.Series into train and test samples.
    """
    test = df.sample(frac=frac, axis=0)
    train = df.drop(index=test.index)
    return train, test


def set_logger(
    console_log: bool = True,
    file_log: bool = True,
    file_loc: Union[Path, str] = "app.log",
    log_level: Union[
        int, str
    ] = "INFO",  # CRITICAL FATAL ERROR WARN WARNING INFO DEBUG NOTSET
    logger_name: str = "app-log",
):
    # Set up the logger
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Create console handler
    if console_log:
        console_handler = logging.StreamHandler()
        # console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        # print(console_handler)
        logger.addHandler(console_handler)
    print(logger)
    # Create file handler
    if file_log:
        file_loc = Path(file_loc)
        file_loc.parents[0].mkdir(parents=True, exist_ok=True)
        file_loc.touch()
        file_handler = logging.FileHandler(file_loc, encoding="utf-8")
        # file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # print(logger)
    logger.info("Log created.")
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

def log_or_print(
    message: str,
    level: str = "info",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Helper function to log or print messages.

    Parameters
    ----------
    message : str
        The message to log or print.
    level : str, optional
        The logging level, by default "info".
    logger : logging.Logger, optional
        The logger to use for logging, by default None.
    """
    if logger:
        if level == "info":
            logger.info(message)
        elif level == "error":
            logger.error(message)
    else:
        print(message)

def load_yaml_config_file(config_file: str, section: str, logger:logging.Logger) -> Dict:
    """
    Load a YAML configuration file and return the specified section.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.
    section : str
        Section of the configuration file to return.

    Returns
    -------
    Dict
        The specified section of the configuration file.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    ValueError
        If the specified section is not found in the configuration file.
    """

    if not pathlib.Path(config_file).exists():
        log_or_print(f"Config file not found: {config_file}", level="error", logger=logger)
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    section_dict = config.get(section, {})

    if section == {}:
        log_or_print(f"Section {section} not found in config file.", level="error", logger=logger)
        raise ValueError(f"Section {section} not found in config file.")

    log_or_print(f"Loaded config file {config_file} and section {section}.", logger=logger)

    return section_dict


def init_logger(
    config_file: str,
    name: str = None
) -> logging.Logger:
    """
    Initialize a logger based on the provided configuration.

    Parameters
    ----------
    config_file : str
        The path to the configuration file.
    name : str
        The name of the logger.

    Returns
    -------
    logging.Logger
        The initialized logger.
    """

    logger_config = load_yaml_config_file(config_file, "logger", logger=None)
    name = name if name else logger_config.get("logger_name", "default_logger")
    log_level = logger_config.get("log_level", "INFO").upper()
    dir_logger = pathlib.Path(logger_config.get("dir_logger", "logs"))
    N_log_keep = int(logger_config.get("N_log_keep", 5))

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Create path_logs dir if it does not exist
    dir_logger.mkdir(parents=True, exist_ok=True)
    print(f"Logs will be saved in {dir_logger}")

    # Generate log file name based on the data
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"{name}_log_{current_date}.log"
    log_file_path = dir_logger / log_file_name

    # Remove old log files if they exceed the limit
    log_files = sorted(dir_logger.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    if len(log_files) >= N_log_keep:
        for old_file in log_files[N_log_keep - 1:]:
            old_file.unlink()

    # Create handlers based on config
    if logger_config.get("file_log", True):
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    if logger_config.get("console_log", True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    return logger
