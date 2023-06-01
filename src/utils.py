import json
from pathlib import Path
from typing import List, Union, Dict

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

    if isinstance(use_item_list, str):
        if use_item_list == "all":
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
