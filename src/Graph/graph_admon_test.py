# %%
import pandas as pd
import numpy as np
from unidecode import unidecode
import re
from itertools import permutations, combinations

import plotly.express as px
import networkx as nx
import plotly.graph_objects as go
from pyvis.network import Network

import seaborn as sns
import colorcet as cc
from matplotlib.colors import to_hex
from bs4 import BeautifulSoup as bs
from tabulate import tabulate

# %%
# minors = pd.read_parquet("data/minors.parquet")
# m_cols = sorted(minors.columns)
insiders = pd.read_parquet("/export/usuarios_ml4ds/lbartolome/NextProcurement/sproc/place_feb_21/insiders.parquet")
i_cols = sorted(insiders.columns)
# outsiders = pd.read_parquet("data/outsiders.parquet")  # agregados
# o_cols = sorted(outsiders.columns)


# %%
def get_nas(cell):
    nas = []
    if (hasattr(cell, "__iter__")) and (not isinstance(cell, str)):
        for el in cell:
            if el == "nan" or pd.isna(el):
                nas.append(True)
            else:
                nas.append(False)
    else:
        nas.append(pd.isna(cell))
    return not all(nas)


def cell_islist(cell):
    return (hasattr(cell, "__iter__")) and (not isinstance(cell, str))


def cell_isna(cell):
    nas = pd.isnull(cell)
    if cell_islist(cell):
        if all(nas):
            return True
        else:
            return False
    return nas


def fill_na(cell, fill=[]):
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


def pad_list(row, lot_cols):
    length = len(row[lot_cols[0]])
    data = []
    for el in row:
        length_el = length - len(el)
        data.append(list(el) + [None] * length_el)
    return pd.Series(data)


def get_cell_cpvs(cell):
    return np.array(eval(cell), dtype=int, ndmin=1)


def scale(array, dmin=0, dmax=1):
    array_min = np.min(array)
    array_max = np.max(array)
    array_std = (array - array_min) / (array_max - array_min)
    array_scaled = array_std * (dmax - dmin) + dmin
    return array_scaled


def clean_txt(txt):
    if txt is None:
        return None
    return " ".join(re.sub("[^\w\(\)\[\]]", " ", txt).split())


# %%
# minors.shape, insiders.shape, outsiders.shape,


# %%
df = insiders
cols = df.columns.tolist()

index_names = df.index.names
df.reset_index(inplace=True)
df["identifier"] = df[index_names].astype(str).agg("/".join, axis=1)
df.drop(index_names, inplace=True, axis=1)
df.set_index("identifier", inplace=True)


# %%
join_str = lambda x: ".".join([el for el in x if el])
joint_cnames = {join_str(c): c for c in df.columns}

cpv_cols = sorted([v for k, v in joint_cnames.items() if "ItemClassificationCode" in k])
admon_tree_cols = [
    v
    for k, v in joint_cnames.items()
    if k.endswith("PartyName.Name") and ("LocatedContractingParty" in v)
][::-1]
amount_cols = [
    v
    for k, v in joint_cnames.items()
    if k.startswith("ContractFolderStatus.ProcurementProject.")
    and ("TotalAmount" in v or "TaxExclusiveAmount" in v)
]
date = [
    v
    for k, v in joint_cnames.items()
    if "ContractFolderStatus.ProcurementProject.PlannedPeriod.StartDate" in k
]
lot_cols = sorted(
    [
        v
        for k, v in joint_cnames.items()
        if "ProcurementProjectLot.ID" in k
        or "ProcurementProjectLot.ProcurementProject" in k
    ]
)
win_cols = sorted(
    [
        v
        for k, v in joint_cnames.items()
        if "WinningParty.PartyIdentification.ID" in k
        or "WinningParty.PartyName.Name" in k
    ]
)


# %% [markdown]
# ## Enterprise

# %%
# Get winning companies and unify names
win_cols_keep = [win_cols[i] for i in [0,2]]
companies = df[win_cols_keep]
companies.columns = ["CompanyID", "CompanyName"]
# companies[companies["CompanyID"].apply(len)>1]
companies["CompanyID"] = companies["CompanyID"].apply(lambda x: x[0])
companies["CompanyName"] = companies["CompanyName"].apply(lambda x: x[0])
companies = (
    companies.applymap(fill_na, fill=None)
    .applymap(clean_txt)
    .applymap(lambda x: x.lower() if x else None)
)
companies["unified"] = companies["CompanyName"].apply(
    lambda x: unidecode(x) if x else None
)
companies["CompanyID"] = companies["CompanyID"].apply(
    lambda x: x.upper() if x else None
)

# %%
# # Company mapping
# company_map = companies[companies.duplicated()].dropna(how="all")
# rep = dict(company_map.values)
# n_rep = dict(companies.drop(company_map.index).dropna(how="all").values)

# %% [markdown]
# ## CPV

# %%
# Find elements with CPVs
cname = "ItemClassificationCode"

cpv_cols = sorted([c for c in cols if cname in "".join(c)])
sorted([".".join([el for el in c if el]) for c in cpv_cols])
project_cpvs = df[cpv_cols].applymap(fill_na, fill=None).dropna(how="all")

# cpvs = project_cpvs[cpv_cols[0]].dropna().apply(lambda x: set(get_cell_cpvs(x)))
cpvs = (
    project_cpvs[cpv_cols[0]]
    .dropna()
    .apply(lambda x: set(np.array(eval(x[0]), dtype=int, ndmin=1)))
)
cpvs.name = cname


# %%
# # Nodes
# nodes = (
#     pd.DataFrame(np.log(cpvs.explode().value_counts()) ** 2)
#     .reset_index()
#     .rename(columns={"index": "id", cname: "size"})
# )
# nodes["id"] = nodes["id"].astype(int)
# nodes["label"] = nodes["id"].astype(str)
# nodes["division"] = nodes["label"].str[:2]
# nodes["grupo"] = nodes["label"].str[:3]
# nodes["clase"] = nodes["label"].str[:4]
# nodes["categoria"] = nodes["label"].str[:5]
# ids = list(
#     set(
#         nodes[["division", "size"]]
#         .groupby("division")
#         .agg(pd.Series.idxmax)["size"]
#         .tolist()
#     )
# )
# nodes.loc[~nodes.index.isin(ids), "label"] = None
# nodes.loc[ids, "font"] = "300px arial black"
# divisions = nodes["division"].unique()
# palette = sns.color_palette(cc.glasbey, len(divisions))
# cmap = dict(zip(divisions, [to_hex(p) for p in palette]))
# nodes["color"] = nodes["division"].apply(cmap.get)

# # Edges
# edges = pd.DataFrame()
# edges[["source", "target"]] = (
#     cpvs[cpvs.apply(len) > 1]
#     .apply(lambda x: list(combinations(x, 2)))
#     .explode()
#     .tolist()
# )
# edges[["source", "target"]] = edges[["source", "target"]].astype(int)
# swap = edges["source"] > edges["target"]
# edges.loc[swap, ["source", "target"]] = edges.loc[swap, ["target", "source"]].values
# edges["weight"] = 1
# edges = edges.groupby(["source", "target"], axis=0).agg(sum).reset_index()
# edges = edges[edges["weight"] > 5]
# # edges["weight"] = np.log(edges["weight"]) ** 2
# edges["weight"] = scale(edges["weight"], 1, 150)
# nodes = nodes.loc[
#     nodes["id"].isin(set(edges["source"].tolist() + edges["target"].tolist()))
# ]


# %%
# edges.to_csv("edges.csv", index=False)
# nodes.to_csv("nodes.csv", index=False)


# %%
# def create_graph(nodes, edges, name):
#     G = nx.Graph()
#     G.add_nodes_from(
#         [(k, v) for k, v in nodes.set_index("id").to_dict(orient="index").items()]
#     )
#     G.add_edges_from(
#         [
#             (k[0], k[1], v)
#             for k, v in edges.set_index(["source", "target"])
#             .to_dict(orient="index")
#             .items()
#         ]
#     )
#     G = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
#     pos = nx.spring_layout(G, k=0.01, iterations=50)

#     net = Network(
#         notebook=True, neighborhood_highlight=True, select_menu=True, filter_menu=True
#     )
#     net.from_nx(G, show_edge_weights=True)
#     neighbor_map = net.get_adj_list()
#     net.show_buttons(filter_=["physics"])
#     net.force_atlas_2based(
#         gravity=-200,
#         central_gravity=0.01,
#         spring_length=0,
#         spring_strength=0.6,
#         damping=0.4,
#         overlap=0,
#     )
#     br = "\n"
#     for node in net.nodes:
#         x, y = pos[node["id"]]
#         node["x"] = x * 10000
#         node["y"] = y * 10000
#         # node["value"] = node["size"]*100000
#         node["title"] = (
#             f"{node['id']}"
#             # f"\nNeighbors:\n{br.join(str(el) for el in neighbor_map[node['id']])}"
#         )
#     #     # node["color"] = cmap[node["division"]]
#     net.toggle_physics(False)
#     net.save_graph(f"{name}.html")


# # create_graph(nodes, edges, "CPVs")


# %% [markdown]
# ## Project Lots

# %%
project_lots = df[lot_cols].applymap(fill_na, fill=None).dropna(how="all")

# %%
#re.sub(r"[a-z]","")

# %%
#[[el for el in c if el] for c in project_lots.columns]

# %%
lot_cols = []
for c in project_lots.columns:
    print(c)
    v = [el for el in c if el]
    lot_cols.append(re.sub(r"[a-z]", "", ".".join(v[:-1])) + f".{v[-1]}")
cpvCol = [c for c in lot_cols if c.endswith("ItemClassificationCode")][0]

# %%
# Find projects with Lots
import pdb; pdb.set_trace()
project_lots = df[lot_cols].applymap(fill_na, fill=None).dropna(how="all")

lot_cols = []
for c in project_lots.columns:
    v = [el for el in c if el]
    lot_cols.append(re.sub(r"[a-z]", "", ".".join(v[:-1])) + f".{v[-1]}")
project_lots.columns = lot_cols
cpvCol = [c for c in lot_cols if c.endswith("ItemClassificationCode")][0]

project_lots[cpvCol] = project_lots[cpvCol].apply(
    lambda x: np.array(",".join(x), ndmin=1) if x is not None else None
)

project_lots = project_lots.applymap(
    lambda x: x[0] if x is not None else None
).applymap(
    lambda x: []
    if x is None
    else eval(f"['''{x}''']")
    if not (x.startswith("[") and x.endswith("]"))
    else eval(x)
)

project_lots = project_lots.apply(
    pad_list, axis=1, result_type="broadcast", lot_cols=lot_cols
)


# %%
# Clean repeated elements
project_lots["items"] = project_lots.apply(
    lambda x: len(set([len(el) for el in x])), axis=1
)
project_lots.loc[project_lots["items"] > 1, cpvCol] = project_lots.loc[
    project_lots["items"] > 1, cpvCol
].apply(lambda x: [x])
project_lots = project_lots.drop("items", axis=1)

# project_lots = project_lots.drop("contratosMenoresPerfilesContratantes_2021.zip/contratosMenoresPerfilesContratantes_20210127_041240.atom/364")
# project_lots[~project_lots.apply(lambda x: len(set([len(el) for el in x]))==1, axis=1)]


# %%
# Get each lot separately
project_lots_sep = project_lots[lot_cols].apply(pd.Series.explode)
for c in project_lots_sep.columns:
    if "name" in c.lower():
        project_lots_sep[c] = project_lots_sep[c].astype(str)
    elif c == cpvCol:
        project_lots_sep.loc[~project_lots_sep[cpvCol].isna(), cpvCol] = (
            project_lots_sep.loc[
                ~project_lots_sep[cpvCol].isna(),
                cpvCol,
            ]
            .apply(lambda x: np.array(x, dtype=float, ndmin=1).astype(int))
            .values
        )
    else:
        pass
        # project_lots_sep[c] = project_lots_sep[c].astype(float)


# %% [markdown]
# ### Comentarios
#
# En la entrada: contratosMenoresPerfilesContratantes_2018.zip/contratosMenoresPerfilesContratantes_20190225_151353_7.atom/203 con <id>https://contrataciondelestado.es/sindicacion/datosAbiertosMenores/3023064</id>
# Son cinco lotes. Cada uno tiene cuatro códigos, pero en el parquet se guardan como una lista:
# ['c1', 'c2', 'c3', 'c4', '[c1, c2, c3, c4]', '[c1, c2, c4, c3]', '[c1, c2, c3, c4]', '[c1, c3, c2, c4]']
#
# Y en la entrada: contratosMenoresPerfilesContratantes_2018.zip/contratosMenoresPerfilesContratantes_20190225_151353_6.atom/154 con <id>https://contrataciondelestado.es/sindicacion/datosAbiertosMenores/3028033</id>
# Hay dos lotes, el primero tiene dos elementos y el segundo solamente uno, pero están en una misma lista (como si hubiera tres lotes).
# ['c1', 'c2', 'c1'] cuando realmente debería ser [['c1', 'c2'], 'c1']
#
# Aquí solamente hay un lote con 2 CPVs, que están en el formato ['c1', 'c2'] y deberían ser [['c1', 'c2']]
# contratosMenoresPerfilesContratantes_2021.zip/contratosMenoresPerfilesContratantes_20210127_041240.atom/364

# %% [markdown]
# ## Administraciones

# %% [markdown]
# #### Select relevant columns (administration hierarchy, project amount)

# %%
admons = df.loc[:, admon_tree_cols + [amount_cols[0]] + date]
# Rename columns
tree_level_cols = [f"A_level{c}" for c in range(len(admon_tree_cols))]
amount_col = ["Amount"]
date_col = ["Date"]
admons_cols = tree_level_cols + amount_col + date_col
admons.columns = admons_cols
# Clean values
admons = admons.fillna(np.nan).replace([np.nan], [None])
admons.loc[:, tree_level_cols] = admons[tree_level_cols].applymap(clean_txt)
admons.loc[:, "Date"] = pd.to_datetime(
    admons["Date"], errors="coerce", dayfirst=True
).dt.year
# Include the full path of the administration
admon_tree = (
    admons[tree_level_cols]
    .agg(lambda x: unidecode(";".join([el for el in x if el]).lower()), axis=1)
    .values
)
admons.loc[:, "admon_tree"] = admon_tree


# %% [markdown]
# #### Select only certain administrations

# %%
# select_admon = "sector publico"
# words = ["caceres", "^(?!.*sector publico)"]
words = [
    "sector publico",
    # "ayuntamiento",
    # "ministerio",
    # "."
]
select_admon = (
    r"\b"
    + r"\b|\b".join([r".*\b".join(w) for w in permutations(words, len(words))])
    + r"\b"
)

admon = admons.loc[
    admons["admon_tree"].apply(lambda x: any(re.findall(select_admon, x))),
    :,
]
admon["LowestLevel"] = admon[tree_level_cols[-1]].apply(lambda x: unidecode(x.lower()))
admon[tree_level_cols] = admon[tree_level_cols].fillna("")


# %%
parent_tree = (
    admon[date_col + tree_level_cols].groupby(tree_level_cols).agg(len).reset_index()
)
used_cols = tree_level_cols[::-1]
for i, c in enumerate(used_cols[:-1]):
    # Correct positions: the values should be in the selected columns, but they are repeated because other columsn are empty (thus not aggregated)
    aux = (
        parent_tree[used_cols[i : i + 2]]
        .reset_index()
        .groupby(c)
        .agg(lambda x: list(x))
    )
    correct_index = (
        aux.loc[
            (aux["index"].apply(len) == 2)
            & (aux[used_cols[i + 1]].apply(lambda x: "" in x)),
            "index",
        ]
        .explode()
        .values
    )
    # Obtain the indices that must be shifted
    incorrect_positions = (
        parent_tree.drop(correct_index)[used_cols[i + 1]].apply(len) == 0
    )
    shift_index = incorrect_positions[incorrect_positions].index

    # Unify repeated values
    merged_rows = (
        parent_tree.loc[correct_index]
        .groupby(c)
        .agg(
            {
                **{el: lambda x: sorted(x)[-1] for el in tree_level_cols if el != c},
                "Date": sum,
            }
        )
        .reset_index()
    )

    # Shift
    if any(shift_index):
        parent_tree.loc[shift_index, tree_level_cols] = parent_tree.loc[
            shift_index, tree_level_cols
        ].shift(periods=-1, axis=1, fill_value="")

    parent_tree = pd.concat([parent_tree.drop(correct_index), merged_rows]).reset_index(
        drop=True
    )

# Finally, remove some values that might be the same but with different symbols (eg. tildes, caps, etc.)
admon_tree = (
    parent_tree[tree_level_cols]
    .agg(lambda x: unidecode(";".join([el for el in x if el]).lower()), axis=1)
    .values
)
parent_tree.loc[:, "admon_tree"] = admon_tree
aux = parent_tree["admon_tree"].duplicated(keep=False)
aux_index = aux[aux].index

merged_rows = (
    parent_tree.loc[aux_index]
    .groupby("admon_tree")
    .agg(
        {
            **{el: lambda x: sorted(x)[0] for el in tree_level_cols},
            "Date": sum,
        }
    )
    .reset_index(drop=True)
)
parent_tree = pd.concat([parent_tree.drop(aux_index), merged_rows]).reset_index(
    drop=True
)[tree_level_cols]
parent_tree.loc[:, "LowestLevel"] = parent_tree.agg(
    lambda x: unidecode([el for el in x if el][-1].lower()), axis=1
)


# %%
admon[tree_level_cols] = (
    admon[tree_level_cols]
    .agg(lambda x: ";".join([el for el in x if el]), axis=1)
    .apply(lambda x: x.split(";"))
    .apply(lambda x: x + [""] * (len(tree_level_cols) - len(x)))
    .tolist()
)

# admon_counts = pd.merge(
#     admon[["Date", "Amount", "LowestLevel"]],
#     # parent_tree,
#     parent_tree[~parent_tree["LowestLevel"].duplicated(keep=False)],
#     how="left",
#     on="LowestLevel",
# )
admon_counts = admon[
    ["Date", "Amount", "LowestLevel"] + tree_level_cols + ["admon_tree"]
]
admon_counts.loc[:, "Counts"] = 1
admon_counts = admon_counts.set_index(admon.index)
# admon_counts = admon_counts.dropna()
# admon_counts["Date"] = admon_counts["Date"].astype(int)
admon_counts["Date"] = (
    admon_counts["Date"]
    .apply(lambda x: int(x) if not np.isnan(x) else 0)
    .replace(0, None)
)
# admon_counts["admon_tree"] = (
#     admon_counts[tree_level_cols]
#     .agg(lambda x: unidecode(";".join([el for el in x if el]).lower()), axis=1)
#     .values
# )


# %%
def get_nodes(
    df: pd.DataFrame,
    id_col: str = "id",
    color_col: str = "color",
    aggregate_cols: list = [],
    sum_cols: list = [],
    additional: list = [],
):
    """
    Params
    ------
    df: pd.DataFrame
        Dataframe with info to show
    """
    aux = df[[id_col, color_col] + aggregate_cols + sum_cols + additional]
    aux["# items"] = 1
    nodes = (
        aux.groupby([id_col] + aggregate_cols)
        .agg(
            {
                "# items": sum,
                color_col: lambda x: list(set(x))[0],
                **{k: sum for k in sum_cols},
                **{k: list for k in additional},
            }
        )
        .reset_index()
    )
    nodes.insert(1, "label", nodes["id"].astype(str))

    return nodes


def get_edges(
    series: pd.DataFrame,
):
    # Edges
    edges = pd.DataFrame()
    edges[["source", "target"]] = (
        series[series.apply(len) > 1]
        .apply(lambda x: list(combinations(x, 2)))
        .explode()
        .tolist()
    )
    edges[["source", "target"]] = edges[["source", "target"]].astype(int)
    swap = edges["source"] > edges["target"]
    edges.loc[swap, ["source", "target"]] = edges.loc[swap, ["target", "source"]].values
    edges["weight"] = 1
    edges = edges.groupby(["source", "target"], axis=0).agg(sum).reset_index()
    edges = edges[edges["weight"] > 2]
    # edges["weight"] = np.log(edges["weight"]) ** 2
    edges["weight"] = scale(edges["weight"], 1, 30)

    return edges


# %%
# Administations and CPVs
admon_cpvs = pd.merge(
    pd.merge(admon_counts, cpvs, how="inner", left_index=True, right_index=True),
    companies["CompanyID"],
    how="inner",
    left_index=True,
    right_index=True,
)
comunidades = {
    "andalucia": "Andalucía",
    "aragon": "Aragón",
    "asturias": "Principado de Asturias",
    "balear": "Illes Balears",
    "canarias": "Canarias",
    "cantabria": "Cantabria",
    "y leon": "Castilla y León",
    "la mancha": "Castilla-La Mancha",
    "catalu": "Cataluña",
    "valencia(?! de)": "Comunitat Valenciana",
    "extremadura": "Extremadura",
    "galicia": "Galicia",
    "madrid": "Comunidad de Madrid",
    "murcia": "Región de Murcia",
    "navarra": "Comunidad Foral de Navarra",
    "vasco": "País Vasco",
    "euska": "País Vasco",
    "la rioja": "La Rioja",
    "ceuta": "Ciudad Autónoma de Ceuta",
    "melilla": "Ciudad Autónoma de Melilla",
}
admon_cpvs["Comunidad"] = [[]] * len(admon_cpvs)
for k, v in comunidades.items():
    com = (
        admon_cpvs["admon_tree"]
        .apply(lambda x: [comunidades[k]] if re.search(k, x) else None)
        .dropna()
    )
    admon_cpvs.loc[com.index, "Comunidad"] = (
        admon_cpvs.loc[com.index, "Comunidad"] + com
    )

edges = get_edges(admon_cpvs["ItemClassificationCode"])

# We must manage somehow where Comunidad has more than one item

admon_cpvs = admon_cpvs.explode("ItemClassificationCode").explode("Comunidad")
admon_cpvs["id"] = admon_cpvs["ItemClassificationCode"].astype(int)
admon_cpvs["label"] = admon_cpvs["id"].astype(str)
admon_cpvs["division"] = admon_cpvs["label"].str[:2]
admon_cpvs["grupo"] = admon_cpvs["label"].str[:3]
admon_cpvs["clase"] = admon_cpvs["label"].str[:4]
admon_cpvs["categoria"] = admon_cpvs["label"].str[:5]

nodes = get_nodes(
    admon_cpvs,
    id_col="id",
    color_col="division",
    aggregate_cols=["Date", "Comunidad"],
    sum_cols=["Amount", "Counts"],
    additional=["A_level0"],
)


# %%
edges = admon_cpvs[["Comunidad", "CompanyID"]].dropna()
edges["weight"] = 1
edges = edges.groupby(["Comunidad", "CompanyID"]).agg(sum).reset_index()
edges["weight"] = scale(edges["weight"], dmin=10, dmax=50)

nodes = pd.DataFrame(
    pd.concat([edges["Comunidad"].value_counts(), edges["CompanyID"].value_counts()])
).reset_index()
nodes.columns = ["id", "size"]
nodes["size"] = scale(nodes["size"].values, dmin=1, dmax=100)
nodes["polygon"] = 4
nodes.loc[nodes["id"].isin(edges["Comunidad"].unique()), "polygon"] = 3

edges.columns = ["source", "target", "weight"]

edges.to_csv("edges_ent.csv", index=False)
nodes.reset_index().to_csv("nodes_ent.csv", index=False)

# %%
# nodes_reduced = (
#     pd.pivot_table(
#         nodes[["id", "Amount", "Date", "Comunidad"]],
#         values="Amount",
#         index="id",
#         columns=["Date", "Comunidad"],
#     )
#     .dropna(how="all")
#     .to_dict(orient="index")
# )
# by_year_amount = {}
# for k, v in nodes_reduced.items():
#     for k2, v2 in v.items():
#         if pd.notna(v2):
#             by_year_amount[k] = by_year_amount.get(k, {})
#             by_year_amount[k][k2[0]] = {**by_year_amount[k].get(k2[0], {}), k2[1]: v2}


# %%
# nodes = pd.merge(
#     nodes[["id", "label", "division", "Counts"]]
#     .groupby(["id", "label", "division"])
#     .agg(sum)
#     .reset_index()
#     .rename(columns={"Counts": "size"})
#     .set_index("id"),
#     pd.Series(by_year_amount, name="Info"),
#     how="inner",
#     left_index=True,
#     right_index=True,
# ).reset_index()
# nodes["size"] = np.log(nodes["size"] + 10) ** 2
# ids = list(
#     set(
#         nodes[["division", "size"]]
#         .groupby("division")
#         .agg(pd.Series.idxmax)["size"]
#         .tolist()
#     )
# )
# nodes.loc[~nodes.index.isin(ids), "label"] = None
# nodes.loc[ids, "font"] = "300px arial black"
# divisions = nodes["division"].unique()
# palette = sns.color_palette(cc.glasbey, len(divisions))
# cmap = dict(zip(divisions, [to_hex(p) for p in palette]))
# nodes["color"] = nodes["division"].apply(cmap.get)
# nodes = nodes.set_index("id")
# nodes["Info"] = nodes["Info"].apply(lambda x: f'<div>{tabulate(pd.DataFrame.from_dict(x).fillna(0), headers="keys", tablefmt="html")}</div>')

# %%
# nodes

# %%
# edges.to_csv("edges.csv", index=False)
# nodes.reset_index().to_csv("nodes.csv", index=False)

# %%
# G = nx.Graph()
# G.add_nodes_from([(k, v) for k, v in nodes.to_dict(orient="index").items()])
# G.add_edges_from(
#     [
#         (k[0], k[1], v)
#         for k, v in edges.set_index(["source", "target"])
#         .to_dict(orient="index")
#         .items()
#     ]
# )
# G = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
# pos = nx.spring_layout(G, k=0.01, iterations=50)

# net = Network(
#     notebook=True, neighborhood_highlight=True, select_menu=True, filter_menu=True
# )
# net.from_nx(G, show_edge_weights=True)
# neighbor_map = net.get_adj_list()
# net.options.edges.smooth.type="curvedCW"
# net.options.edges.smooth.roundness=0.15
# net.show_buttons(
#     filter_=[
#         # "nodes",
#         # "edges",
#         # "layout",
#         # "interaction",
#         # "manipulation",
#         "physics",
#         # "selection",
#         "renderer",
#     ]
# )
# net.force_atlas_2based(
#     gravity=-200,
#     central_gravity=0.01,
#     spring_length=0,
#     spring_strength=0.6,
#     damping=0.4,
#     overlap=0,
# )
# br = "\n"
# for node in net.nodes:
#     x, y = pos[node["id"]]
#     node["x"] = x * 10000
#     node["y"] = y * 10000
#     # node["value"] = node["size"]*100000
#     node["title"] = (
#         f"<b>{node['id']}</b>"
#         "\n\n"
#         # f"\nNeighbors:\n{br.join(str(el) for el in neighbor_map[node['id']])}"
#         # f"{node.get('Info', None)}"
#         # f"{pretty_print(node.get('Info', None))}"
#         f"{node.get('Info', None)}"
#         # f"{tabulate(pd.DataFrame.from_dict(node.get('Info', None)).fillna(0), headers='keys', tablefmt='html')}"
#     )
#     # node["title"] += (
#     #     "<div> Neighbors:<br></div>"
#     #     + "<div>"
#     #     + "<br>".join(str(neighbor_map[node["id"]]))
#     #     + "</div>"
#     # )
# #     # node["color"] = cmap[node["division"]]
# net.toggle_physics(False)
# html = net.generate_html()


# %%
# soup = bs(html, 'html.parser')
# soup.style.insert_before(soup.new_tag(name="script", type="text/javascript", src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.16.1/vis-network.min.js"))
# # soup.style.insert_before(soup.new_string(orig_tooltip))
# soup.style.insert(1, soup.new_string(css_tooltip))
# soup.body.find("script").string = soup.body.find("script").string.replace("return network;", custom_popup)
# html = re.sub("(\n *\n){3,}", "\n\n", str(soup))

# with open("amos a ver2.html", "w") as f:
#     f.write(html)

# %%
# orig_tooltip = """
# <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.16.1/vis-network.min.js"></script>
# """

# css_tooltip = """
# div.popup {
#     position: absolute;
#     top: 0px;
#     left: 0px;
#     display: none;
#     background-color: #f5f4ed;
#     -moz-border-radius: 3px;
#     -webkit-border-radius: 3px;
#     border-radius: 3px;
#     border: 1px solid #808074;
#     box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2);
# }

# /* hide the original tooltip */
# .vis-network-tooltip {
#     display: none;
# }

# th, td {
# padding-left: 10px;
# padding-right: 10px;
# }

# table td {
#     border-top: thin solid;
#     border-bottom: thin solid;
# }

# table td:first-child {
#     border-left: thin solid;
# }

# table td:last-child {
#     border-right: thin solid;
# }
# """

# custom_popup = """
# // make a custom popup
# var popup = document.createElement("div");
# popup.className = 'popup';
# popupTimeout = null;
# popup.addEventListener('mouseover', function () {
#     console.log(popup)
#     if (popupTimeout !== null) {
#         clearTimeout(popupTimeout);
#         popupTimeout = null;
#     }
# });
# popup.addEventListener('mouseenter', function () {
#     console.log(popup)
#     if (popupTimeout !== null) {
#         clearTimeout(popupTimeout);
#         popupTimeout = null;
#     }
# });
# popup.addEventListener('mousemove', function () {
#     console.log(popup)
#     if (popupTimeout !== null) {
#         clearTimeout(popupTimeout);
#         popupTimeout = null;
#     }
# });
# popup.addEventListener('mouseout', function () {
#     if (popupTimeout === null) {
#         hidePopup();
#     }
# });
# container.appendChild(popup);


# // use the popup event to show
# network.on("showPopup", function (params) {
#     showPopup(params);
# });

# // use the hide event to hide it
# network.on("hidePopup", function (params) {
#     hidePopup();
# });


# // hiding the popup through css
# function hidePopup() {
#     popupTimeout = setTimeout(function () { popup.style.display = 'none'; }, 500);
# }

# // showing the popup
# function showPopup(nodeId) {
#     // get the data from the vis.DataSet
#     var nodeData = nodes.get(nodeId);
#     // get the position of the node
#     var posCanvas = network.getPositions([nodeId])[nodeId];

#     if (!nodeData) {
#         var edgeData = edges.get(nodeId);
#         var poses = network.getPositions([edgeData.from, edgeData.to]);
#         var middle_x = (poses[edgeData.to].x - poses[edgeData.from].x) * 0.5;
#         var middle_y = (poses[edgeData.to].y - poses[edgeData.from].y) * 0.5;
#         posCanvas = poses[edgeData.from];
#         posCanvas.x = posCanvas.x + middle_x;
#         posCanvas.y = posCanvas.y + middle_y;

#         popup.innerHTML = edgeData.title;
#     } else {
#         popup.innerHTML = nodeData.title;
#         // get the bounding box of the node
#         var boundingBox = network.getBoundingBox(nodeId);
#         posCanvas.x = posCanvas.x + 0.5 * (boundingBox.right - boundingBox.left);
#         posCanvas.y = posCanvas.y + 0.5 * (boundingBox.top - boundingBox.bottom);
#     };


#     //position tooltip:

#     // convert coordinates to the DOM space
#     var posDOM = network.canvasToDOM(posCanvas);

#     // Give it an offset
#     posDOM.x += 10;
#     posDOM.y -= 20;

#     // show and place the tooltip.
#     popup.style.display = 'block';
#     popup.style.top = posDOM.y + 'px';
#     popup.style.left = posDOM.x + 'px';
# }

# return network;
# """


# %% [markdown]
# ## Repr.


# %%
def show_treemap(
    df: pd.DataFrame,
    path_cols: list,
    val_col: str,
    len_cols: list = [],
    sum_cols: list = [],
):
    """
    Params
    ------
    df: pd.DataFrame
        Dataframe with info to show
    path_cols: list[str]
        Columns to aggregate by levels
    val_col: str
        Name of column that will be counted
    len_cols: list[str]
        Other columns that will be counted
    sum_cols: list[str]
        Columns that will be added
    """
    ids = []
    labels = []
    parents = []
    values = []
    custom_data = []

    # Append total
    df = df.dropna().replace("", None)
    df.insert(0, "Total", "Total")
    path_cols = ["Total"] + path_cols

    for i, c in enumerate(path_cols):
        group = (
            df.loc[:, path_cols[: i + 1] + [val_col] + len_cols + sum_cols]
            .groupby(path_cols[: i + 1])
            .agg(
                {
                    val_col: len,
                    **{k: len for k in len_cols},
                    **{k: sum for k in sum_cols},
                }
            )
            .reset_index()
        )
        ids.extend(
            group.iloc[:, : i + 1].astype(str).agg("/".join, axis=1).values.tolist()
        )
        labels.extend(group[c].to_list())
        parents.extend(
            group.iloc[:, :i].astype(str).agg("/".join, axis=1).values.tolist()
        )
        values.extend(group[val_col].to_list())
        custom_data.extend(group[len_cols + sum_cols].to_numpy().astype(float))

    custom_data = np.array(custom_data)
    # hovertemplate = "<i>%{label}</i>" + "<br><b>Count</b>: %{value}" + "<br><b>Count</b>: %{value}" + "<extra></extra>"
    hovertemplate = (
        "<i>%{label}</i>"
        + "<br><b>Count</b>: %{value}"
        + "<br><b>TotalAmount</b>: %{customdata[0]:.2f}"
        + "<extra></extra>"
    )

    # Set color map
    color = scale(custom_data[:, 0], 1, 10)
    # color = custom_data[:, 0]
    color = np.log(np.where(color <= 0, 1e-10, color))
    cmid = color.mean()
    cmin = color.min()

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            customdata=custom_data,
            ids=ids,
            # domain={'x': [0.0, 1.0], 'y': [1.0, 0.0]},
            branchvalues="total",
            hovertemplate=hovertemplate,
            marker=dict(
                colors=color,
                colorscale="RdBu",
                # cmid=cmid,
                cmin=cmin,
            ),
        )
    )

    fig.show()


# %%
path_cols = ["Date"] + tree_level_cols[2:6]
val_col = "Counts"
len_cols = []
sum_cols = ["Amount"]
show_treemap(
    admon_counts.loc[admon_counts["Date"] > 2019],
    path_cols,
    val_col,
    len_cols,
    sum_cols,
)

# %%
path_cols = date_col + tree_level_cols[2:]
val_col = "Counts"
len_cols = []
sum_cols = ["Amount"]
show_treemap(admon_counts, path_cols, val_col, len_cols, sum_cols)


# %%
