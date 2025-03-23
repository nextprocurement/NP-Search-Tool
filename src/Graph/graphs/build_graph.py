"""
This module builds a graph from a similarity matrix constructed from the Battacharyya distance between the topic distributions of the documents. The graph is visualized using the ForceAtlas2 layout algorithm and communities are detected using the Louvain method. The main topic of each document is used to color the nodes according to their community.
"""
import os
import pathlib
import argparse
from ast import literal_eval
from collections import Counter
from typing import List
import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns # type: ignore
import networkx as nx # type: ignore
import networkx.algorithms.community as nx_comm # type: ignore
from scipy import sparse
from fa2 import ForceAtlas2 # type: ignore
from kneed import KneeLocator # type: ignore
from gensim.matutils import corpus2csc # type: ignore

# This is a hack to avoid an execution error depending on the version of networkx. You can simply ignore but not remove it
if not hasattr(nx, "to_scipy_sparse_matrix"):
    def to_scipy_sparse_matrix(G, dtype='f', format='lil'):
        return nx.to_scipy_sparse_array(G)
nx.to_scipy_sparse_matrix = to_scipy_sparse_matrix

def get_doc_main_topc(doc_distr:np.ndarray):
    """Get the main topic of a document based on its topic distribution.
    
    Parameters:
    ----------
    doc_distr : np.ndarray
        Topic distribution of the document.
    
    Returns:
    -------
    int
        Index of the main topic.
    """
    sorted_tpc_indices = np.argsort(doc_distr)[::-1]
    top = sorted_tpc_indices[:1][0]
    return top

def save_graph_data(G: nx.Graph, output_path: str, filename_prefix: str):
    """Save graph data (edges and nodes) to CSV files.
    
    Parameters:
    ----------
    G : networkx.Graph
        The graph to save.
    output_path : str
        Directory where the CSV files will be saved.
    filename_prefix : str
        Prefix for the output CSV files.
    """
    # Save nodes
    nodes_filename = f"{output_path}/{filename_prefix}_nodes.csv"
    pd.DataFrame(list(G.nodes()), columns=['Node']).to_csv(nodes_filename, index=False)
    print(f"-- -- Nodes saved to {nodes_filename}")

    # Save edges with weights
    edges_filename = f"{output_path}/{filename_prefix}_edges.csv"
    edges_data = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
    pd.DataFrame(edges_data, columns=['Source', 'Target', 'Weight']).to_csv(edges_filename, index=False)
    print(f"-- -- Edges saved to {edges_filename}")
    
def visualize_graph(
    G: nx.Graph, 
    output_path: str, 
    filename: str, 
    topic_labels: List[str], 
    positions: dict, 
    dpi=300
):
    """
    Visualize a graph with community detection and topic coloring.

    Nodes are colored according to their community, and each community is labeled using its most frequent topic. The layout is defined by the given node positions. The output includes a static image of the graph and a CSV with node-community assignments.
    
    Parameters:
    ----------
    G : networkx.Graph
        The graph to visualize.
    output_path : str
        Directory where the image and CSV will be saved.
    filename : str
        Name of the output image file (should end in .png).
    topic_labels : list of str
        List of topic labels corresponding to each node.
    positions : dict
        Dictionary mapping node IDs to 2D positions.
    dpi : int, optional
        Resolution of the saved figure (default: 300).
    """
    
    # Detect communities
    communities = nx_comm.louvain_communities(G, seed=42)
    modularity = nx_comm.modularity(G, communities)
    nc = len(communities)

    print(f"-- -- Number of communities: {nc}")
    print(f"-- -- Modularity: {modularity}")

    # Assign color to each community
    palette = sns.color_palette("hsv", n_colors=nc)
    node2comm = {node: i for i, com in enumerate(communities) for node in com}
    node_colors = [palette[node2comm[node]] for node in G.nodes()]
    degrees = [G.degree(node) for node in G.nodes()]
    node_sizes = [np.log(deg + 1) * 20 for deg in degrees]

    # A community is assigned the most common topic label of its nodes
    community_labels = []
    for com in communities:
        topic_counter = Counter([topic_labels[node] for node in com])
        most_common_topic = topic_counter.most_common(1)[0][0]
        community_labels.append(most_common_topic)

    # Scale node positions and plot graph
    scaled_positions = {k: (v[0]*1.5, v[1]*1.5) for k, v in positions.items()}

    plt.figure(figsize=(14, 14), dpi=dpi)
    nx.draw_networkx_nodes(G, scaled_positions, node_size=node_sizes, node_color=node_colors, alpha=0.85)
    nx.draw_networkx_edges(G, scaled_positions, alpha=0.3, width=0.3)

    legend_elements = [
        Patch(facecolor=palette[i], label=community_labels[i]) for i in range(nc)
    ]
    plt.legend(
        handles=legend_elements,
        loc='lower left',
        fontsize='small',
        title='Comunidades (tópico dominante)',
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        borderpad=1
    )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_path}/{filename}", dpi=dpi, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    # Save community assignment to CSV
    node_data = []
    for node in G.nodes():
        comm_id = node2comm[node]
        topic_label = community_labels[comm_id]
        degree = G.degree(node)
        node_data.append({
            "node": node,
            "community": comm_id,
            "community_label": topic_label,
            "degree": degree
        })

    df_nodes = pd.DataFrame(node_data)
    csv_path = f"{output_path}/{filename.replace('.png', '')}_communities.csv"
    df_nodes.to_csv(csv_path, index=False)
    
def find_adaptive_threshold(
    S: sparse.spmatrix,
    elbow_adjust: float = 0.05,
    min_thr: float = 0.3,
    max_thr: float = 0.9
) -> float:
    """
    Compute an adaptive threshold for a similarity matrix using the elbow method.

    This function identifies the elbow point in the sorted similarity values
    (descending order) using the KneeLocator algorithm. The detected elbow
    is adjusted downward by a fixed amount to reduce over-connectivity.
    If no elbow is found, a fallback maximum threshold is used.

    Parameters
    ----------
    S : scipy.sparse.spmatrix
        Sparse symmetric similarity matrix.
    elbow_adjust : float, optional
        Amount to subtract from the detected elbow value. Default is 0.05.
    min_thr : float, optional
        Minimum allowed threshold. Default is 0.3.
    max_thr : float, optional
        Maximum allowed threshold. Default is 0.9.

    Returns
    -------
    thr : float
        Adaptively selected similarity threshold, clipped between `min_thr` and `max_thr`.
    """
    similarities = np.sort(S.data)[::-1]
    x = np.arange(len(similarities))

    # Detect elbow
    kneedle = KneeLocator(x, similarities, curve="convex", direction="decreasing")
    
    # If we find an elbow, adjust the threshold by a lowering factor
    if kneedle.knee is not None:
        thr = similarities[kneedle.knee] - elbow_adjust
        print(f"--> Elbow detected in {similarities[kneedle.knee]:.4f}, adjusting to {thr:.4f}")

    else:
        thr = max_thr  # Fallback to max threshold
        print(f"--> No elbow detected, using max threshold {thr:.4f}")

    # Ensure threshold is within bounds
    thr = np.clip(thr, min_thr, max_thr)
    
    return thr

def process_and_visualize(
    S: sparse.spmatrix, 
    df: pd.DataFrame, 
    sample_factor: float, 
    output_path: str, 
    gravity:int=50, 
    random_state:int=0, 
    dpi:int=300
):
    """
    Process the similarity matrix and visualize the graph.
    
    Parameters:
    ----------
    S : scipy.sparse.spmatrix
        Sparse symmetric similarity matrix.
    df : pd.DataFrame
        DataFrame with document data.
    sample_factor : float
        Factor by which the data was sampled.
    output_path : str
        Directory where the graph data will be saved.
    gravity : int, optional
        Gravity parameter for ForceAtlas2 layout algorithm (default: 50).
    random_state : int, optional
        Random seed for reproducibility (default: 0).
    dpi : int, optional
        Resolution of the saved figures (default: 300).
    """
    
    path_save = output_path / "graphs"
    os.makedirs(path_save, exist_ok=True)
    
    # save the matrix
    print("-- -- Number of non-zero components: ", S.nnz)
    nnz_prop = S.nnz / (S.shape[0] * S.shape[1])
    print("-- -- Proportion of non-zero components: ", nnz_prop)
    print("-- -- Proportion of zeros: ", 1 - nnz_prop)
    
    # ----------------------------------------------------------------
    # NOTE: We could simply use S without recalculating it, but we choose to do so to check whether the results improve, given that S is only an approximation. 
    X = [literal_eval(el) for el in df['thetas'].values.tolist()]
    X = corpus2csc(X).T
    n_topics = X.shape[1]
    print(f"-- -- Number of topics: {n_topics}")
    print(f"X: sparse matrix with {X.nnz} nonzero values out of {len(df) * n_topics}")
    print(X.shape)

    # Normalization
    X = sparse.csr_matrix(X / np.sum(X, axis=1))
    print(f"-- -- Average row sum: {np.mean(X.sum(axis=1).T)}")

    S = np.sqrt(X) * np.sqrt(X.T)
    # ----------------------------------------------------------------

    # NOTE: We could simply use S without recalculating it, but we choose to do so to check whether the results improve, given that S is only an approximation.
    print("-- -- Number of non-zero components: ", S.nnz)
    nnz_prop = S.nnz / (S.shape[0] * S.shape[1])
    print("-- -- Proportion of non-zero components: ", nnz_prop)
    print("-- -- Proportion of zeros: ", 1 - nnz_prop)

    # S is symmetric. Keep only upper triangular part
    S = sparse.triu(S, k=1)
    print('-- -- Number of non-zero components in S:', S.nnz)

    # Find threshold for similarity matrix using the elbow method
    thr = find_adaptive_threshold(S)

    # Apply the threshold to similarity matrix
    S.data[S.data < thr] = 0
    S.eliminate_zeros()
    print(f"-- -- Number of edges: {S.nnz}")
    print('-- -- Estimated number of links in full corpus:', len(S.data) / 2 / sample_factor**2)

    # Transform graph to networkx format
    G = nx.from_scipy_sparse_array(S)

    # Filter edges by weight
    edges_to_remove = [(u, v) for u, v, w in G.edges(data=True) if w['weight'] < thr]
    G.remove_edges_from(edges_to_remove)
    
    # Save the graph data
    save_graph_data(G, path_save, "graph")

    # Largest connected component (LCC) from the graph
    nodes = list(nx.connected_components(G))
    lcc = max(nodes, key=len)
    G_lcc = G.subgraph(lcc)
    positions_lcc = nx.spring_layout(G_lcc, seed=random_state)

    plt.figure(figsize=(10, 10), dpi=dpi)
    nx.draw(G_lcc, pos=positions_lcc, node_size=50, width=0.1)
    plt.savefig(f"{path_save}/draw_graph_lcc.png", dpi=dpi)

    # Compute positions using layout algorithm
    forceatlas2 = ForceAtlas2(gravity=gravity)
    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=200)
    G = G.subgraph(list(G.nodes()))
    valid_positions = {k: positions[k] for k in list(positions)}

    plt.figure(figsize=(10, 10), dpi=dpi)
    nx.draw_networkx_nodes(G, valid_positions, node_size=50, node_color="blue", alpha=0.7)
    nx.draw_networkx_edges(G, valid_positions, edge_color="green", alpha=0.1)
    plt.axis('off')
    plt.savefig(f"{path_save}/draw_graph_forceatlas2.png", dpi=dpi)

    topic_labels = df['label'].tolist()
    visualize_graph(G_lcc, path_save, f"draw_graph_communities.png", topic_labels, positions_lcc, dpi)

def main(
    model_path:str,
    n_docs:int=10000,
    gravity:int=30,
    random_state:int=0,
    dpi:int=300,
    use_sample:bool=True
):
    model_path = pathlib.Path(model_path)
    S = sparse.load_npz(model_path / "model_data/TMmodel/distances.npz")
    thetas = sparse.load_npz(model_path / "model_data/TMmodel/thetas.npz").toarray()
    
    corpusFile = model_path / "train_data/corpus.txt"
    with corpusFile.open("r", encoding="utf-8") as f:
        lines = f.readlines()  
        f.seek(0)
        try:
            documents_ids = [line.rsplit(" 0 ")[0].strip() for line in lines]
        except:
            documents_ids = [line.rsplit("\t0\t")[0].strip() for line in lines]
            
    df_corpus = pd.DataFrame({"doc_id": documents_ids})
    df_corpus["thetas"] = list(thetas)    
    df_corpus["main_topic"] = df_corpus["thetas"].apply(get_doc_main_topc)
    
    if use_sample:
        # Take a sample of documents
        sample_factor = n_docs / len(df_corpus)
        df_sample = df_corpus.sample(n_docs, random_state=random_state)
        print(f"Dataset reduced to {n_docs} documents")
    else:
        df_sample = df_corpus
        sample_factor = 1
        n_docs = len(df_corpus)
    
    with open(model_path /"model_data/TMmodel/tpc_labels.txt", 'r') as file:
        lines = file.readlines()
    topic_labels = [line.strip() for line in lines]
    
    def stringfy_thetas(thetas):
        thetas_non = [(i,float(theta)) for i,theta in enumerate(thetas) if float(theta) != 0.0]
        return str(thetas_non)
    df_sample["thetas"] = df_sample["thetas"].apply(stringfy_thetas)
    
    print(f"Topic labels for {model_path}: {topic_labels}")
    
    # save label for each doc corresponding to the "main topic"
    df_sample["label"] = df_corpus["main_topic"].apply(lambda x: topic_labels[x])
    
    process_and_visualize(S, df_sample, sample_factor, model_path, gravity, random_state, dpi)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and visualize network data.")
    parser.add_argument(
        '--path_model',
        type=str,
        required=False, 
        help="Path to the data file",
        default="/data/source/45_large"
    )
    parser.add_argument(
        '--n_docs',
        type=int,
        default=200,
        help="Number of documents to sample")
    parser.add_argument(
        '--gravity',
        type=int,
        default=50,
        help="Gravity parameter for ForceAtlas2"
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=0,
        help="Random state for reproducibility"
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help="DPI for saving figures"
    )
    parser.add_argument(
        '--use_sample',
        type=bool,
        default=False,
        help="Use a sample of the data"
    )

    args = parser.parse_args()
    main(args.path_model, args.n_docs, args.gravity, args.random_state, args.dpi, args.use_sample)