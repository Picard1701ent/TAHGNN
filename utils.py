import os
import random
from itertools import combinations
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import torch
from nilearn import connectome

import config

cfg = SimpleNamespace(**vars(config))


def calculate_dist_matrix(features):
    sum_square = np.sum(np.square(features), axis=1, keepdims=True)
    dist_matrix = sum_square + sum_square.T - 2 * np.dot(features, features.T)
    dist_matrix = np.sqrt(np.maximum(dist_matrix, 0))
    return dist_matrix


def calculate_degree_matrix(adj_matrix):
    degrees = np.sum(adj_matrix, axis=1)
    degree_matrix = np.diag(degrees)
    return degree_matrix


def calculate_node_distance_matrix(adj_matrix):
    dist_matrix = scipy.sparse.csgraph.shortest_path(
        adj_matrix, method="D", unweighted=True
    )
    return dist_matrix


def calculate_hypergraph_matrix(dist_matrix):
    V = dist_matrix.shape[0]
    hyperedges = []

    for i in range(V):
        nearest_indices = np.argsort(dist_matrix[i])[1 : cfg.k + 1]
        hyperedges.append([i] + nearest_indices.tolist())

    E = len(hyperedges)
    hyper_incidence_matrix = np.zeros((V, E))
    for e_idx, edge in enumerate(hyperedges):
        for node in edge:
            hyper_incidence_matrix[node, e_idx] = 1

    return hyper_incidence_matrix, hyperedges


def convert_node_strength_format(node_strength, hypergraph_matrix, hyperedge):
    weighted_hypergraph_matrix = hypergraph_matrix.copy()
    for i in range(hypergraph_matrix.shape[1]):
        for j in range(len(hyperedge[i])):
            if isinstance(hyperedge, np.ndarray):
                weighted_hypergraph_matrix[hyperedge[i, j], i] = node_strength[i, j]
            else:
                weighted_hypergraph_matrix[hyperedge[i][j], i] = node_strength[i][j]

    return weighted_hypergraph_matrix


def calculate_node_strength(
    dist_matrix, structure_dist_matrix, hyperedges, hypergraph_matrix
):
    feature_similarity_matrix = 1.0 / (dist_matrix + cfg.eps)
    feature_similarity_matrix *= calculate_depression(structure_dist_matrix, 1.1)
    if isinstance(hyperedges, list):
        if all(len(edge) == len(hyperedges[0]) for edge in hyperedges):
            hyperedges = np.array(hyperedges, dtype=np.int32)
        else:
            hyperedges = [
                np.array(hyperedge, dtype=np.int32) for hyperedge in hyperedges
            ]
    node_strengths = []
    for hyperedge in hyperedges:
        structure_dist = structure_dist_matrix[np.ix_(hyperedge, hyperedge)]
        structrue_connection_weight = 1.0 / (structure_dist + cfg.eps)
        np.fill_diagonal(structrue_connection_weight, 0.0)
        similarity = feature_similarity_matrix[np.ix_(hyperedge, hyperedge)]
        np.fill_diagonal(similarity, 0.0)
        node_strength = similarity + structrue_connection_weight * cfg.alpha
        node_strength = np.sum(node_strength, axis=0)
        node_strengths.append(node_strength)
    try:
        node_strengths = np.array(node_strengths, dtype=np.float32)
    except:
        node_strengths = node_strengths
    node_strengths = convert_node_strength_format(
        node_strengths, hypergraph_matrix, hyperedges
    )
    return node_strengths


def get_shortest_path(graph, source, target):
    try:
        path = nx.shortest_path(graph, source=source, target=target)
        return path
    except nx.NetworkXNoPath:
        return [float("inf")]


def graph_shortest_path(graph, hyperedge):
    pair_combinations = list(combinations(hyperedge, 2))
    shortest_paths = []
    for pair in pair_combinations:
        shortest_paths.append(get_shortest_path(graph, pair[0], pair[1]))
    return shortest_paths, pair_combinations


def calculate_gloable_efficiency(dist_matrix, structure_connection_matrix, hyperedges):
    feature_similarity_matrix = 1.0 / (dist_matrix + cfg.eps)
    structure_distance = calculate_node_distance_matrix(structure_connection_matrix)
    feature_similarity_matrix = feature_similarity_matrix * calculate_depression(
        structure_distance, 1.1
    )
    if isinstance(hyperedges, list):
        if all(len(edge) == len(hyperedges[0]) for edge in hyperedges):
            hyperedges = np.array(hyperedges, dtype=np.int32)
        else:
            hyperedges = [
                np.array(hyperedge, dtype=np.int32) for hyperedge in hyperedges
            ]
    G = nx.from_numpy_array(structure_connection_matrix, create_using=nx.Graph)
    gloable_efficiencies = []
    for i, hyperedge in enumerate(hyperedges):
        num_node = hyperedge.shape[-1]
        shortest_paths, pair_combinations = graph_shortest_path(G, hyperedge)
        weighted_dists = []
        for shortest_path, pair_combination in zip(shortest_paths, pair_combinations):
            if float("inf") in shortest_path:
                weighted_dists.append(
                    20.0
                    + cfg.gamma
                    * feature_similarity_matrix[
                        pair_combination[0], pair_combination[1]
                    ]
                )
            else:
                weighted_dist = 0
                for i in range(len(shortest_path) - 1):
                    weighted_dist += 1.0

                weighted_dists.append(
                    weighted_dist
                    + cfg.gamma
                    * feature_similarity_matrix[
                        pair_combination[0], pair_combination[1]
                    ]
                )
        weighted_dists = np.array(weighted_dists)
        gloable_efficiency = np.sum(1.0 / weighted_dists) / (num_node * (num_node - 1))
        gloable_efficiencies.append(gloable_efficiency)
    gloable_efficiencies = np.array(gloable_efficiencies)
    return gloable_efficiencies


def compute_structure_info(features, structure):
    dist_matrix = calculate_dist_matrix(features)
    structure_dist_matrix = calculate_node_distance_matrix(structure)
    m_dist_matrix = modified_dist_matrix(dist_matrix, structure_dist_matrix)
    hypergraph, hyperedges = calculate_hypergraph_matrix(m_dist_matrix)
    node_strength = calculate_node_strength(
        dist_matrix, structure_dist_matrix, hyperedges, hypergraph
    )
    gloable_efficiency = calculate_gloable_efficiency(
        dist_matrix, structure, hyperedges
    )
    return hypergraph, node_strength, gloable_efficiency, dist_matrix


def load_featuresNstructures():
    pos_structure = cfg.pos_structure
    neg_structure = cfg.neg_structure

    pos_features = cfg.pos_features
    neg_features = cfg.neg_features

    pos_name_list = os.listdir(pos_features)
    neg_name_list = os.listdir(neg_features)

    pos_label = np.ones(len(pos_name_list))
    neg_label = np.zeros(len(neg_name_list))

    label = np.concatenate([pos_label, neg_label])
    features = []
    structures = []
    for pos_name in pos_name_list:
        code = pos_name[-8:-4]
        code = code.zfill(5)
        dti_name = f"{code}_dti_FACT_45_02_1_0_Matrix_FN_AAL_Contract_90_2MM_90.txt"
        dti = np.loadtxt(os.path.join(pos_structure, dti_name))
        fmri = np.loadtxt(os.path.join(pos_features, pos_name))
        fmri = subject_connectivity(fmri)
        features.append(fmri)
        structures.append(dti)

    for neg_name in neg_name_list:
        code = neg_name[-8:-4]
        code = code.zfill(5)
        dti_name = f"{code}_dti_FACT_45_02_1_0_Matrix_FN_AAL_Contract_90_2MM_90.txt"
        dti = np.loadtxt(os.path.join(neg_structure, dti_name))
        fmri = np.loadtxt(os.path.join(neg_features, neg_name))
        fmri = subject_connectivity(fmri)

        features.append(fmri)
        structures.append(dti)

    features = np.array(features)
    structures = np.array(structures)
    return features, structures, label


def subject_connectivity(timeseries, kind="correlation"):
    if kind in ["tangent", "partial correlation", "correlation"]:
        conn_measure = connectome.ConnectivityMeasure(
            kind=kind, standardize="zscore_sample"
        )
        connectivity = conn_measure.fit_transform([timeseries])[0]
    return connectivity


def load_data():
    features, structures, labels = load_data2()
    hypergraphs, node_strengths, gloable_efficiencys, dist_matrixs = [], [], [], []
    for feature, structure in zip(features, structures):
        hypergraph, node_strength, gloable_efficiency, dist_matrix = (
            compute_structure_info(feature, structure)
        )
        hypergraphs.append(hypergraph)
        node_strengths.append(node_strength)
        gloable_efficiencys.append(gloable_efficiency)
        dist_matrixs.append(dist_matrix)
    dist_matrixs = np.array(dist_matrixs)
    hypergraphs = np.array(hypergraphs)
    node_strengths = np.array(node_strengths)
    gloable_efficiencys = np.array(gloable_efficiencys)
    return (
        features,
        labels,
        structures,
        hypergraphs,
        node_strengths,
        gloable_efficiencys,
        dist_matrixs,
    )


def softmax(x, axis=None):
    shift_x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shift_x)
    softmax_x = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    return softmax_x


def gaussian_kenel(distance_matrix: torch.Tensor, sigma=1):
    return torch.exp(-(distance_matrix**2) / (2 * sigma**2))


def modified_dist_matrix(dist_matrix, structure_dist):
    structure_sim = np.where(structure_dist == 0, 1.0, 1.0 / structure_dist)

    structure_sim = softmax(structure_sim, axis=-1)
    dist_sim = np.where(dist_matrix == 0, 1.0, 1.0 / dist_matrix)
    dist_sim = softmax(dist_sim, axis=-1)
    combine_dist_sim = cfg.beta * dist_sim + structure_sim
    combine_dist_matrix = 1 / combine_dist_sim
    np.fill_diagonal(combine_dist_matrix, 0)
    return combine_dist_matrix


def calculate_depression(structure_dist_matrix, l="exp"):
    l = cfg.sigma
    structure_dist_matrix = np.where(
        np.isinf(structure_dist_matrix), 20, structure_dist_matrix
    )
    if l == "exp":
        return np.exp(-structure_dist_matrix)
    elif isinstance(l, (int, float)) and l > 0:
        return l ** (-structure_dist_matrix)
    else:
        raise ValueError("Parameter l must be 'exp' or a positive number.")


def modify_symmetric_matrix(matrix):
    n, m = matrix.shape
    if n != m:
        raise ValueError("Input matrix must be square")

    if not np.allclose(matrix, matrix.T):
        raise ValueError("Input matrix must be symmetric")

    triu_indices = np.triu_indices(n, k=1)
    upper_triangular_values = matrix[triu_indices]
    zero_indices = np.where(upper_triangular_values == 0)[0]

    positive_indices = np.where(upper_triangular_values > 0)[0]

    upper_triangular_values[positive_indices] = 0
    rand_iteration = np.random.randint(20, 80)
    for i in range(rand_iteration):
        random_idx = np.random.choice(zero_indices)

        random_value = np.random.randint(1, 11)

        upper_triangular_values[random_idx] = random_value

    modified_matrix = np.zeros((n, n))
    modified_matrix[triu_indices] = upper_triangular_values

    modified_matrix = modified_matrix + modified_matrix.T

    np.fill_diagonal(modified_matrix, np.diag(matrix))

    return modified_matrix


def load_data2():
    if cfg.dataset == "ADNC":
        pos_features = "dataset/AD"
        neg_features = "dataset/NC"
        pos_dti = "dataset/AD_dti"
        neg_dti = "dataset/NC_dti"
    elif cfg.dataset == "ADMCI":
        pos_features = "dataset/AD"
        neg_features = "dataset/MCI"
        pos_dti = "dataset/AD_dti"
        neg_dti = "dataset/MCI_dti"
    elif cfg.dataset == "MCINC":
        pos_features = "dataset/MCI"
        neg_features = "dataset/NC"
        pos_dti = "dataset/MCI_dti"
        neg_dti = "dataset/NC_dti"
    pos_file_dirs = os.listdir(pos_features)
    neg_file_dirs = os.listdir(neg_features)

    labels = []
    datas = []
    pos_datas = []
    neg_datas = []
    for pos_file_dir in pos_file_dirs:
        _, ext = os.path.splitext(pos_file_dir)
        if ext == ".mat":
            pos_file = scipy.io.loadmat(os.path.join(pos_features, pos_file_dir))
            key = pos_file.keys()
            if len(key) > 10:
                pos_data = process_mat_data(pos_file)
            else:
                pos_data = pos_file[list(pos_file.keys())[-1]][:, :90]
            pos_data = subject_connectivity(pos_data)
            datas.append(pos_data)
            pos_datas.append(pos_data)

        elif ext == ".txt":
            pos_data = np.loadtxt(os.path.join(pos_features, pos_file_dir))[:, :90]
            pos_data = subject_connectivity(pos_data)
            datas.append(pos_data)
            pos_datas.append(pos_data)

        elif ext == ".csv":
            df_preview = pd.read_csv(os.path.join(pos_features, pos_file_dir), nrows=2)
            num_columns = df_preview.shape[1]
            usecols = range(1, num_columns)
            pos_data = pd.read_csv(
                os.path.join(pos_features, pos_file_dir), skiprows=1, usecols=usecols
            )
            pos_data = pos_data.apply(pd.to_numeric, errors="coerce")
            pos_data = pos_data.dropna().values[:, :90]
            pos_data = subject_connectivity(pos_data)
            datas.append(pos_data)
            pos_datas.append(pos_data)
        labels.append(1)

    for neg_file_dir in neg_file_dirs:
        _, ext = os.path.splitext(neg_file_dir)

        if ext == ".mat":
            neg_file = scipy.io.loadmat(os.path.join(neg_features, neg_file_dir))
            key = neg_file.keys()
            if len(key) > 10:
                neg_data = process_mat_data(neg_file)
            else:
                neg_data = neg_file[list(neg_file.keys())[-1]][:, :90]

            neg_data = subject_connectivity(neg_data)
            datas.append(neg_data)
            neg_datas.append(neg_data)
        elif ext == ".txt":
            neg_data = np.loadtxt(os.path.join(neg_features, neg_file_dir))[:, :90]
            neg_data = subject_connectivity(neg_data)
            datas.append(neg_data)
            neg_datas.append(neg_data)

        elif ext == ".csv":
            df_preview = pd.read_csv(os.path.join(neg_features, neg_file_dir), nrows=2)
            num_columns = df_preview.shape[1]
            usecols = range(1, num_columns)
            neg_data = pd.read_csv(
                os.path.join(neg_features, neg_file_dir), skiprows=1, usecols=usecols
            )
            neg_data = neg_data.apply(pd.to_numeric, errors="coerce")
            neg_data = neg_data.dropna().values[:, :90]
            neg_data = subject_connectivity(neg_data)
            datas.append(neg_data)
            neg_datas.append(neg_data)

        labels.append(0)
    pos_data_structures = []
    for pos_file_dir in pos_file_dirs:
        pos_file_dir = pos_file_dir[:-4] + ".txt"
        data_structure = np.loadtxt(os.path.join(pos_dti, pos_file_dir))
        data_structure = np.where(data_structure != 0, 1, 0)
        pos_data_structures.append(data_structure)
        if data_structure.shape[0] != 90:
            print(pos_file_dir)
            print(data_structure.shape)

    neg_data_structures = []
    for neg_file_dir in neg_file_dirs:
        neg_file_dir = neg_file_dir[:-4] + ".txt"
        data_structure = np.loadtxt(os.path.join(neg_dti, neg_file_dir))
        data_structure = np.where(data_structure != 0, 1, 0)
        neg_data_structures.append(data_structure)
        if data_structure.shape[0] != 90:
            print(neg_file_dir)
            print(data_structure.shape)

    data_structures = pos_data_structures + neg_data_structures
    data_structures = np.stack(data_structures, axis=0)
    datas = np.stack(datas, axis=0)
    labels = np.array(labels)

    return datas, data_structures, labels


def shuffle_list(input_list):
    shuffled_list = input_list.copy()

    for i in range(len(shuffled_list) - 1, 0, -1):
        j = random.randint(0, i)
        shuffled_list[i], shuffled_list[j] = shuffled_list[j], shuffled_list[i]

    return shuffled_list


def process_mat_data(data):
    keys = list(data.keys())

    all_data_list = []

    for key in keys:
        if key != "Time" and not key.startswith("__"):
            field_data = data[key]
            all_data_list.append(field_data.flatten())
    all_data_matrix = np.vstack(all_data_list).T

    return all_data_matrix
