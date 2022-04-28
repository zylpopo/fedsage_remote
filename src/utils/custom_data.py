import numpy as np
import pandas as pd
import stellargraph as sg
import torch
import torch.nn.functional
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn import preprocessing
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, CitationFull, Coauthor, Amazon
from torch_geometric.loader import DataLoader

from src.global_task import Global


def load(folder, num_clients, batch_size):
    subgraph_dict = {"subgraph": [], "train_idx": [], "val_idx": [], "test_idx": []}
    name_list = ["x", "y", "edge_index", "train_idx", "val_idx", "test_idx"]

    for client_id in range(num_clients):
        vals = {}
        for name in name_list:
            vals[name] = np.load(folder+"/client_{}_{}.npy".format(client_id, name))

        subgraph = Data(x=torch.from_numpy(vals["x"]),
                        y=torch.from_numpy(vals["y"]),
                        edge_index=torch.from_numpy(vals["edge_index"].T))

        subgraph_loader = DataLoader(subgraph, batch_size=batch_size, shuffle=True, pin_memory=True)

        train_idx, val_idx, test_idx = list(map(lambda x: torch.LongTensor(x),
                                                [vals["train_idx"],
                                                 vals["val_idx"],
                                                 vals["test_idx"]]))

        subgraph_dict["subgraph"].append(subgraph_loader)
        subgraph_dict["train_idx"].append(train_idx)
        subgraph_dict["val_idx"].append(val_idx)
        subgraph_dict["test_idx"].append(test_idx)


    global_train_idx = np.load(folder + "/global_train_idx.npy")
    global_val_idx = np.load(folder + "/global_val_idx.npy")
    global_test_idx = np.load(folder + "/global_test_idx.npy")
    global_train_idx, global_val_idx, global_test_idx = list(map(lambda x: torch.LongTensor(x),
                                                [global_train_idx,
                                                 global_val_idx,
                                                 global_test_idx]))

    return subgraph_dict, global_train_idx, global_val_idx, global_test_idx


def obtain_global_node_idx_in_each_client(subgraph_dict, global_train_idx, global_val_idx, global_test_idx):
    num_clients = len(subgraph_dict["subgraph"])

    idx_map = {client_id:{} for client_id in range(num_clients)}

    global_train_ptr = 0
    global_val_ptr = 0
    global_test_ptr = 0

    for client_id in range(num_clients):
        for local_train_idx in subgraph_dict["train_idx"][client_id]:
            idx_map[client_id][local_train_idx.item()] = global_train_idx[global_train_ptr].item()
            global_train_ptr += 1
        for local_val_idx in subgraph_dict["val_idx"][client_id]:
            idx_map[client_id][local_val_idx.item()] = global_val_idx[global_val_ptr].item()
            global_val_ptr += 1
        for local_test_idx in subgraph_dict["test_idx"][client_id]:
            idx_map[client_id][local_test_idx.item()] = global_test_idx[global_test_ptr].item()
            global_test_ptr += 1

    return idx_map

def _obtain_global_graph_pyg(data_name):
    data_root = "/home/ubuntu/work/HIntraSGraphFL/datasets"
    if data_name in ["cora", "citeseer", "PubMed"]:
        pyg_dataset = Planetoid(data_root, data_name)
        num_classes = pyg_dataset.num_classes
        feature_dimension = pyg_dataset.num_features
        pyg_data = pyg_dataset.data

    elif data_name in ["ogbn-products", "ogbn-arxiv"]:
        pyg_dataset = PygNodePropPredDataset(root = data_root, name = data_name)
        num_classes = pyg_dataset.num_classes
        feature_dimension = pyg_dataset.num_features
        pyg_data = pyg_dataset.data
        pyg_data.y = pyg_data.y.reshape(-1)

    elif data_name in ["cora_ml", "dblp"]:
        pyg_dataset = CitationFull(data_root, data_name)
        num_classes = pyg_dataset.num_classes
        feature_dimension = pyg_dataset.num_features
        pyg_data = pyg_dataset.data

    elif data_name in ["CS", "Physics"]:
        pyg_dataset = Coauthor(data_root, data_name)
        num_classes = pyg_dataset.num_classes
        feature_dimension = pyg_dataset.num_features
        pyg_data = pyg_dataset.data
    elif data_name in ["Computers", "Photo"]:
        pyg_dataset = Amazon(data_root, data_name)
        num_classes = pyg_dataset.num_classes
        feature_dimension = pyg_dataset.num_features
        pyg_data = pyg_dataset.data

    return pyg_data, num_classes, feature_dimension


def _obtain_node_data(x, y, column_names):
    subj = ["subject_{}".format(class_id) for class_id in y.numpy()]
    node_data = pd.DataFrame(x.numpy(), columns=column_names[:-1])
    node_data[column_names[-1]] = subj
    return node_data # DataFrame:(num_nodes, fts_dim+1)

def _obtain_edgelist(edge_index):
    source_p2 = edge_index.numpy()[0]
    target_p2 = edge_index.numpy()[1]
    remove_self_loop = set()
    source = []
    target = []

    for edge_id in range(edge_index.numpy().shape[1]):
        if (target_p2[edge_id], source_p2[edge_id]) not in remove_self_loop:
                remove_self_loop.add( (source_p2[edge_id], target_p2[edge_id]) )
                source.append(source_p2[edge_id])
                target.append(target_p2[edge_id])

    edgelist = pd.DataFrame()
    edgelist["source"] = source
    edgelist["target"] = target

    return edgelist


def obtain_custom_data(data_name, folder, num_clients, batch_size):
    subgraph_dict, global_train_idx, global_val_idx, global_test_idx = load(folder=folder, num_clients=num_clients, batch_size=batch_size)
    idx_map = obtain_global_node_idx_in_each_client(subgraph_dict, global_train_idx, global_val_idx, global_test_idx)
    owner_node_ids = {owner_id:[] for owner_id in range(num_clients)}
    for owner_i in range(num_clients):
        local_len = len(idx_map[owner_i])
        for local_idx in range(local_len):
            owner_node_ids[owner_i].append(idx_map[owner_i][local_idx])

    # obtain global graph
    pyg_data, num_classes, fts_dim = _obtain_global_graph_pyg(data_name)
    pyg_x = pyg_data.x
    pyg_y = pyg_data.y
    pyg_edge_index = pyg_data.edge_index

    # obtain global node_data, edgelist, subjects, features
    subject = "subject"
    feature_names = ["w_{}".format(fts_id) for fts_id in range(fts_dim)]
    column_names = feature_names + [subject]

    global_node_data = _obtain_node_data(pyg_x, pyg_y, column_names)
    global_features = global_node_data[feature_names]
    global_edgelist = _obtain_edgelist(pyg_edge_index)
    global_subjects = global_node_data[subject]

    # obtain local graphs
    whole_graph = sg.StellarGraph({"entity": global_features}, {"relation": global_edgelist})

    edges = np.copy(whole_graph.edges())
    df = pd.DataFrame()
    df['source'] = [edge[0] for edge in edges]
    df['target'] = [edge[1] for edge in edges]

    nodes_id = whole_graph.nodes()
    local_G = []
    local_node_subj = []
    local_nodes_ids = []
    target_encoding = preprocessing.LabelBinarizer()
    target = target_encoding.fit_transform(global_subjects)
    local_target = []


    for owner_i in range(num_clients):
        partition_i = owner_node_ids[owner_i]
        locs_i = whole_graph.node_ids_to_ilocs(partition_i)
        sbj_i = global_subjects.copy(deep=True)
        sbj_i.values[:] = "" if global_subjects.values[0].__class__ == str else 0
        sbj_i.values[locs_i] = global_subjects.values[locs_i]

        local_node_subj.append(sbj_i)
        local_target_i = np.zeros(target.shape, np.int32)
        local_target_i[locs_i] += target[locs_i]
        local_target.append(local_target_i)
        local_nodes_ids.append(partition_i)

        feats_i = np.zeros(whole_graph.node_features().shape)
        feats_i[locs_i] = feats_i[locs_i] + whole_graph.node_features()[locs_i]

        nodes = sg.IndexedArray(feats_i, nodes_id)
        graph_i = sg.StellarGraph(nodes=nodes, edges=df)
        local_G.append(graph_i)


    all_classes = target_encoding.classes_  # [1, 2, 3, ..., num_classes]
    global_task = Global(whole_graph, global_subjects, target)


    # train, test subject split
    train_subjects = []
    test_subjects = []

    for owner_i in range(num_clients):
        local_train_idx = subgraph_dict["train_idx"][owner_i].numpy().tolist()
        local_test_idx = subgraph_dict["test_idx"][owner_i].numpy().tolist()
        local_train_idx_in_global = [ idx_map[owner_i][idx] for idx in local_train_idx]
        local_test_idx_in_global = [ idx_map[owner_i][idx] for idx in local_test_idx]

        train_locs_i = whole_graph.node_ids_to_ilocs(local_train_idx_in_global)
        train_sbj_i = global_subjects.copy(deep=True)
        train_sbj_i.values[train_locs_i] = global_subjects.values[train_locs_i]
        train_sbj_i = train_sbj_i[train_locs_i]
        train_subjects.append(train_sbj_i)

        test_locs_i = whole_graph.node_ids_to_ilocs(local_test_idx_in_global)
        test_sbj_i = global_subjects.copy(deep=True)
        test_sbj_i.values[test_locs_i] = global_subjects.values[test_locs_i]
        test_sbj_i = test_sbj_i[test_locs_i]
        test_subjects.append(test_sbj_i)



    return local_G, local_node_subj, local_target, local_nodes_ids, all_classes, global_task, train_subjects, test_subjects

