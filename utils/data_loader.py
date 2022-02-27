import pandas as pd
import random
from utils.model_utils import *


def ACGCN_MMP_Dataset(args, data, is_train, drop_last=False):

    device = args['DEVICE']

    if is_train:
        data_swap = data.copy()
        data_swap.smiles1, data_swap.smiles2 = data.smiles1, data.smiles2
        data = pd.concat((data, data_swap))
        data = data.reset_index()
        data = data[data.columns[1:]]
        
    dgl_graph_smiles1, dgl_graph_smiles2 = [], []
    
    for smiles1, smiles2 in zip(data.smiles1, data.smiles2):
        dgl_graph_smiles1.append(create_graph(smiles1, device))
        dgl_graph_smiles2.append(create_graph(smiles2, device))
        
    data_set = [[{'GRAPH_SMILES1': dgl_graph_smiles1[i], 'GRAPH_SMILES2': dgl_graph_smiles2[i]}] for i in range(len(data))]
    y = list(data.label)

    # Shuffle dataset
    if is_train:
        rd_idx = random.sample(range(len(data)), len(data))
        data_set = [data_set[i] for i in rd_idx]
        y = [y[i] for i in rd_idx]

    batch_size = args['BATCH_SIZE']
    remainder = len(data_set) % batch_size
    splited_data_set = np.array_split(data_set[remainder:], int(len(data_set)/batch_size))
    splited_data_set = [list(x) for x in splited_data_set]
    if remainder != 0:
        if not drop_last:
            splited_data_set.append(data_set[:remainder])
    
    splited_y = np.array_split(y[remainder:], int(len(data_set)/batch_size))
    splited_y = [list(y) for y in splited_y]
    if remainder != 0:
        if not drop_last:  
            splited_y.append(y[:remainder])
    splited_y = [torch.from_numpy(np.array(label)) for label in splited_y]

    loader = [(i, (splited_data_set[i]), splited_y[i]) for i in range(len(splited_data_set))]
        
    return loader


def ACGCN_SUB_Dataset(args, data, is_train, drop_last=False):

    device = args['DEVICE']

    if is_train:
        data_swap = data.copy()
        data_swap.substituent1, data_swap.substituent2 = data.substituent1, data.substituent2
        data = pd.concat((data, data_swap))
        data = data.reset_index()
        data = data[data.columns[1:]]

    dgl_graph_core, dgl_graph_sub1, dgl_graph_sub2 = [], [], []

    for core, sub1, sub2 in zip(data.core, data.substituent1, data.substituent2):

        dgl_graph_core.append(create_graph(core, device))
        dgl_graph_sub1.append(create_graph(sub1, device))
        dgl_graph_sub2.append(create_graph(sub2, device))
        
    data_set = [[{'GRAPH_CORE': dgl_graph_core[i], 'GRAPH_SUB1': dgl_graph_sub1[i], 'GRAPH_SUB2': dgl_graph_sub2[i]}] for i in range(len(data))]
    y = list(data.label)

    if is_train:
        rd_idx = random.sample(range(len(data)), len(data))
        data_set = [data_set[i] for i in rd_idx]
        y = [y[i] for i in rd_idx]
        
    batch_size = args['BATCH_SIZE']
    remainder = len(data_set) % batch_size
    splited_data_set = np.array_split(data_set[remainder:], int(len(data_set)/batch_size))
    splited_data_set = [list(x) for x in splited_data_set]
    if remainder != 0:
        if not drop_last:
            splited_data_set.append(data_set[:remainder])
    
    splited_y = np.array_split(y[remainder:], int(len(data_set)/batch_size))
    splited_y = [list(x) for x in splited_y]

    if remainder != 0:
        if not drop_last:
            splited_y.append(y[:remainder])
    splited_y = [torch.from_numpy(np.array(l)) for l in splited_y]
    
    loader = [(i, (splited_data_set[i]), splited_y[i]) for i in range(len(splited_data_set))]
        
    return loader
