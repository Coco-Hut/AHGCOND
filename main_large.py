import os
import time
import math
import torch
import random
from tqdm import trange
import argparse

import numpy as np
import scipy.sparse as sp
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from layers import *
from models import *
from preprocessing import *
from train_utils import *
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from convert_datasets_to_pygDataset import dataset_Hypergraph
from utils_cond import match_loss,save_reduced,ReparamModule
from parameter_parser import args
from ahgcond import *

import warnings
warnings.filterwarnings('ignore')

# ============= Seeding ==================#
def set_seed(seed, cuda):
    print('Set Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda in [0,1]:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(args.seed,args.cuda)

if __name__=='__main__':

    if args.dname in ['magpm']:
        data_path=f'../data/magsm/'
    
    data=load_large_graph(data_path)

    args.num_features = data.x.shape[1]
    args.num_classes = len(data.y.unique())
    data.y = data.y - data.y.min()
    
    if not hasattr(data, 'n_x'):
        data.n_x = torch.tensor([data.x.shape[0]])
    if not hasattr(data, 'num_hyperedges'):
        # note that we assume the he_id is consecutive.
        data.num_hyperedges = torch.tensor(
            [data.edge_index[1].max()-data.n_x+1])

    # Data Spliting
    split_idx_lst = []
    for run in range(args.runs):
        split_idx = rand_train_test_idx(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
        split_idx_lst.append(split_idx)

    split_idx=split_idx_lst[0] 
    
    model = parse_method(args, data)
    # put things to device
    if args.cuda in [0, 1]:
        device = torch.device('cuda:'+str(args.cuda)
                                if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    model, data = model.to(device), data.to(device)
    if args.method == 'UniGCNII':
        args.UniGNN_degV = args.UniGNN_degV.to(device)
        args.UniGNN_degE = args.UniGNN_degE.to(device)

    num_params = count_parameters(model)
    
    gc=AHGCOND(data, split_idx, args, device)
    
    start_time = time.time()
    gc.reduce()
    end_time = time.time()
    print(f'The condenssation time is {end_time-start_time}')
    