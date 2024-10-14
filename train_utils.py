#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import scipy.sparse as sp
import torch_sparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import pickle as pkl
from layers import *
from models import *
from preprocessing import *
from torch_sparse import SparseTensor,mul
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')

# =================== Generalization Utils ==================#

def print_statistics(data,raw_cond_data,reduced_data):

      # case study: condensed data analysis
      print(f'Original Node number: {data.x.shape[0]}\n',
            f'Original Hyperedge number: {torch.unique(data.edge_index[1]).numel()-data.x.shape[0]}\n',
            f'Original Sparsity: {data.edge_index.shape[1]/(data.x.shape[0]*(torch.unique(data.edge_index[1]).numel()))}\n',
            f'Initial Node number: {raw_cond_data.edge_index.shape[0]}\n',
            f'Initial Hyperedge: {raw_cond_data.edge_index.shape[1]}\n',
            f'Condensed Node number: {torch.unique(reduced_data.edge_index[0]).numel()}\n',
            f'Isolated Node number: {raw_cond_data.edge_index.shape[0]-torch.unique(reduced_data.edge_index[0]).numel()}\n',
            f'Condensed Hyperedge: {torch.unique(reduced_data.edge_index[1]).numel()}\n',
            f'Incidence Number: {reduced_data.edge_index.shape[1]}\n'
            f'Sparsity: {reduced_data.edge_index.shape[1]/(raw_cond_data.edge_index.shape[0]*raw_cond_data.edge_index.shape[1])}\n')

def icd2idx(icd_matrix,args,norm=False):
    
    icd_syn=torch.sigmoid(icd_matrix).detach()
    icd_syn_thresh=icd_syn
    icd_syn_thresh[icd_syn_thresh<args.threshold]=0
    
    if norm:
        H=icd_syn_thresh.numpy()
        DV = np.sum(H, axis=1)
        D_v_inv=DV**-1
        D_v_inv[D_v_inv == float("inf")] = 0
        D_v_inv_mat=torch.diag(torch.from_numpy(D_v_inv).float())
        norm_v_weight=torch.mm(D_v_inv_mat,icd_syn_thresh)
        edge_index_syn=torch.nonzero(norm_v_weight).T 
        icd_weight_syn=norm_v_weight[edge_index_syn[0], edge_index_syn[1]]
        
        return edge_index_syn,icd_weight_syn

    edge_index_syn=torch.nonzero(icd_syn_thresh).T 
    icd_weight_syn=icd_syn_thresh[edge_index_syn[0], edge_index_syn[1]]
    
    return edge_index_syn,icd_weight_syn

def load_cond_data(args,map_loc='cpu'):
    
    save_path = f'{args.save_path}/reduced_graph/{args.cond_method}'
    if args.self_loop:
        hyperedge_syn=torch.load(f'{save_path}/adj_{args.dname}_{args.reduction_rate}_{args.seed}_{args.self_loop}.pt',map_location=map_loc)
        feat_syn=torch.load(f'{save_path}/feat_{args.dname}_{args.reduction_rate}_{args.seed}_{args.self_loop}.pt',map_location=map_loc)
        labels_syn=torch.load(f'{save_path}/label_{args.dname}_{args.reduction_rate}_{args.seed}_{args.self_loop}.pt',map_location=map_loc)
    else:
        hyperedge_syn=torch.load(f'{save_path}/adj_{args.dname}_{args.reduction_rate}_{args.seed}.pt',map_location=map_loc)
        feat_syn=torch.load(f'{save_path}/feat_{args.dname}_{args.reduction_rate}_{args.seed}.pt',map_location=map_loc)
        labels_syn=torch.load(f'{save_path}/label_{args.dname}_{args.reduction_rate}_{args.seed}.pt',map_location=map_loc)

    return Data(x=feat_syn,edge_index=hyperedge_syn,y=labels_syn)

def raw2train(data,args):

    _, relabel_he_idx = torch.unique(data.edge_index[1], return_inverse=True)
    data.edge_index[1]=relabel_he_idx
    
    args.num_features = data.x.shape[1]
    args.num_classes = len(data.y.unique())
    if not hasattr(data, 'n_x'):
        data.n_x = torch.tensor([data.x.shape[0]])
    if not hasattr(data, 'num_hyperedges'):
        # note that we assume the he_id is consecutive.
        data.num_hyperedges = torch.tensor(
            [data.edge_index[0].max()-data.n_x+1])
        
    if args.method in ['HNHN']:
        data.totedges=torch.unique(data.edge_index[1]).numel()
        H = ConstructH_HNHN(data)
        data = generate_norm_HNHN(H, data, args)
        data.edge_index[1] -= data.edge_index[1].min()
    elif args.method in ['AllSetTransformer', 'AllDeepSets']:
        data.norm=torch.ones_like(data.edge_index[1])
    elif args.method in ['UniGCNII']:
        data = ConstructH(data)
        data.edge_index = sp.csr_matrix(data.edge_index)
    
    return data

def gen_V_E(edge_index,device):
    # scipy edge_index
    (row, col), value = torch_sparse.from_scipy(edge_index)
    V, E = row, col
    V, E = V.to(device), E.to(device)
    
    degV = torch.from_numpy(edge_index.sum(1)).view(-1, 1).float().to(device)
    degE = scatter(degV[V], E, dim=0, reduce='mean')
    degE = degE.pow(-0.5)
    degV = degV.pow(-0.5)
    degV[torch.isinf(degV)] = 1
        
    return V,E,degV,degE

# ============================== =============================#

def load_large_graph(data_path=f'../data/magsm/'):

    with open(os.path.join(data_path, f'features.pickle'), 'rb') as f:
        features = pkl.load(f)

    with open(os.path.join(data_path, f'hypergraph.pickle'), 'rb') as f:
        edge_index = pkl.load(f)

    with open(os.path.join(data_path, f'labels.pickle'), 'rb') as f:
        labels = pkl.load(f)
    
    return Data(x=features,edge_index=edge_index,y=labels)

def load_bipartite(args):
    
    data_path = os.path.join('..', 'data', args.dname)
    
    try:
        with open(os.path.join(data_path, f'H.pickle'), 'rb') as f:
            H = pkl.load(f)
            H = torch.stack([H[0],H[1]])
    except:
        with open(os.path.join(data_path, f'H.pkl'), 'rb') as f:
            center, _, hyperedges = pkl.load(f)

        V_idx = []
        E_idx = []
        for ie, edge in enumerate(hyperedges):
            V_idx.extend([int(v) for v in edge])
            E_idx.extend([ie] * len(edge))
        V_idx = torch.tensor(V_idx).long()
        E_idx = torch.tensor(E_idx).long()
        H = torch.stack([V_idx, E_idx])
    
    try:
        with open(os.path.join(data_path, f'L.pickle'), 'rb') as f:
            labels = pkl.load(f)
    except:
        labels = None
        
    try:
        with open(os.path.join(data_path, f'X.pickle'), 'rb') as f:
            features = pkl.load(f)
    except:
        features = gen_svd_feat(H,args)
        
    return Data(x=features,edge_index=H,y=labels)

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def gen_svd_feat(hyperedge_index,args):
    
    # 1. transform to scipy sparse matrix
    H=SparseTensor(row=hyperedge_index[0],col=hyperedge_index[1])
    H=H.fill_value(1.) 
    train=H.to_scipy() 
    train_csr = (train!=0).astype(np.float32)
    
    # 2. normalizing the adj matrix
    rowD = np.array(train.sum(1)).squeeze()
    colD = np.array(train.sum(0)).squeeze()
    for i in range(len(train.data)):
        train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)
        
    train = train.tocoo()
    
    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce()
    print('Performing SVD...')
    svd_u,s,svd_v = torch.svd_lowrank(adj_norm, q=args.k)
    u_mul_s = svd_u @ (torch.diag(s))
    v_mul_s = svd_v @ (torch.diag(s))
    del s
    print('SVD done.')
    
    return u_mul_s

def mem_computer(data):
    memory_x = data.x.element_size() * data.x.nelement()
    memory_hyperedge = data.edge_index.element_size() * data.edge_index.nelement()
    memory_y = data.y.element_size() * data.y.nelement()
    all_size=(memory_x+memory_hyperedge+memory_y)/1024/1024
    print(f'Data Storage Memory: {all_size:.4f}MB')

# =================== Inductive Training ==================#

def partion_subgraph(data,mask_id,device):

    subset_mask=torch.isin(data.edge_index[0],mask_id.to(device))
    sub_hypergraph=data.edge_index[:,subset_mask]

    relabel_map=dict(zip(mask_id.tolist(), [i for i in range(len(mask_id))]))
    relabel_src_idx=torch.tensor([relabel_map[v.item()] for v in sub_hypergraph[0]]).to(device)
    _, relabel_dst_idx = torch.unique(sub_hypergraph[1], return_inverse=True)

    subset_edge_index=torch.stack((relabel_src_idx,relabel_dst_idx),dim=0)

    return Data(x=data.x[mask_id],edge_index=subset_edge_index,y=data.y[mask_id]).to(device)

def Tran2Ind(data,split_idx,device):
    train_data=partion_subgraph(data,split_idx['train'],device)
    val_data=partion_subgraph(data,split_idx['valid'],device)
    test_data=partion_subgraph(data,split_idx['test'],device)
    return train_data,val_data,test_data

# ==========================  =========================#

# =================== Batch Training ==================#
def sparse_edge_tensor(hyperedge_index):
    
    H=SparseTensor(row=hyperedge_index[0],col=hyperedge_index[1]) # |V|x|E|
    H=H.fill_value(1.) 
    
    A=H.matmul(H.t()) 
    row,col,edge_attr=A.coo()
    edge_index=torch.stack([row,col])
    
    return edge_index,edge_attr

def convert_batch_hypergraph(data,batch,device):
    
    batch_mask=torch.isin(data.edge_index[0],batch.n_id)
    sub_hypergraph=data.edge_index[:,batch_mask]

    relabel_map=dict(zip(batch.n_id.tolist(), [i for i in range(len(batch.n_id))]))
    relabel_src_idx=torch.tensor([relabel_map[v.item()] for v in sub_hypergraph[0]]).to(device)
    _, relabel_dst_idx = torch.unique(sub_hypergraph[1], return_inverse=True)

    batch_edge_index=torch.stack((relabel_src_idx,relabel_dst_idx),dim=0)

    return Data(x=batch.x,edge_index=batch_edge_index)

# ==========================  =========================#

def parse_method(args, data):
    #     Currently we don't set hyperparameters w.r.t. different dataset
    if args.method == 'AllSetTransformer':
        if args.LearnMask:
            model = SetGNN(args, data.norm)
        else:
            model = SetGNN(args)
    
    elif args.method == 'AllDeepSets':
        args.PMA = False
        args.aggregate = 'add'
        if args.LearnMask:
            model = SetGNN(args,data.norm)
        else:
            model = SetGNN(args)

#     elif args.method == 'SetGPRGNN':
#         model = SetGPRGNN(args)

    elif args.method == 'CEGCN':
        model = CEGCN(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'CEGAT':
        model = CEGAT(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      heads=args.heads,
                      output_heads=args.output_heads,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'HyperGCN':
        #         ipdb.set_trace()
        He_dict = get_HyperGCN_He_dict(data)
        model = HyperGCN(V=data.x.shape[0],
                         E=He_dict,
                         X=data.x,
                         num_features=args.num_features,
                         num_layers=args.All_num_layers,
                         num_classses=args.num_classes,
                         args=args
                         )

    elif args.method == 'HGNN':
        # model = HGNN(in_ch=args.num_features,
        #              n_class=args.num_classes,
        #              n_hid=args.MLP_hidden,
        #              dropout=args.dropout)
        model = HCHA(args)

    elif args.method == 'HNHN':
        model = HNHN(args)

    elif args.method == 'HCHA':
        model = HCHA(args)

    elif args.method == 'MLP':
        model = MLP_model(args)
    elif args.method == 'UniGCNII':
            if args.cuda in [0,1]:
                device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')
            (row, col), value = torch_sparse.from_scipy(data.edge_index)
            V, E = row, col
            V, E = V.to(device), E.to(device)
            model = UniGCNII(args, nfeat=args.num_features, nhid=args.MLP_hidden, nclass=args.num_classes, nlayer=args.All_num_layers, nhead=args.heads,
                             V=V, E=E)
    #     Below we can add different model, such as HyperGCN and so on
    elif args.method == 'HyperND':
        model=HyperND(args)

    return model

class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            return best_result[:, 1], best_result[:, 3]

    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
#             print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])

@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        data.y[split_idx['test']], out[split_idx['test']])

#     Also keep track of losses
    train_loss = F.nll_loss(
        out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = F.nll_loss(
        out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss = F.nll_loss(
        out[split_idx['test']], data.y[split_idx['test']])
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out

@torch.no_grad()
def evaluate_ind(model,train_data,val_data,test_data,eval_func, result=None):

    model.eval()
    out_train = model(train_data)
    out_train = F.log_softmax(out_train, dim=1)
    
    out_val = model(val_data)
    out_val = F.log_softmax(out_val, dim=1)
    
    out_test = model(test_data)
    out_test = F.log_softmax(out_test, dim=1)

    train_acc = eval_func(train_data.y, out_train)
    valid_acc = eval_func(val_data.y, out_val)
    test_acc = eval_func(test_data.y, out_test)

#     Also keep track of losses
    train_loss = F.nll_loss(out_train,train_data.y)
    valid_loss = F.nll_loss(out_val,val_data.y)
    test_loss = F.nll_loss(out_test,test_data.y)
    
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

#     ipdb.set_trace()
#     for i in range(y_true.shape[1]):
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model,data,args,split_idx,device,ind_data=None):

    criterion = nn.NLLLoss()
    eval_func = eval_acc

    if args.inductive:
        ind_train,ind_val,ind_test=ind_data['train'],ind_data['valid'],ind_data['test']
    
    model.train()

    ### Training loop ###
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    if args.method == 'UniGCNII':
        optimizer = torch.optim.Adam([
            dict(params=model.reg_params, weight_decay=0.01),
            dict(params=model.non_reg_params, weight_decay=5e-4)
        ], lr=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = float('-inf')
    for epoch in range(args.epochs):
        # Training part
        model.train()
        optimizer.zero_grad()
        out=model(data)
        out = F.log_softmax(out, dim=1)
        if args.inductive:
            loss = criterion(out,data.y)
        else:
            loss = criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

        if args.inductive:
            result = evaluate_ind(model,ind_train,ind_val,ind_test,eval_func)
        else:
            result = evaluate(model,data,split_idx,eval_func)

        if (epoch+1) % args.display_step == 0 and args.display_step > 0:
            print(f'Epoch: {epoch+1:02d}, '
                    f'Train Loss: {loss:.4f}, '
                    f'Valid Loss: {result[4]:.4f}, '
                    f'Test  Loss: {result[5]:.4f}, '
                    f'Train Acc: {100 * result[0]:.2f}%, '
                    f'Valid Acc: {100 * result[1]:.2f}%, '
                    f'Test  Acc: {100 * result[2]:.2f}%')
