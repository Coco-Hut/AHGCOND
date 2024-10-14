import os
import time
import copy
import math
import torch
import random
from tqdm import trange
from itertools import product
import pickle
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
from utils_cond import MHP
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from parameterized_adj import PGE
from convert_datasets_to_pygDataset import dataset_Hypergraph
from utils_cond import match_loss,save_reduced,ReparamModule
import argparse

import warnings
warnings.filterwarnings('ignore')

class AHGCondBase:

    def __init__(self, data, split_idx, args, device,**kwargs):

        self.data = data
        self.args = args
        self.device = device

        self.split_idx=split_idx
        self.labels=data.y
        self.idx_train=split_idx['train'] 
        self.labels_train=data.y[self.idx_train] 
        self.nclass=data.y.max().cpu().numpy()+1
        self.test_res=[] 
        
        if self.args.inductive==True:
            self.convert2ind() 
         
        self.generate_labels_syn() 

        self.generate_coeff() 
        
        if self.args.inductive==True:
            self.generate_ind_idx()
        
        self.init_syn_data() 

        self.reset_parameters()
        
        self._gen_class_train_loaders() 
        
    def reset_parameters(self):
        if self.args.rand_proj:
            self.feat_syn.data.copy_(self.random_proj())
        elif self.args.m_hop:
            self.feat_syn.data.copy_(self.multi_hop_selection())
        else:
            print('Random Initialization')
            self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))
        self.threshs.data.copy_(torch.randn(self.threshs.size()))
        
    def convert2ind(self):
        self.ind_train,self.ind_val,self.ind_test=Tran2Ind(self.data,self.split_idx,self.device)
    
    def generate_ind_idx(self):

        self.ind_train_idx_class=[]
        
        for c in range(self.nclass):
            if c in self.num_class_dict:
                index=torch.where(self.ind_train.y==c)[0]
                self.ind_train_idx_class.append(index)
    
    def generate_labels_syn(self):

        counter = Counter(self.labels_train.cpu().numpy())
        num_class_dict = {}

        sorted_counter = sorted(counter.items(), key=lambda x:x[1])
        labels_syn = []
        syn_class_indices = {}

        for ix, (c, num) in enumerate(sorted_counter):
            num_class_dict[c] = math.ceil(num * self.args.reduction_rate)
            syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]

        self.labels_syn,_ = torch.sort(torch.LongTensor(labels_syn).to(self.device)) 
        self.num_class_dict=num_class_dict

    def generate_coeff(self):

        self.train_idx_class=[]
        self.ind_train_idx_class=[]
        self.coeff=[]
        self.coeff_sum=0
        train_indices=self.idx_train.to(self.device)

        for c in range(self.nclass):
            if c in self.num_class_dict:

                all_idx_c=torch.where(self.labels==c)[0]
                index=all_idx_c[torch.isin(all_idx_c,train_indices)]
                self.train_idx_class.append(index)
                
                coe = self.num_class_dict[c] / max(self.num_class_dict.values()) 
                self.coeff_sum+=coe
                self.coeff.append(coe)
            else:
                self.coeff.append(0)
        self.coeff_sum=torch.tensor(self.coeff_sum).to(self.device) 

    def get_hyperedge_num(self):

        indices = torch.nonzero(torch.isin(self.data.edge_index[0], self.idx_train.to(self.device))).squeeze(1)
        num=torch.unique(self.data.edge_index[1][indices]).numel()

        if self.args.hedge_reduction_rate:
            n_e=math.ceil(num * self.args.hedge_reduction_rate)
        else:
            n_e=math.ceil(num * self.args.reduction_rate)

        return n_e

    def random_proj(self):
        
        feat_syn=[]

        for c in range(self.nclass):
            lab_idx=torch.where(self.labels_train==c)[0]
            x_idx=self.idx_train[lab_idx]
            x_c=self.data.x[x_idx] 

            n_c=x_c.size(0)
            n_c_prime=self.num_class_dict[c]

            random_matrix=torch.randn(n_c,n_c_prime).to(self.device)
            projected_matrix = torch.matmul(random_matrix.T, x_c)
            
            feat_syn.append(projected_matrix)

        return torch.cat(feat_syn,dim=0)

    def multi_hop_selection(self):

        X=MHP(self.data.edge_index.to('cpu'),self.data.x.to('cpu'),self.args.p) 

        feat_syn=[]

        for c in range(self.nclass):
            lab_idx=torch.where(self.labels_train==c)[0] 
            x_idx=self.idx_train[lab_idx] 
            x_mhp_c=X[x_idx]
            
            k=self.num_class_dict[c]
            indices = torch.randperm(x_mhp_c.size(0))[:k]  
            selected_rows = x_mhp_c[indices] 

            feat_syn.append(selected_rows)

        feat_syn=torch.cat(feat_syn,dim=0)

        if self.args.f_add_noise:
            feat_syn+=torch.randn_like(feat_syn)/self.args.tau_n

        return feat_syn.to(self.device)

    def init_syn_data(self):

        nnodes_syn = len(self.labels_syn)
        self.n = nnodes_syn
        self.e = nnodes_syn
        self.d = self.data.x.shape[1]

        self.feat_syn = nn.Parameter(torch.FloatTensor(self.n, self.d).to(self.device))
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=self.args.lr_feat)

        self.pge=PGE(self.d,self.n,device=self.device,args=self.args).to(self.device)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=self.args.lr_pge)

        self.threshs=nn.Parameter(torch.FloatTensor(self.n).to(self.device))
        self.optimizer_threshs=torch.optim.Adam([self.threshs], lr=self.args.lr_threshs)
        
        if self.args.extend_self_loop:  
            self.self_loop=torch.eye(self.n).to(self.device)
            self.self_loop.requires_grad=False
        
        self.rand_pge=PGE(self.d,self.n,device=self.device,args=self.args).to(self.device)

        self.icd_weight_syn=None
        
        self.hist_grad=[None for i in range(self.nclass)]

        print(f'Shape of the condensed hypergraph: {self.n} x {self.e}')
            
    def get_loops(self, args):
        # Get the two hyper-parameters of outer-loop and inner-loop.
        # The following values are empirically good.
        """
        Retrieves the outer-loop and inner-loop hyperparameters.

        Parameters
        ----------
        args : Namespace
            Arguments object containing hyperparameters for training and model.

        Returns
        -------
        tuple
            Outer-loop and inner-loop hyperparameters.
        """
        return args.outer_loop, args.inner_loop
    
    def _gen_train_loader(self,data,idx_train):
        
        clique_edge_index,_=sparse_edge_tensor(data.edge_index)
        clique_graph=copy.deepcopy(data)
        clique_graph.edge_index=clique_edge_index
        train_mask = torch.from_numpy(np.isin(np.arange(data.x.shape[0]), idx_train.cpu().numpy()))
       
        train_loader = NeighborLoader(
            clique_graph.contiguous(),  
            input_nodes=train_mask,
            num_neighbors=[5, 5], 
            batch_size=self.args.batch_size,  
            shuffle=True,  
        )
        
        return train_loader
    
    def _gen_class_train_loaders(self):
        self.data_loaders=[]
        if self.args.inductive==False:
            for c in range(self.nclass):
                self.data_loaders.append(self._gen_train_loader(self.data,self.train_idx_class[c]))
        else:
            for c in range(self.nclass):
                self.data_loaders.append(self._gen_train_loader(self.ind_train,self.ind_train_idx_class[c]))
    
    def train_class(self,model,t):
        """
        Trains the model and computes the loss.

        Parameters
        ----------
        model : torch.nn.Module
            The model object.
        Returns
        -------
        torch.Tensor
            The computed loss value.
        """
        if self.args.inductive==False:
            data=self.data
        else:
            data=self.ind_train
            
        feat_syn_outer = self.feat_syn
        icd_syn_outer = self.icd_index_syn
        icd_weight_syn_outer=self.icd_weight_syn

        loss = torch.tensor(0.0).to(self.device)
        for c in range(self.nclass):
            
            for batch in self.data_loaders[c]:

                batch = batch.to(self.device)
                batch_data=convert_batch_hypergraph(data,batch,self.device)
                out = model(batch_data)
                loss_real = F.nll_loss(out[:batch.batch_size], batch.y[:batch.batch_size])
                break 
            
            gw_reals = torch.autograd.grad(loss_real, model.parameters())
            gw_reals = list((_.detach().clone() for _ in gw_reals))
            
            if t == 0 :
                self.hist_grad[c]=gw_reals
            else:
                for ig in range(len(self.hist_grad[c])):
                    gwr = self.hist_grad[c][ig]
                    gwc = gw_reals[ig]
                    gw_reals[ig]=(1-1/t)*gwr+(1/t)*gwc
                self.hist_grad[c]=gw_reals

            # ------------------------------------------------------------------
            cdata=Data(x=feat_syn_outer,edge_index=icd_syn_outer,icd_weight=icd_weight_syn_outer)
            output_syn = model(cdata)
            loss_syn = F.nll_loss(output_syn[self.labels_syn == c], self.labels_syn[self.labels_syn == c])
            gw_syns = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)
            # ------------------------------------------------------------------
            coeff = self.num_class_dict[c] / self.n
            ml = match_loss(gw_syns, gw_reals, self.args, device=self.device)
            loss += coeff * ml

        return loss
    
    def test_with_val(self,cond_data,iters=200):
    
        criterion = nn.NLLLoss()
        eval_func = eval_acc

        if self.args.cond_model=='HGNN':
            model=HCHA(self.args).to(self.device)
        
        model.train()
        
        ### Training loop ###
        start_time = time.time()

        model.reset_parameters()
        if self.args.method == 'UniGCNII':
            optimizer = torch.optim.Adam([
                dict(params=model.reg_params, weight_decay=0.01),
                dict(params=model.non_reg_params, weight_decay=5e-4)
            ], lr=0.01)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)

        best_val = float('-inf')
        
        for epoch in range(iters):
            #         Training part
            model.train()
            optimizer.zero_grad()
            out = model(cond_data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out,cond_data.y)
            loss.backward()
            optimizer.step()

            if self.args.inductive==False:
                result = evaluate(model, self.data, self.split_idx, eval_func)
            else:
                result = evaluate_ind(model,self.ind_train,self.ind_val,self.ind_test,eval_func)
            
            if (epoch+1) % self.args.display_step == 0:
                print(f'Epoch: {epoch+1:02d}, '
                        f'Train Loss: {loss:.4f}, '
                        f'Valid Loss: {result[4]:.4f}, '
                        f'Test  Loss: {result[5]:.4f}, '
                        f'Train Acc: {100 * result[0]:.2f}%, '
                        f'Valid Acc: {100 * result[1]:.2f}%, '
                        f'Test  Acc: {100 * result[2]:.2f}%')
                
            if result[1]>best_val:
                best_val=result[1]
            
            if (epoch+1)==iters:
                self.test_res.append(result[2])
                
        end_time = time.time()
        print(f'Evaluation Time: {end_time-start_time:.2f}')
        
        return best_val
    
    def intermediate_evaluation(self, best_val, loss_avg, save=True):
        """
        Performs intermediate evaluation and saves the best model.

        Parameters
        ----------
        best_val : float
            The best validation accuracy observed so far.
        loss_avg : float
            The average loss.
        save : bool, optional
            Whether to save the model (default is True).

        Returns
        -------
        float
            The updated best validation accuracy.
        """
        args = self.args
        if args.verbose:
            print('loss_avg: {}'.format(loss_avg))
        
        feat_syn=self.feat_syn.detach()

        icd_syn=self.pge.inference(self.feat_syn).detach().to(self.device)
        icd_syn_thresh=icd_syn.clone()
        if self.args.learn_thresh:
            threshs=self.threshs.detach() 
            train_threshold=torch.sigmoid(threshs)
            threshold_matrix = train_threshold.unsqueeze(0).expand_as(icd_syn_thresh) 
            saved_icd_syn = torch.where(icd_syn_thresh > threshold_matrix, icd_syn_thresh, torch.tensor(0.0,device=self.device))
            icd_syn_thresh = torch.relu(icd_syn_thresh - threshold_matrix)
        else:
            icd_syn_thresh[icd_syn_thresh<self.args.threshold]=0  
            saved_icd_syn=copy.deepcopy(icd_syn_thresh)

        if self.args.extend_self_loop:
            icd_syn_thresh=torch.cat([self.self_loop,icd_syn_thresh],dim=1)      
            icd_syn=torch.cat([self.self_loop,icd_syn],dim=1)  
        
        edge_index_syn=torch.nonzero(icd_syn_thresh).T 
        icd_weight_syn=icd_syn[edge_index_syn[0], edge_index_syn[1]]
        
        cond_data=Data(x=feat_syn,edge_index=edge_index_syn,icd_weight=icd_weight_syn,y=self.labels_syn)
        
        current_val=self.test_with_val(cond_data,iters=self.args.iters)

        if save and current_val > best_val:
            best_val = current_val
            self.cond_data=cond_data 
            save_reduced(saved_icd_syn,cond_data.x,cond_data.y,args) 
                
        return best_val 

class AHGCOND(AHGCondBase):

    def __init__(self, data, split_idx, args, device, **kwargs):
        super(AHGCOND, self).__init__(data, split_idx, args, device, **kwargs)

    def get_prior(self,noise_std=0.01):
        
        def add_noise_to_model(model, noise_std=0.01):
            for param in model.parameters():
                noise = torch.randn_like(param) * noise_std  
                param.data += noise 

        if self.args.prior=='s_free':
            prior_dist=torch.eye(self.n).to(self.device)
        elif self.args.prior=='g_noise':
            gauss_icd=torch.randn(self.n,self.n).to(self.device)
            prior_dist=gauss_icd-torch.diag(torch.diag(gauss_icd, 0))+torch.eye(self.n).to(self.device)
        elif self.args.prior=='g_noise_f':
            noise=torch.randn(self.n,self.d).to(self.device)
            prior_dist=self.pge(noise).to(self.device)
        elif self.args.prior=='g_noise_f_add':
            noise=torch.randn(self.n,self.d).to(self.device)
            prior_dist=self.pge(noise+self.feat_syn).to(self.device)
        elif self.args.prior=='r_pge':
            prior_dist=self.rand_pge(self.feat_syn).to(self.device)
        elif self.args.prior=='r_pge_add':
            rand_pge=copy.deepcopy(self.pge)
            add_noise_to_model(rand_pge, noise_std=noise_std)
            prior_dist=rand_pge(self.feat_syn).to(self.device)
        else:
            raise ValueError('Not Implemented Error!')

        return prior_dist

    def reduce(self):

        if self.args.cond_model=='HGNN':
            model=HCHA(self.args).to(self.device)

        outer_loop,inner_loop=self.get_loops(self.args)
        loss_avg = 0
        best_val = 0

        for it in trange(self.args.cond_epochs):
            
            model.reset_parameters()
            model_parameters=list(model.parameters())
            optimizer_model=torch.optim.Adam(model_parameters,lr=self.args.lr)
            model.train()
            
            for ol in range(outer_loop):

                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                if self.args.learn_thresh:
                    self.optimizer_threshs.zero_grad()

                icd_syn=self.pge(self.feat_syn).to(self.device)
                icd_syn_thresh=icd_syn.clone()
                if self.args.learn_thresh:
                    train_threshold=torch.sigmoid(self.threshs)   
                    threshold_matrix = train_threshold.unsqueeze(0).expand_as(icd_syn_thresh)  # [n, n]
            
                    icd_syn_thresh = torch.relu(icd_syn_thresh - threshold_matrix)
                else:
                    icd_syn_thresh[icd_syn_thresh<self.args.threshold]=0          
                
                if self.args.extend_self_loop:
                    icd_syn_thresh=torch.cat([self.self_loop,icd_syn_thresh],dim=1)  
                    icd_syn=torch.cat([self.self_loop,icd_syn],dim=1)  
                
                if self.args.smooth_feat:
                    
                    clique=icd_syn_thresh@icd_syn_thresh.T
                    clq_edge_index_syn=torch.nonzero(clique).T
                    clq_edge_weight_syn=clique[clq_edge_index_syn[0], clq_edge_index_syn[1]]
                    feat_difference = torch.exp(-0.5 * torch.pow(self.feat_syn[clq_edge_index_syn[0]] - self.feat_syn[clq_edge_index_syn[1]], 2))
                    smoothness_loss = torch.dot(clq_edge_weight_syn,torch.mean(feat_difference,1).flatten())/torch.sum(clq_edge_weight_syn)
                else:
                    smoothness_loss=torch.tensor(0.0).to(self.device)

                if self.args.mi_bn:
                    prior=self.get_prior(noise_std=self.args.noise_std)
                    mi_loss=torch.norm(icd_syn-prior,p=2)
                else:
                    mi_loss=torch.tensor(0.0).to(self.device)

                self.icd_index_syn=torch.nonzero(icd_syn_thresh).T 

                self.icd_weight_syn=icd_syn[self.icd_index_syn[0],self.icd_index_syn[1]] 

                loss_gm=self.train_class(model) 
                loss=loss_gm+self.args.mi_beta*mi_loss+self.args.smoothness_alpha*smoothness_loss

                loss_avg += loss.item()

                loss.backward()
 
                if ol%(self.args.tau_s+self.args.tau_f)<self.args.tau_s:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()        

                if self.args.learn_thresh:
                    self.optimizer_threshs.step()
                
                feat_syn_inner=self.feat_syn.detach()
                
                icd_syn=self.pge.inference(self.feat_syn).detach().to(self.device)
                icd_syn_thresh=icd_syn.clone()
                if self.args.learn_thresh:
                    threshs=self.threshs.detach() 
                    train_threshold=torch.sigmoid(threshs)   
                    threshold_matrix = train_threshold.unsqueeze(0).expand_as(icd_syn_thresh) 
                    
                    icd_syn_thresh = torch.relu(icd_syn_thresh - threshold_matrix)
                else:
                    icd_syn_thresh[icd_syn_thresh<self.args.threshold]=0    
                
                if self.args.extend_self_loop:
                    icd_syn_thresh=torch.cat([self.self_loop,icd_syn_thresh],dim=1) 
                    icd_syn=torch.cat([self.self_loop,icd_syn],dim=1)       
                
                icd_syn_inner=torch.nonzero(icd_syn_thresh).T 
                icd_weight_syn_inner=icd_syn[icd_syn_inner[0], icd_syn_inner[1]]
                
                cdata=Data(x=feat_syn_inner,edge_index=icd_syn_inner,icd_weight=icd_weight_syn_inner)
                
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner=model(cdata)
                    loss_syn_inner = F.nll_loss(output_syn_inner,self.labels_syn)
                    loss_syn_inner.backward()
                    optimizer_model.step()
                
                loss_avg /= (self.nclass * outer_loop)
            
            if it%self.args.checkpoint==0:
                best_val=self.intermediate_evaluation(best_val,loss_avg)
                print(f'Best Valid Acc: {100 * best_val:.2f}%')
        
        