import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from contextlib import contextmanager

import warnings
warnings.filterwarnings('ignore')

# ============= Heat Kernel Propagation ==================#

def MHP(hyperedge_list,x,p=5):

    num_nodes=x.size(0)
    num_hyperedges=torch.unique(hyperedge_list[1]).numel()

    nodes = hyperedge_list[0]  
    hyperedges = hyperedge_list[1]  
    values = torch.ones(hyperedge_list.size(1)) 

    indices = torch.stack([nodes, hyperedges])
    H = torch.sparse_coo_tensor(indices, values, (num_nodes, num_hyperedges))
    H_dense=H.to_dense()

    DV = H_dense.sum(dim=1, keepdim=True)
    DV_inv=DV.pow(-0.5)
    DV_inv[DV_inv == float("inf")] = 0
    DV_inv_mat=torch.diag(DV_inv.squeeze())
    DV_inv_sp=DV_inv_mat.to_sparse()

    DE = H_dense.sum(dim=0, keepdim=True)
    DE_inv=DE.pow(-0.5)
    DE_inv[DE_inv == float("inf")] = 0
    DE_inv_mat=torch.diag(DE_inv.squeeze())

    norm_e=torch.sparse.mm(H,DE_inv_mat)
    H_norm=torch.sparse.mm(DV_inv_sp,norm_e)
    H_norm_sp = H_norm.to_sparse()
    M=torch.sparse.mm(H_norm_sp, H_norm.T)

    lambda_p=torch.FloatTensor([p])

    P=torch.exp(lambda_p*M)/torch.exp(lambda_p)
    X=torch.sparse.mm(P.to_sparse(),x)

    return X    

# ============= Gradient Matching ==================#

def reshape_gw(gwr, gws):
    shape = gwr.shape

    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])

    return gwr, gws

def distance_wb(args, gwr, gws):
    shape = gwr.shape
    gwr, gws = reshape_gw(gwr, gws)

    if len(shape) == 1:
        return 0
    
    if args.dis_metric == "ctrl":
        alpha = 1 - args.beta
        beta = args.beta

        cosine_similarity = F.cosine_similarity(gwr, gws, dim=-1)
        euclidean_distance = torch.norm(gwr - gws, dim=-1)

        distance = alpha * (1 - cosine_similarity) + beta * euclidean_distance
    else:
        distance = 1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001)
    return torch.sum(distance)


def match_loss(gw_syn, gw_real, args, device):
    dis = torch.tensor(0.0).to(device)

    if args.dis_metric == "ours" or args.dis_metric == "ctrl":

        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(args, gwr, gws)

    elif args.dis_metric == "mse":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    elif args.dis_metric == "cos":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
            torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001
        )

    else:
        exit("DC error: unknown distance function")

    return dis


class ReparamModule(nn.Module):
    def _get_module_from_name(self, mn):
        if mn == '':
            return self
        m = self
        for p in mn.split('.'):
            m = getattr(m, p)
        return m

    def __init__(self, module):
        super(ReparamModule, self).__init__()
        self.module = module

        param_infos = []  # (module name/path, param name)
        shared_param_memo = {}
        shared_param_infos = []  # (module name/path, param name, src module name/path, src param_name)
        params = []
        param_numels = []
        param_shapes = []
        for mn, m in self.named_modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None:
                    if p in shared_param_memo:
                        shared_mn, shared_n = shared_param_memo[p]
                        shared_param_infos.append((mn, n, shared_mn, shared_n))
                    else:
                        shared_param_memo[p] = (mn, n)
                        param_infos.append((mn, n))
                        params.append(p.detach())
                        param_numels.append(p.numel())
                        param_shapes.append(p.size())

        assert len(set(p.dtype for p in params)) <= 1, \
            "expects all parameters in module to have same dtype"

        # store the info for unflatten
        self._param_infos = tuple(param_infos)
        self._shared_param_infos = tuple(shared_param_infos)
        self._param_numels = tuple(param_numels)
        self._param_shapes = tuple(param_shapes)

        # flatten
        flat_param = nn.Parameter(torch.cat([p.reshape(-1) for p in params], 0))
        self.register_parameter('flat_param', flat_param)
        self.param_numel = flat_param.numel()
        del params
        del shared_param_memo

        # deregister the names as parameters
        for mn, n in self._param_infos:
            delattr(self._get_module_from_name(mn), n)
        for mn, n, _, _ in self._shared_param_infos:
            delattr(self._get_module_from_name(mn), n)

        # register the views as plain attributes
        self._unflatten_param(self.flat_param)

        # now buffers
        # they are not reparametrized. just store info as (module, name, buffer)
        buffer_infos = []
        for mn, m in self.named_modules():
            for n, b in m.named_buffers(recurse=False):
                if b is not None:
                    buffer_infos.append((mn, n, b))

        self._buffer_infos = tuple(buffer_infos)
        self._traced_self = None

    def trace(self, example_input, **trace_kwargs):
        assert self._traced_self is None, 'This ReparamModule is already traced'

        if isinstance(example_input, torch.Tensor):
            example_input = (example_input,)
        example_input = tuple(example_input)
        example_param = (self.flat_param.detach().clone(),)
        example_buffers = (tuple(b.detach().clone() for _, _, b in self._buffer_infos),)

        self._traced_self = torch.jit.trace_module(
            self,
            inputs=dict(
                _forward_with_param=example_param + example_input,
                _forward_with_param_and_buffers=example_param + example_buffers + example_input,
            ),
            **trace_kwargs,
        )

        self._forward_with_param = self._traced_self._forward_with_param
        self._forward_with_param_and_buffers = self._traced_self._forward_with_param_and_buffers
        return self

    def clear_views(self):
        for mn, n in self._param_infos:
            setattr(self._get_module_from_name(mn), n, None)

    def _apply(self, *args, **kwargs):
        if self._traced_self is not None:
            self._traced_self._apply(*args, **kwargs)
            return self
        return super(ReparamModule, self)._apply(*args, **kwargs)

    def _unflatten_param(self, flat_param):
        ps = (t.view(s) for (t, s) in zip(flat_param.split(self._param_numels), self._param_shapes))
        for (mn, n), p in zip(self._param_infos, ps):
            setattr(self._get_module_from_name(mn), n, p)  # This will set as plain attr
        for (mn, n, shared_mn, shared_n) in self._shared_param_infos:
            setattr(self._get_module_from_name(mn), n, getattr(self._get_module_from_name(shared_mn), shared_n))

    @contextmanager
    def unflattened_param(self, flat_param):
        saved_views = [getattr(self._get_module_from_name(mn), n) for mn, n in self._param_infos]
        self._unflatten_param(flat_param)
        yield
        for (mn, n), p in zip(self._param_infos, saved_views):
            setattr(self._get_module_from_name(mn), n, p)
        for (mn, n, shared_mn, shared_n) in self._shared_param_infos:
            setattr(self._get_module_from_name(mn), n, getattr(self._get_module_from_name(shared_mn), shared_n))

    @contextmanager
    def replaced_buffers(self, buffers):
        for (mn, n, _), new_b in zip(self._buffer_infos, buffers):
            setattr(self._get_module_from_name(mn), n, new_b)
        yield
        for mn, n, old_b in self._buffer_infos:
            setattr(self._get_module_from_name(mn), n, old_b)

    def _forward_with_param_and_buffers(self, flat_param, buffers, *inputs, **kwinputs):
        with self.unflattened_param(flat_param):
            with self.replaced_buffers(buffers):
                return self.module(*inputs, **kwinputs)

    def _forward_with_param(self, flat_param, *inputs, **kwinputs):
        with self.unflattened_param(flat_param):
            return self.module(*inputs, **kwinputs)

    def forward(self, *inputs, flat_param=None, buffers=None, **kwinputs):
        flat_param = torch.squeeze(flat_param)
        if flat_param is None:
            flat_param = self.flat_param
        if buffers is None:
            return self._forward_with_param(flat_param, *inputs, **kwinputs)
        else:
            return self._forward_with_param_and_buffers(flat_param, tuple(buffers), *inputs, **kwinputs)

# ============= Save Condensed Graph ==================#

'''
def save_reduced(adj_syn=None, feat_syn=None, labels_syn=None, args=None):
    save_path = f'{args.save_path}/reduced_graph/{args.cond_method}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if adj_syn is not None:
        torch.save(adj_syn,
                   f'{save_path}/adj_{args.dname}_{args.reduction_rate}_{args.seed}_{args.self_loop}.pt')
    if feat_syn is not None:
        torch.save(feat_syn,
                   f'{save_path}/feat_{args.dname}_{args.reduction_rate}_{args.seed}_{args.self_loop}.pt')
    if labels_syn is not None:
        torch.save(labels_syn,
                   f'{save_path}/label_{args.dname}_{args.reduction_rate}_{args.seed}_{args.self_loop}.pt')
    print(f"Saved {save_path}/adj_{args.dname}_{args.reduction_rate}_{args.seed}_{args.self_loop}.pt")
'''

def save_reduced(adj_syn=None, feat_syn=None, labels_syn=None, args=None):
    save_path = f'{args.save_path}/reduced_graph/{args.cond_method}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if adj_syn is not None:
        torch.save(adj_syn,
                   f'{save_path}/adj_{args.dname}_{args.reduction_rate}_{args.seed}.pt')
    if feat_syn is not None:
        torch.save(feat_syn,
                   f'{save_path}/feat_{args.dname}_{args.reduction_rate}_{args.seed}.pt')
    if labels_syn is not None:
        torch.save(labels_syn,
                   f'{save_path}/label_{args.dname}_{args.reduction_rate}_{args.seed}.pt')
    print(f"Saved {save_path}/adj_{args.dname}_{args.reduction_rate}_{args.seed}.pt")
