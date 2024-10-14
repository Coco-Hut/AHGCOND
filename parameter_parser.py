import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_prop', type=float, default=0.1)
parser.add_argument('--valid_prop', type=float, default=0.1)
parser.add_argument('--dname', default='cora',choices=['cora', 'citeseer', 'pubmed','coauthor_cora', 'coauthor_dblp','NTU2012', 'ModelNet40', 
                                                        'zoo', 'Mushroom', '20newsW100', 'yelp', 'house-committees-100', 'walmart-trips-100','magpm'])
parser.add_argument('--seed',default=2024)
# method in ['SetGNN','CEGCN','CEGAT','HyperGCN','HGNN','HCHA']
parser.add_argument('--method', default='HGNN',choices=['HGNN','HCHA'])
parser.add_argument('--inductive',default=False,type=bool) # tranductive/inductive
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch_training',default=False,type=bool)
# Number of runs for each split (test fix, only shuffle train/val)
parser.add_argument('--runs', default=1, type=int)
parser.add_argument('--cuda', default=1, choices=[-1, 0, 1], type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--wd', default=0.0, type=float)
# How many layers of full NLConvs
parser.add_argument('--All_num_layers', default=1, type=int)
parser.add_argument('--MLP_num_layers', default=2,
                    type=int)  # How many layers of encoder
parser.add_argument('--MLP_hidden', default=256,
                    type=int)  # Encoder hidden units
parser.add_argument('--Classifier_num_layers', default=2,
                    type=int)  # How many layers of decoder
parser.add_argument('--Classifier_hidden', default=128,
                    type=int)  # Decoder hidden units
parser.add_argument('--display_step', type=int, default=10)
parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
# ['all_one','deg_half_sym']
parser.add_argument('--normtype', default='all_one')
parser.add_argument('--add_self_loop', action='store_false')
# NormLayer for MLP. ['bn','ln','None']
#parser.add_argument('--normalization', default='ln')
parser.add_argument('--deepset_input_norm', default = True)
parser.add_argument('--GPR', action='store_false')  # skip all but last dec
# skip all but last dec
parser.add_argument('--LearnMask', action='store_false')
parser.add_argument('--num_features', default=0, type=int)  # Placeholder
parser.add_argument('--num_classes', default=0, type=int)  # Placeholder
# Choose std for synthetic feature noise
parser.add_argument('--feature_noise', default='0', type=str)
# whether the he contain self node or not
parser.add_argument('--exclude_self', action='store_true')
parser.add_argument('--PMA', action='store_true')
#     Args for HyperGCN
parser.add_argument('--HyperGCN_mediators', action='store_true')
parser.add_argument('--HyperGCN_fast', action='store_true')
#     Args for Attentions: GAT and SetGNN
parser.add_argument('--heads', default=4, type=int)  # Placeholder
parser.add_argument('--output_heads', default=1, type=int)  # Placeholder
#     Args for HNHN
parser.add_argument('--HNHN_alpha', default=-1.5, type=float)
parser.add_argument('--HNHN_beta', default=-0.5, type=float)
parser.add_argument('--HNHN_nonlinear_inbetween', default=True, type=bool)
#     Args for HCHA
parser.add_argument('--HCHA_symdegnorm', action='store_true')
#     Args for UniGNN
parser.add_argument('--UniGNN_use-norm', action="store_true", help='use norm in the final layer')
parser.add_argument('--UniGNN_degV', default = 0)
parser.add_argument('--UniGNN_degE', default = 0)
#     Args for HyperND
parser.add_argument('--restart_alpha', default=0.5, type=float)
parser.add_argument('--HyperND_ord', default = 1., type=float)
parser.add_argument('--HyperND_tol', default = 1e-4, type=float)
parser.add_argument('--HyperND_steps', default = 100, type=int)
parser.add_argument('--normalization',default='ln',type=str)
#     Args for Condensation
parser.add_argument('--reduction_rate',default=0.05,type=float) 
parser.add_argument('--hedge_reduction_rate',default=0.0,type=float) 
parser.add_argument('--cond_method',default='ahgcond',type=str,choices=['random','herding','kcenter','hgcondm','hgcondx','ahgcond']) # 压缩数据的方法
parser.add_argument('--cond_model',default='HGNN',type=str)

parser.add_argument('--batch_size',default=128,type=str) 
parser.add_argument('--lr_feat',default=0.035,type=float) 
parser.add_argument('--lr_pge',default=0.002,type=float)
parser.add_argument('--lr_threshs',default=0.002,type=float) 
parser.add_argument('--self_loop',default=False,type=bool) 
parser.add_argument('--outer_loop',default=20,type=int) 
parser.add_argument('--inner_loop',default=10,type=int) 
parser.add_argument('--cond_epochs',default=100,type=int) 
parser.add_argument('--iters',default=200,type=int) 
parser.add_argument('--tau_s',type=int,default=5,help='structure training epoch') 
parser.add_argument('--tau_f',type=int,default=15,help='feature training epoch') 
parser.add_argument('--threshold', type=float, default=0.9, help='adj threshold.') 
parser.add_argument('--dis_metric',type=str,default='ctrl')

parser.add_argument('--beta',type=float,default=0.15) 
parser.add_argument('--checkpoint',type=int,default=1)
parser.add_argument('--verbose',type=str,default=True)
parser.add_argument('--save_path',type=str,default='condensed_graph')

parser.add_argument('--learn_thresh',type=str,default=True)
parser.add_argument('--hard_self_loop',type=str,default=False) 
parser.add_argument('--extend_self_loop',type=str,default=False) 
parser.add_argument('--smooth_feat',type=str,default=False)
parser.add_argument('--smoothness_alpha',type=float,default=1)
parser.add_argument('--mi_bn',type=bool,default=True) 
parser.add_argument('--mi_beta',type=float,default=0.2) 
parser.add_argument('--prior',type=str,default='g_noise_f_add') 
parser.add_argument('--noise_std',type=float,default=0.01)

parser.add_argument('--rand_proj',type=bool,default=False)
parser.add_argument('--m_hop',type=bool,default=True) 
parser.add_argument('--f_add_noise',type=bool,default=True) 
parser.add_argument('--tau_n',type=float,default=2.0)
parser.add_argument('--p',type=int,default=5) 

parser.add_argument('--with_loop',type=bool,default=True)
parser.add_argument('--loop_weight',type=float,default=1) 

parser.set_defaults(PMA=True) 
parser.set_defaults(add_self_loop=True)
parser.set_defaults(exclude_self=False)
parser.set_defaults(GPR=False)
parser.set_defaults(LearnMask=False)
parser.set_defaults(HyperGCN_mediators=True)
parser.set_defaults(HyperGCN_fast=True)
parser.set_defaults(HCHA_symdegnorm=False) 

args = parser.parse_args(args=[])