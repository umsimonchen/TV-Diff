import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
import os
import numpy as np
import random
import pickle
import scipy.sparse as sp
# paper: Revisiting Neighborhood-based Link Prediction for Collaborative Filtering. WWW'22

seed = 0
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

class LinkProp(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LinkProp, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['LinkProp'])
        self.alpha = float(args['-alpha'])
        self.beta = float(args['-beta'])
        self.gamma = float(args['-gamma'])
        self.delta = float(args['-delta'])
        self.r = int(args['-r'])
        self.t = float(args['-t'])
        self.model = LinkProp_Encoder(self.data, self.emb_size, self.alpha, self.beta, self.gamma, self.delta, self.r, self.t)

    def train(self):
        model = self.model.cuda()
        self.prediction = model.sparse_norm
        _, early_stopping = self.fast_evaluation(0) # Training-free
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = self.prediction[u]#torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class LinkProp_Encoder(nn.Module):
    def __init__(self, data, emb_size, alpha, beta, gamma, delta, r, t):
        super(LinkProp_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.r = r 
        self.t = t
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda() # (m+n)*(m+n)
        
        # item-item similarity matrix
        adj_mat = data.interaction_mat
        user_deg = np.array(adj_mat.sum(axis=1)) # m
        item_deg = np.array(adj_mat.sum(axis=0)) # n
        user_alpha = np.power(user_deg, -alpha).flatten()
        user_alpha[np.isinf(user_alpha)] = 0.
        user_alpha = sp.diags(user_alpha)
        
        item_beta = np.power(item_deg, -beta).flatten()
        item_beta[np.isinf(item_beta)] = 0.
        item_beta = sp.diags(item_beta)
        
        user_gamma = np.power(user_deg, -gamma).flatten()
        user_gamma[np.isinf(user_gamma)] = 0.
        user_gamma = sp.diags(user_gamma)
        
        item_delta = np.power(item_deg, -delta).flatten()
        item_delta[np.isinf(item_delta)] = 0.
        item_delta = sp.diags(item_delta)
        
        sparse_ab = (user_alpha.dot(adj_mat)).dot(item_beta)
        sparse_gd = (user_gamma.dot(adj_mat)).dot(item_delta)
        sparse_norm = sparse_ab.dot(adj_mat.T).dot(sparse_gd)
        self.sparse_norm = TorchGraphInterface.convert_sparse_mat_to_tensor(sparse_norm).cuda().to_dense() # m*n
        
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict
        
    def forward(self):
        all_user_embeddings = [self.embedding_dict['user_emb']]
        all_item_embeddings = [self.embedding_dict['item_emb']]
        for k in range(self.r):
            if self.t==0.:
                new_user_embedding = torch.sparse.mm(self.sparse_norm, all_item_embeddings[k])
                new_item_embedding = torch.sparse.mm(self.sparse_norm.transpose(0,1), all_user_embeddings[k])
                all_user_embeddings.append(new_user_embedding)
                all_item_embeddings.append(new_item_embedding)
                
        all_user_embeddings = torch.stack(all_user_embeddings, dim=1)
        all_user_embeddings = torch.mean(all_user_embeddings, dim=1)
        all_item_embeddings = torch.stack(all_item_embeddings, dim=1)
        all_item_embeddings = torch.mean(all_item_embeddings, dim=1)
        return all_user_embeddings, all_item_embeddings
    

        


