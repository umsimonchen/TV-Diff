import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_user
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
import os
import numpy as np
import random
import pickle
import scipy.sparse as sp
import math
from scipy.sparse.linalg import svds
# paper: Graph Spectral Filtering with Chebyshev Interpolation for Recommendation. SIGIR'25
# https://github.com/chanwoo0806/ChebyCF/blob/main/

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

class ChebyCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(ChebyCF, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['ChebyCF'])
        self.K= int(args['-K'])
        self.phi = float(args['-phi'])
        self.eta = int(args['-eta'])
        self.alpha = float(args['-alpha'])
        self.beta = float(args['-beta'])
        self.model = ChebyCF_Encoder(self.data, self.emb_size, self.K, self.phi, self.eta, self.alpha, self.beta)

    def train(self):
        model = self.model.cuda()
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        
        user_idx = list(range(self.data.user_num))
        total_batch = math.ceil(self.data.user_num/self.batch_size)
        while not early_stopping:
            random.shuffle(user_idx)
            for batch in range(total_batch):
                x_start = self.data.interaction_mat[user_idx[batch*self.batch_size:(batch+1)*self.batch_size]]
                model.fit(x_start)
                
            with torch.no_grad():
                self.prediction = []
                for batch in range(total_batch):
                    batch_x_start = self.data.interaction_mat[user_idx[batch*self.batch_size:(batch+1)*self.batch_size]]
                    batch_x_start = TorchGraphInterface.convert_sparse_mat_to_tensor(batch_x_start).to_dense()
                    batch_prediction = model.full_predict(batch_x_start)
                    self.prediction.append(batch_prediction.cpu().numpy())
                    print("Finished evaluating batch: %d / % d."%(batch+1,total_batch))
                self.prediction = np.concatenate(self.prediction, axis=0)
            measure, early_stopping = self.fast_evaluation(epoch)
            epoch += 1
        self.prediction = self.best_prediction
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")

    def save(self):
        with torch.no_grad():
            self.best_prediction = self.prediction

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = self.prediction[u]
        return score

def get_norm_inter(inter):
    user_degree = np.array(inter.sum(axis=1)).flatten() # Du
    item_degree = np.array(inter.sum(axis=0)).flatten() # Di
    user_d_inv_sqrt = np.power(user_degree.clip(min=1), -0.5)
    item_d_inv_sqrt = np.power(item_degree.clip(min=1), -0.5)
    user_d_inv_sqrt[user_degree == 0] = 0
    item_d_inv_sqrt[item_degree == 0] = 0
    user_d_inv_sqrt = sp.diags(user_d_inv_sqrt)  # Du^(-0.5)
    item_d_inv_sqrt = sp.diags(item_d_inv_sqrt)  # Di^(-0.5)
    norm_inter = (user_d_inv_sqrt @ inter @ item_d_inv_sqrt).tocoo() # Du^(-0.5) * R * Di^(-0.5)
    return norm_inter # R_tilde

def sparse_coo_tensor(mat):
    # scipy.sparse.coo_matrix -> torch.sparse.coo_tensor
    return torch.sparse_coo_tensor(
        indices=torch.tensor(np.vstack([mat.row, mat.col])),
        values=torch.tensor(mat.data, dtype=torch.float32),
        size=mat.shape)

class Laplacian(nn.Module):
    def __init__(self, inter):
        super().__init__()
        norm_inter = get_norm_inter(inter)
        norm_inter = sparse_coo_tensor(norm_inter)
        self.register_buffer('norm_inter', norm_inter) # shape (num_users, num_items)
        
    def __mul__(self, x):
        # L_tilde = 2L/lambda_max - I
        # = 2 (I - R_tilde^T * R_tilde)/1 - I
        # = R_tilde^T * R_tilde * -2 + I
        y = torch.spmm(self.norm_inter, x)
        y = torch.spmm(self.norm_inter.t(), y) * (-2)
        y += x
        return y

class ChebyFilter(nn.Module):
    def __init__(self, order, flatness):
        super().__init__()
        self.order = order
        self.flatness = flatness
        
    def plateau(self):
        x = torch.arange(self.order + 1)
        x = torch.cos((self.order - x) / self.order * math.pi).round(decimals=3)
        output = torch.zeros_like(x)
        output[x<0]  = (-x[x<0]).pow(self.flatness) *  0.5  + 0.5
        output[x>=0] = (x[x>=0]).pow(self.flatness) * (-0.5) + 0.5
        return output.round(decimals=3)

    def cheby(self, x, init):
        if self.order==0: return [init]
        output = [init, x * init]
        for _ in range(2, self.order+1):
            output.append(x * output[-1] * 2 - output[-2])
        return torch.stack(output)
    
    def fit(self, inter):
        # Laplacian_tilde
        self.laplacian = Laplacian(inter) # shape (num_items, num_items)
        
        # Chebyshev Nodes and Target Transfer Function Values
        cheby_nodes = torch.arange(1, (self.order+1)+1)
        cheby_nodes = torch.cos(((self.order+1) + 0.5 - cheby_nodes) / (self.order+1) * math.pi)
        target = self.plateau()
        # Chebyshev Interpolation Coefficients
        coeffs = self.cheby(x=cheby_nodes, init=target).sum(dim=1) * (2/(self.order+1))
        coeffs[0] /= 2
        self.register_buffer('coeffs', coeffs)
    
    def forward(self, signal):
        signal = signal.T
        bases = self.cheby(x=self.laplacian, init=signal)
        output = torch.einsum('K,KNB->BN', self.coeffs, bases)
        return output

class IdealFilter(nn.Module):
    def __init__(self, threshold, weight):
        super().__init__()
        self.threshold = threshold
        self.weight = weight
    
    def fit(self, inter):
        norm_inter = get_norm_inter(inter)
        _, _, vt = svds(norm_inter, which='LM', k=self.threshold)
        ideal_pass = torch.tensor(vt.T.copy())
        self.register_buffer('ideal_pass', ideal_pass) # shape (num_items, threshold)
        
    def forward(self, signal):
        ideal_preds = signal @ self.ideal_pass @ self.ideal_pass.T
        return ideal_preds * self.weight

class DegreeNorm(nn.Module):
    def __init__(self, power):
        super().__init__()
        self.power = power
    
    def fit(self, inter):
        item_degree = torch.tensor(np.array(inter.sum(axis=0)).flatten())
        zero_mask = (item_degree == 0)
        pre_norm = item_degree.clamp(min=1).pow(-self.power)
        pst_norm = item_degree.clamp(min=1).pow(+self.power)
        pre_norm[zero_mask], pst_norm[zero_mask] = 0, 0
        self.register_buffer('pre_normalize', pre_norm)  # (num_items,)
        self.register_buffer('post_normalize', pst_norm) # (num_items,)
        
    def forward_pre(self, signal):
        return signal * self.pre_normalize
    
    def forward_post(self, signal):
        return signal * self.post_normalize

class ChebyCF_Encoder(nn.Module):
    def __init__(self, data, emb_size, K, phi, eta, alpha, beta):
        super(ChebyCF_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.K = K
        self.phi = phi
        self.eta = eta
        self.alpha = alpha 
        self.beta = beta
        self.cheby = ChebyFilter(K, phi)
        self.ideal = IdealFilter(self.eta, self.alpha) if self.eta>0 and self.alpha > 0 else None
        self.norm = DegreeNorm(self.beta) if self.beta > 0 else None
    
    def fit(self, inter):
        self.cheby.fit(inter)
        if self.ideal:
            self.ideal.fit(inter)
        if self.norm:
            self.norm.fit(inter)
    
    def forward(self, signal):
        if self.norm:
            signal = self.norm.forward_pre(signal)
        output = self.cheby.forward(signal)
        if self.ideal:
            output += self.ideal.forward(signal)
        if self.norm:
            output = self.norm.forward_post(output)
        return output
    
    def mask_observed(self, pred_score, observed_inter):
        # Mask out the scores for items that have been already interacted with.
        return pred_score * (1 - observed_inter) - 1e8 * observed_inter
    
    def full_predict(self, observed_inter):
        pred_score = self.forward(observed_inter)
        return self.mask_observed(pred_score, observed_inter)
    
    
    