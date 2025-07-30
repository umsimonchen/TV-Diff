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
import math
import pickle
import enum
import math
# paper: Recommendation via Collaborative Diffusion Generative Model. KSEM'22
# https://github.com/joojowalker/CODIGEM/blob/main

seed = 0
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

class CODIGEM(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(CODIGEM, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['CODIGEM'])
        T = int(args['-T'])
        beta = float(args['-beta'])
        
        D = self.data.item_num
        M = 200 # default number
        p_dnns = nn.ModuleList([nn.Sequential(nn.Linear(D, M), nn.PReLU(),
                                        nn.Linear(M, M), nn.PReLU(),
                                        nn.Linear(M, M), nn.PReLU(),
                                        nn.Linear(M, M), nn.PReLU(),
                                        nn.Linear(M, M), nn.PReLU(),
                                        nn.Linear(M, 2*D)) for _ in range(T-1)])
        
        decoder_net = nn.Sequential(nn.Linear(D, M), nn.PReLU(),
                            nn.Linear(M, M), nn.PReLU(),
                            nn.Linear(M, M), nn.PReLU(),
                            nn.Linear(M, M), nn.PReLU(),
                            nn.Linear(M, M), nn.PReLU(),
                            nn.Linear(M, D), nn.Tanh())
        
        self.model = CODIGEM_Encoder(p_dnns, decoder_net, beta, T, D)
        
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        total_anneal_steps = 200000 # default constant
        anneal_cap = 0.2
        while not early_stopping :
            total_batch = math.ceil(self.data.user_num/self.batch_size)
            update_count = 0
            for batch in range(total_batch):
                if total_anneal_steps > 0:
                    anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
                else: 
                    anneal = anneal_cap
                    
                data = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat[batch*self.batch_size:(batch+1)*self.batch_size]).cuda().float().to_dense()
                loss, recon = model(data, anneal)
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                update_count += 1
            print('training:', epoch + 1, 'loss:', loss.item())
            with torch.no_grad():
                self.prediction = []
                for batch in range(total_batch):
                    if total_anneal_steps > 0:
                        anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
                    else: 
                        anneal = anneal_cap
                        
                    data = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat[batch*self.batch_size:(batch+1)*self.batch_size]).cuda().float().to_dense()
                    loss, recon = model.forward(data, anneal)
                    self.prediction.append(recon.cpu().numpy())
                    print("Finished sampling batch: %d / % d."%(batch+1,total_batch))
                self.prediction = np.concatenate(self.prediction, axis=0)
                
            _, early_stopping = self.fast_evaluation(epoch)
            epoch += 1
        self.prediction = self.best_prediction
        # with open('reconstruct', 'wb') as fp:
        #     pickle.dump(self.prediction, fp)
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")

    def save(self):
        with torch.no_grad():
            self.best_prediction = self.prediction

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = self.prediction[u]
        return score

class CODIGEM_Encoder(nn.Module):
    def __init__(self, p_dnns, decoder_net, beta, T, D):
        super(CODIGEM_Encoder, self).__init__()
        self.p_dnns = p_dnns
        self.decoder_net = decoder_net
        self.D = D
        self.T = T
        self.device = torch.device('cuda')
        self.beta = torch.FloatTensor([beta]).to(self.device)
        
    @staticmethod
    def reparameterization(mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def reparameterization_gaussian_diffusion(self, x, i):
        return torch.sqrt(1. - self.beta) * x + torch.sqrt(self.beta) * torch.randn_like(x)

    def forward(self, x, anneal, reduction='avg'):
        # =====
        # forward difussion
        zs = [self.reparameterization_gaussian_diffusion(x, 0)]

        for i in range(1, self.T):
            zs.append(self.reparameterization_gaussian_diffusion(zs[-1], i))

        # =====
        # backward diffusion
        mus = []
        log_vars = []

        for i in range(len(self.p_dnns) - 1, -1, -1):

            h = self.p_dnns[i](zs[i+1])
            mu_i, log_var_i = torch.chunk(h, 2, dim=1)
            mus.append(mu_i)
            log_vars.append(log_var_i)

        mu_x = self.decoder_net(zs[0])

        # =====ELBO
        # RE

        # Normal RE
        PI = torch.from_numpy(np.asarray(np.pi))
        def log_standard_normal(x, reduction=None, dim=None):
            log_p = -0.5 * torch.log(2. * PI) - 0.5 * x**2.
            if reduction == 'avg':
                return torch.mean(log_p, dim)
            elif reduction == 'sum':
                return torch.sum(log_p, dim)
            else:
                return log_p
            
        def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
            log_p = -0.5 * torch.log(2. * PI) - 0.5 * log_var - \
                0.5 * torch.exp(-log_var) * (x - mu)**2.
            if reduction == 'avg':
                return torch.mean(log_p, dim)
            elif reduction == 'sum':
                return torch.sum(log_p, dim)
            else:
                return log_p
    
        RE = log_standard_normal(x - mu_x).sum(-1)

        # KL
        KL = (log_normal_diag(zs[-1], torch.sqrt(1. - self.beta) * zs[-1],
                              torch.log(self.beta)) - log_standard_normal(zs[-1])).sum(-1)

        for i in range(len(mus)):
            KL_i = (log_normal_diag(zs[i], torch.sqrt(1. - self.beta) * zs[i], torch.log(
                self.beta)) - log_normal_diag(zs[i], mus[i], log_vars[i])).sum(-1)

            KL = KL + KL_i

        # Final ELBO
        anneal = 1
        if reduction == 'sum':
            loss = -(RE - anneal * KL).sum()
        else:
            loss = -(RE - anneal * KL).mean()

        return loss, mu_x

    def init_weights(self):
        for layer in self.p_dnns:
            # Xavier Initialization for weights
            size = layer[0].weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer[0].weight.data.normal_(0.0, 0.0001)

            # Normal Initialization for Biases
            layer[0].bias.data.normal_(0.0, 0.0001)

        for layer in self.decoder_net:
            # Xavier Initialization for weights
            if str(layer) == "Linear":
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0/(fan_in + fan_out))
                layer.weight.data.normal_(0.0, 0.0001)

                # Normal Initialization for Biases
                layer.bias.data.normal_(0.0, 0.0001)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        