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
# Testing model inspired by 
# paper: Cooperative Graph Neural Networks. ICML'24
# https: //github.com/benfinkelshtein/CoGNN

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

class CoRec(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(CoRec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['CoRec'])
        self.n_layers = int(args['-n_layer'])
        self.model = CoRec_Encoder(self.data, self.emb_size, self.n_layers)
        self.unobserved_adj = torch.logical_not(TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat).cuda().to_dense().to(torch.bool)).to(torch.float32)

    def train(self):
        record_list = []
        loss_list = []
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        while not early_stopping:
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                
                # # w/ NS
                # ui_score = torch.matmul(rec_user_emb[user_idx], rec_item_emb.transpose(0, 1))
                # _, indices = torch.sort(ui_score, dim=1, descending=True, stable=True)
                # neg_idx = torch.zeros_like(ui_score, dtype=torch.float32)
                # neg_idx.scatter_(1, indices[:,:int(0.06*self.data.item_num)], 1.0)
                # neg_idx = torch.mul(neg_idx, self.unobserved_adj[user_idx])
                # _, neg_idx = torch.sort(torch.mul(torch.randn_like(neg_idx), neg_idx), dim=1, descending=True, stable=True)
                # neg_idx = neg_idx[:,0]
                
                # negative sample generation w/o rejective
                ui_score = torch.matmul(rec_user_emb[user_idx], rec_item_emb.transpose(0, 1))
                ui_score = torch.clamp(ui_score, min=0.0)
                neg_idx = torch.multinomial(ui_score, 1, replacement=False).squeeze(1)
                
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            measure, early_stopping = self.fast_evaluation(epoch)
            record_list.append(measure)
            loss_list.append(batch_loss.item())
            epoch += 1
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")
        # record training loss        
        with open('training_record','wb') as fp:
            pickle.dump([record_list, loss_list], fp)

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class CoRec_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(CoRec_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        #all_embeddings = all_embeddings[-1] #max
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


