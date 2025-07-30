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
import math
# paper: CDAE: Collaborative denoising auto-encoders for top-N recommender systems. WSDM'16
# https://github.com/jasonyaw/CDAE

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

class CDAE(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(CDAE, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['CDAE'])
        self.corruption_ratio = float(args['-corruption_ratio'])
        self.model = CDAE_Encoder(self.data, self.emb_size, self.corruption_ratio)
        self.norm_inter = self.data.norm_inter
    
    def next_batch_user(self, data, batch_size):
        training_data = list(range(data.user_num))
        random.shuffle(training_data)
        ptr = 0
        data_size = data.user_num
        while ptr < data_size:
            if ptr + batch_size < data_size:
                batch_end = ptr + batch_size
            else:
                batch_end = data_size
            u_idx = training_data[ptr:batch_end]
                
            ptr = batch_end
            yield u_idx
    
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        user_list = list(range(self.data.user_num))
        self.loss_type = 'BCE'

        total_batch = math.ceil(self.data.user_num/self.batch_size)
        
        while not early_stopping:
            random.shuffle(user_list)
            for n, batch in enumerate(self.next_batch_user(self.data, self.batch_size)):
                user_idx = batch
                x_items = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat[user_idx]).cuda().to_dense()
                predict = model(x_items, user_idx)
                if self.loss_type == "MSE":
                    predict = model.o_act(predict)
                    loss_func = nn.MSELoss(reduction="mean")
                elif self.loss_type == "BCE":
                    loss_func = nn.BCEWithLogitsLoss(reduction="mean")
                else:
                    raise ValueError("Invalid loss_type, loss_type must in [MSE, BCE]")
                loss = loss_func(predict, x_items.to_dense())

                #l2-regularization
                loss += self.reg * (model.h_user.weight.norm() + model.h_item.weight.norm())
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('training:', epoch + 1, 'loss:', loss.item())
            with torch.no_grad():
                self.prediction = []
                for i in range(total_batch):
                    user_idx = user_list[i*self.batch_size:(i+1)*self.batch_size]
                    x_items = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat[user_idx]).cuda().to_dense()
                    batch_prediction = model.o_act(model.forward(x_items, user_idx))
                    self.prediction.append(batch_prediction.cpu().numpy())
                    print("Finished evaluating batch: %d / % d."%(i+1,total_batch))
                self.prediction = np.concatenate(self.prediction, axis=0)
            measure, early_stopping = self.fast_evaluation(epoch)
            epoch += 1
        self.prediction = self.best_prediction
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")
        # record training loss        
        # with open('training_record','wb') as fp:
        #     pickle.dump([record_list, loss_list], fp)
        
    def save(self):
        with torch.no_grad():
            self.best_prediction = self.prediction

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = self.prediction[u]
        return score

class CDAE_Encoder(nn.Module):
    def __init__(self, data, emb_size, corruption_ratio):
        super(CDAE_Encoder, self).__init__()
        self.data = data
        self.embedding_size = emb_size
        self.corruption_ratio = corruption_ratio
        self.embedding_dict = self._init_model()
        self.hid_activation = 'relu' # default constant, but {'sigmoid','relu','tanh'}
        self.out_activation = 'sigmoid' # default constant, but {'sigmoid', 'relu'}
        
        if self.hid_activation == "sigmoid":
            self.h_act = nn.Sigmoid()
        elif self.hid_activation == "relu":
            self.h_act = nn.ReLU()
        elif self.hid_activation == "tanh":
            self.h_act = nn.Tanh()
        else:
            raise ValueError("Invalid hidden layer activation function")

        if self.out_activation == "sigmoid":
            self.o_act = nn.Sigmoid()
        elif self.out_activation == "relu":
            self.o_act = nn.ReLU()
        else:
            raise ValueError("Invalid output layer activation function")
            
        self.dropout = nn.Dropout(p=self.corruption_ratio)
    
    def xavier_normal_initialization(self, module):
        r"""using `xavier_normal_`_ in PyTorch to initialize the parameters in
        nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
        using constant 0 to initialize.
    
        .. _`xavier_normal_`:
            https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
    
        Examples:
            >>> self.apply(xavier_normal_initialization)
        """
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
            
    def _init_model(self):
        self.h_user = nn.Embedding(self.data.user_num, self.embedding_size)
        self.h_item = nn.Linear(self.data.item_num, self.embedding_size)
        self.out_layer = nn.Linear(self.embedding_size, self.data.item_num)
        
        # parameters initialization
        self.apply(self.xavier_normal_initialization)
    
    def forward(self, x_items, x_users):
        h_i = self.dropout(x_items)
        h_i = self.h_item(h_i)
        h_u = self.h_user(torch.tensor(x_users).cuda())
        h = torch.add(h_u, h_i)
        h = self.h_act(h)
        out = self.out_layer(h)
        return out
        


