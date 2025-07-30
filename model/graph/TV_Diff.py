import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_user, next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss,InfoNCE
from data.augmentor import GraphAugmentor
import os
import numpy as np
import random
import math
import pickle
import enum
import scipy.sparse as sp
import time

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

class AnisotropicNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, config, data, hidden_size, emb_size, sparse_G=None, time_type="cat", norm=False, dropout=0.5):
        super(AnisotropicNN, self).__init__()
        self.data = data
        self.hidden_size = hidden_size
        self.time_emb_dim = emb_size
        self.sparse_G = sparse_G
        self.time_type = time_type
        self.norm = norm
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        self.drop = nn.Dropout(dropout)
        self.is_topology = True if sparse_G is not None else False
        self.weight_dict = self.init_weights()
    
    def init_weights(self):
        # Xavier Initialization for weights
        xavier_init = nn.init.xavier_uniform_
        weight_dict = {}
        last_layer = None
        for i, layer in enumerate(self.hidden_size):
            if i==0:
                weight_dict['item_weight%d'%i] = nn.Parameter(xavier_init(torch.empty(self.data.item_num, layer)))
                weight_dict['time_weight%d'%i] = nn.Parameter(xavier_init(torch.empty(self.time_emb_dim, layer)))
                if self.is_topology:
                    weight_dict['user_weight%d'%i] = nn.Parameter(xavier_init(torch.empty(self.data.user_num, layer)))
                else:
                    weight_dict['user_weight%d'%i] = nn.Parameter(xavier_init(torch.empty(self.data.item_num, layer)))
            else:
                weight_dict['item_weight%d'%i] = nn.Parameter(xavier_init(torch.empty(last_layer, layer)))
                weight_dict['user_weight%d'%i] = nn.Parameter(xavier_init(torch.empty(last_layer, layer)))
                weight_dict['time_weight%d'%i] = nn.Parameter(xavier_init(torch.empty(last_layer, layer)))
            last_layer = layer
        weight_dict = nn.ParameterDict(weight_dict)
        return weight_dict
    
    def forward(self, x_t, timesteps):
        time_emb = self.timestep_embedding(timesteps, self.time_emb_dim).to(x_t.device)
        time_emb = self.emb_layer(time_emb)
        if self.norm:
            x_t = torch.nn.functional.normalize(x_t)
        x_t = self.drop(x_t)

        for i, layer in enumerate(self.hidden_size):
            if i==0:
                if self.time_type == 'cat': # concatenation
                    u_h = torch.matmul(x_t, self.weight_dict['item_weight%d'%i]) + torch.matmul(time_emb, self.weight_dict['time_weight%d'%i])
                elif self.time_type == 'align': # dimension-alignment
                    u_h = torch.mul(torch.matmul(x_t, self.weight_dict['item_weight%d'%i]), torch.matmul(time_emb, self.weight_dict['time_weight%d'%i]))
                elif self.time_type == 'tg': # time-gating
                    u_h = torch.mul(torch.matmul(x_t, self.weight_dict['item_weight%d'%i]), nn.Sigmoid()(torch.matmul(time_emb, self.weight_dict['time_weight%d'%i])))
                else:
                    raise ValueError('Unimplemented timestep embedding type %s'%self.time_type)
            else:
                u_h = torch.matmul(u_h, self.weight_dict['item_weight%d'%i])
            u_h = torch.tanh(u_h)
                
        for i, layer in enumerate(self.hidden_size):
            if i==0:
                if self.is_topology:
                    i_h = torch.sparse.mm(torch.transpose(self.sparse_G, 0, 1), self.weight_dict['user_weight%d'%i])
                else:
                    i_h = self.weight_dict['user_weight%d'%i]
            else:
                i_h = torch.matmul(i_h, self.weight_dict['user_weight%d'%i]) 
            i_h = torch.tanh(i_h)

        recon_x_t = torch.matmul(u_h, i_h.transpose(0,1))
        return recon_x_t


    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
    
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
    
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

class TV_Diff(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(TV_Diff, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['TV_Diff'])
        self.steps = int(args['-steps']) # {2,5,10,40,50,100}
        self.noise_scale = float(args['-noise_scale']) # {0, 1e-5, 1e-4, 5e-3, 1e-2, 1e-1}
        self.noise_min = float(args['-noise_min']) # {5e-4, 1e-3, 5e-3}
        self.noise_max = float(args['-noise_max']) # {5e-3, 1e-2}
        self.temp = float(args['-temp'])
        self.gamma = float(args['-gamma'])
        self.mean_type = "x0" # default constant, but {"x0", "eps"}
        self.noise_schedule = "linear-var" # default constant, but {"linear", "linear-var", "cosine", "binomial"}
        self.sampling_step = 0 # default constant, but {0, T/4, T/2}
        self.device = torch.device('cuda')
        if self.mean_type == "x0":
            mean_type = ModelMeanType.START_X
        elif self.mean_type == "eps":
            mean_type = ModelMeanType.EPSILON
        else:
            raise ValueError("Unimplemented mean type %s"%mean_type)
        
        # constructe graph for alternating aggregation
        def construct_sparse_graph(adjacency_matrix, graph_type='sys'):
            row_sum = np.array(adjacency_matrix.sum(axis=1))
            col_sum = np.array(adjacency_matrix.sum(axis=0))
            
            if graph_type=='sys':
                row_d_inv = np.power(row_sum, -0.5).flatten()
                row_d_inv[np.isinf(row_d_inv)] = 0.
                row_degree_matrix = sp.diags(row_d_inv)
        
                col_d_inv = np.power(col_sum, -0.5).flatten()
                col_d_inv[np.isinf(col_d_inv)] = 0.
                col_degree_matrix = sp.diags(col_d_inv)
    
                norm_adj = row_degree_matrix.dot(adjacency_matrix).dot(col_degree_matrix).tocsr()            
            
            elif graph_type=='left':
                row_d_inv = np.power(row_sum, -1).flatten()
                row_d_inv[np.isinf(row_d_inv)] = 0.
                row_degree_matrix = sp.diags(row_d_inv)
        
                norm_adj = row_degree_matrix.dot(adjacency_matrix).tocsr()            
                 
            elif graph_type=='linkprop': # Revisiting Neighborhood-based Link Prediction for Collaborative Filtering
                user_inv = np.power(row_sum, -0.25).flatten()
                user_inv[np.isinf(user_inv)] = 0.
                user_mat = sp.diags(user_inv)

                item_inv = np.power(col_sum, -0.5).flatten()
                item_inv[np.isinf(item_inv)] = 0.
                item_mat = sp.diags(item_inv)
                
                norm_adj = (user_mat.dot(adjacency_matrix)).dot(item_mat) # m*n
            
            elif graph_type is None:
                return None
            
            else:
                raise ValueError("Unimplemented graph type %s"%graph_type)
            return TorchGraphInterface.convert_sparse_mat_to_tensor(norm_adj).cuda()

        sparse_G = construct_sparse_graph(self.data.interaction_mat, 'linkprop') # default constant, but {'sys', 'left', 'linkprop', None}
        
        # denosing in reverse process
        self.model = TV_Diff_Encoder(self.data, self.emb_size, mean_type, self.noise_schedule, self.noise_scale, self.noise_min, self.noise_max, self.steps, self.device, self.temp, self.gamma)
        encoder_layers = eval('[1000]') # default constant, but {[300], [200,600], [1000]}
        self.ann_model = AnisotropicNN(self.config, self.data, encoder_layers, self.emb_size, sparse_G, time_type="cat", norm=False)
        
        # negative distribution initialization
        coo = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat)
        n_rows, n_cols = coo.shape
        row_idx, col_idx = coo.coalesce().indices()
        self.row_counts = torch.bincount(row_idx, minlength=n_rows)
        prob = torch.ones((n_rows, n_cols), device=coo.device, dtype=torch.bool)
        prob[row_idx, col_idx] = False
        self.prob = prob.to(torch.float32)
        
    def train(self):
        model = self.model.cuda()
        ann_model = self.ann_model.cuda()
        optimizer_ann = torch.optim.Adam(ann_model.parameters(), lr=self.lRate)#, weight_decay=1e-6)
        early_stopping = False
        epoch = 0
        
        total_batch = math.ceil(self.data.user_num/self.batch_size)
        all_x_start = []
        for i in range(total_batch):
            all_x_start.append(TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat[i*self.batch_size:(i+1)*self.batch_size]).cuda().float().to_dense())
        
        entropy_type = 'bpr' # default constant, but {'ce', 'bpr', 'nll'}
        while not early_stopping :
            s = time.time()
            if entropy_type=='ce' or entropy_type=='nll': # batch-wise user list
                for n, batch in enumerate(next_batch_user(self.data, self.batch_size, self.prob, self.row_counts)):
                    user_idx, neg_x_start = batch
                    x_start = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat[user_idx]).cuda()
                    terms = model.training_losses(ann_model, entropy_type, False, input_zip=(x_start, neg_x_start), user_idx=user_idx)
                    loss = terms["loss"].mean()
                    
                    # backward propagation
                    optimizer_ann.zero_grad()
                    loss.backward()
                    optimizer_ann.step()
                    print("Finished training batch: %d / % d."%(n+1,total_batch))
            
            elif entropy_type=='bpr': # batch-wise training list
                bpr_batch = math.ceil(self.data.training_data_num/self.batch_size)
                for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                    user_idx, pos_idx, neg_idx = batch
                    x_start = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat[user_idx]).cuda()
                    terms = model.training_losses(ann_model, entropy_type, False, input_zip=(x_start, pos_idx, neg_idx), user_idx=user_idx)
                    loss = terms["loss"]
                    
                    # backward propagation
                    optimizer_ann.zero_grad()
                    loss.backward()
                    optimizer_ann.step()
                    if n%100 == 0 and n > 0:
                        print('Finished training batch: %d / % d.'%(n+1,bpr_batch))
            print('training:', epoch + 1, 'loss:', loss.mean().item(), 'time:',time.time()-s)
            with torch.no_grad():
                self.prediction = []
                for batch, batch_x_start in enumerate(all_x_start):
                    batch_prediction = model.p_sample(ann_model, batch_x_start, 0, False)
                    self.prediction.append(batch_prediction.cpu().numpy())
                    print("Finished evaluating batch: %d / % d."%(batch+1,total_batch))
                self.prediction = np.concatenate(self.prediction, axis=0)
                
            _, early_stopping = self.fast_evaluation(epoch)
            epoch += 1
        self.prediction = self.best_prediction
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")
        with open('reconstruct', 'wb') as fp:
            pickle.dump(self.prediction, fp)
    
    def save(self):
        with torch.no_grad():
            self.best_prediction = self.prediction

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = self.prediction[u]
        return score

class TV_Diff_Encoder(nn.Module):
    def __init__(self, data, emb_size, mean_type, noise_schedule, noise_scale, noise_min, noise_max,\
            steps, device, temp, gamma, history_num_per_term=10, beta_fixed=True):
        super(TV_Diff_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device
        self.temp = temp
        self.gamma = gamma
        self.lamda = 3 # default

        if noise_scale != 0.:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(self.device)
            if beta_fixed:
                self.betas[0] = 0.00001  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
                # The variance \beta_1 of the first step is fixed to a small constant to prevent overfitting.
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert len(self.betas) == self.steps, "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"

            self.calculate_for_diffusion()
    
        binary = self.data.interaction_mat.tocoo()
        binary_row = binary.row
        binary_col = binary.col
        binary_data = binary.data
        
        binary_row_t = torch.tensor(binary_row, dtype=torch.int32)
        binary_col_t = torch.tensor(binary_col, dtype=torch.int32)
        binary_data_t = torch.tensor(binary_data, dtype=torch.bool)
        self.observed_adj = torch.sparse_coo_tensor(torch.stack([binary_row_t, binary_col_t]), binary_data_t, (self.data.user_num, self.data.item_num), dtype=torch.bool)
        self.unobserved_adj = ~(self.observed_adj.to_dense()).cuda()    
    
    def get_betas(self):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                return betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        elif self.noise_schedule == "cosine":
            return betas_for_alpha_bar(
            self.steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        elif self.noise_schedule == "binomial":  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")
    
    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)  # alpha_{t-1}
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]).to(self.device)  # alpha_{t+1}
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ) # for x_start (x_0, predicted results)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        ) # for x_t (results at current timesteps)
    
    def p_sample(self, model, x_start, steps, sampling_noise=False):
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0: # full-length reverse
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.:
            for i in indices:
                t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = model(x_t, t)
            return x_t

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
            out = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t
    
    def training_losses(self, model, entropy_type, hard=False, **kwargs):
        if entropy_type=='nll':
            x_start, _ = kwargs['input_zip']
            x_start = x_start.coalesce() # In case of lower version Pytorch
        elif entropy_type=='ce': 
            x_start, neg_x_start = kwargs['input_zip']
            x_start = x_start.coalesce() # In case of lower version Pytorch
            neg_x_start = neg_x_start.coalesce()
        elif entropy_type=='bpr':
            x_start, pos_idx, neg_idx = kwargs['input_zip']
            x_start = x_start.coalesce() # In case of lower version Pytorch
        else:
            raise ValueError('Unimplemented entropy type %s'%entropy_type)
        
        batch_size, device = x_start.size(0), x_start.device
        dense_x_start = x_start.to_dense().cuda()
        ts, _ = self.sample_timesteps(batch_size, device, 'uniform') # expectation of timesteps ts
        
        noise = torch.randn_like(dense_x_start)
        if self.noise_scale != 0.:
            x_t = self.q_sample(dense_x_start, ts, noise)
        else:
            x_t = dense_x_start
        
        terms = {}
        model_output = model(x_t, ts)
        target = {
            ModelMeanType.START_X: dense_x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_t.shape
        
        if hard:
            user_idx=kwargs['user_idx']
            # _, indices = torch.sort(ui_score, dim=1, descending=True, stable=True) # for stable sampling - 1-1
            _, indices = torch.topk(model_output, int(self.gamma*self.data.item_num*1), dim=1) # for faster sampling - 1-2
            neg_idx = torch.zeros_like(model_output, dtype=torch.bool)
            # neg_idx.scatter_(1, indices[:,:int(self.neg_factor*self.data.item_num)], True) # for stable sampling - 2-1
            neg_idx.scatter_(1, indices, 1.0).to(torch.float) # for faster sampling - 2-2
            
            # neg_idx = neg_idx / neg_idx.sum(dim=1, keepdims=True)
            g = -torch.log(-torch.log(torch.rand_like(neg_idx)+1e-10)+1e-10)
            g_neg_idx = torch.exp(torch.log(neg_idx+1e-10)+g / torch.exp(-self.lamda *(1-ts/self.steps)).unsqueeze(1))
            g_neg_idx = torch.logical_and(g_neg_idx, self.unobserved_adj[user_idx])
            _, indices = torch.topk(g_neg_idx, int(self.gamma*self.data.item_num*1), dim=1)
            return 0
        
        mse = mean_flat((target - model_output) ** 2)
        if entropy_type=='nll':
            nll = torch.sparse.sum(nn.functional.logsigmoid(model_output).sparse_mask(x_start), dim=1)
            terms['loss'] = mse - self.temp * nll.to_dense()/self.data.item_num
        elif entropy_type=='ce':
            ce = torch.sparse.sum(nn.functional.logsigmoid(model_output).sparse_mask(x_start), dim=1) \
                  + torch.sparse.sum(nn.functional.logsigmoid(-model_output).sparse_mask(neg_x_start), dim=1)
            terms['loss'] = mse - self.temp * ce.to_dense()/self.data.item_num
        elif entropy_type=='bpr':
            pos_score = model_output[list(range(batch_size)), pos_idx]
            neg_score = model_output[list(range(batch_size)), neg_idx]
            bpr = torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
            terms['loss'] = torch.mean(mse) - self.temp * torch.mean(bpr)
        else:
            raise ValueError('Unimplemented entropy type %s'%entropy_type)
        
        return terms
    
    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance':  # importance sampling
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method='uniform')
            
            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1- uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1. < 1e-5

            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt
        
        elif method == 'uniform':  # uniform sampling
            t = torch.randint(0, self.steps, (batch_size,), device=device).long()
            pt = torch.ones_like(t).float()

            return t, pt
            
        else:
            raise ValueError
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model, x_t, t):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        B, C = x_t.shape[:2]
        assert t.shape == (B, )
        model_output = model(x_t, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x_t.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x_t.shape)
        
        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x_t, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)
        
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t) 

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
    
