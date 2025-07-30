import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise, next_batch_user
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
from data.augmentor import GraphAugmentor
import os
import numpy as np
import random
import math
import pickle
import enum
import math

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

class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps):
        time_emb = self.timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = torch.nn.functional.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        
        return h


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

class GumbelDiff(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(GumbelDiff, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['GumbelDiff'])
        self.steps = int(args['-steps']) # [2,5,10,40,50,100]
        self.noise_scale = float(args['-noise_scale']) # [0, 1e-5, 1e-4, 5e-3, 1e-2, 1e-1]
        self.noise_min = float(args['-noise_min']) # [5e-4, 1e-3, 5e-3]
        self.noise_max = float(args['-noise_max']) # [5e-3, 1e-2]
        self.drop_rate = float(args['-droprate'])
        self.mean_type = "x0" # default constant, but ["x0", "eps"]
        self.noise_schedule = "linear-var" # default constant, but ["linear", "linear-var", "cosine", "binomial"]
        self.sampling_step = 0 # default constant, but [0, T/4, T/2]
        self.device = torch.device('cuda')
        if self.mean_type == "x0":
            mean_type = ModelMeanType.START_X
        elif self.mean_type == "eps":
            mean_type = ModelMeanType.EPSILON
        else:
            raise ValueError("Unimplemented mean type %s"%mean_type)
            
        self.model = GumbelDiff_Encoder(mean_type, self.noise_schedule, self.noise_scale, self.noise_min, self.noise_max, self.steps, self.device)
        
        out_dims = eval('[1000]') + [self.data.item_num] # default constant, but [[300], [200,600], [1000]]
        in_dims = out_dims[::-1]
        self.dnn_model = DNN(in_dims, out_dims, self.emb_size, time_type="cat", norm=False)
        
        self.dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        coo = TorchGraphInterface.convert_sparse_mat_to_tensor(self.dropped_mat)
        n_rows, n_cols = coo.shape
        row_idx, col_idx = coo.coalesce().indices()
        self.row_counts = torch.bincount(row_idx, minlength=n_rows)
        prob = torch.ones((n_rows, n_cols), device=coo.device, dtype=torch.bool)
        prob[row_idx, col_idx] = False
        self.prob = prob.to(torch.float32)
        
    def train(self):
        model = self.model.cuda()
        dnn_model = self.dnn_model.cuda()
        optimizer_dnn = torch.optim.Adam(dnn_model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        total_batch = math.ceil(self.data.user_num/self.batch_size)
        while not early_stopping :
            for n, batch in enumerate(next_batch_user(self.data, self.batch_size, self.prob, self.row_counts)):
                user_idx, neg_x_start = batch
                x_start = TorchGraphInterface.convert_sparse_mat_to_tensor(self.dropped_mat[user_idx]/self.dropped_mat[user_idx].sum(axis=1)).cuda()
                terms = model.training_losses(dnn_model, x_start, neg_x_start, False) # dense tensor
                loss = terms["loss"].mean()
                
            # batch-wise entropy (ce/bpr)
            # for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
            #     user_idx, pos_idx, neg_idx = batch
            #     x_start = TorchGraphInterface.convert_sparse_mat_to_tensor(self.dropped_mat[user_idx]).cuda().float().to_dense()
            #     terms = model.training_losses(dnn_model, x_start, pos_idx, neg_idx, False)
            #     #loss = terms["loss"].mean() + l2_reg_loss(self.reg, torch.mul(x_start, torch.log(1e-6 + torch.sigmoid(terms["model_output"]))))/len(user_idx)
            #     loss = terms
                
                # backward propagation
                optimizer_dnn.zero_grad()
                loss.backward()
                optimizer_dnn.step()
                print("Finished training batch: %d / % d."%(n+1,total_batch))
            print('training:', epoch + 1, 'loss:', loss.mean().item())
            with torch.no_grad():
                all_x_start = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat).cuda().float().to_dense()
                self.prediction = model.p_sample(dnn_model, x_start)
            _, early_stopping = self.fast_evaluation(epoch)
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
        return score.cpu().numpy()

class GumbelDiff_Encoder(nn.Module):
    def __init__(self, mean_type, noise_schedule, noise_scale, noise_min, noise_max,\
            steps, device, history_num_per_term=10, beta_fixed=True):
        super(GumbelDiff_Encoder, self).__init__()
        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device

        self.history_num_per_term = history_num_per_term
        self.Lt_history = torch.zeros(steps, history_num_per_term, dtype=torch.float64).to(device)
        self.Lt_count = torch.zeros(steps, dtype=int).to(device)

        if noise_scale != 0.:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(self.device)
            if beta_fixed:
                self.betas[0] = 0.00001  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
                # The variance \beta_1 of the first step is fixed to a small constant to prevent overfitting.
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert len(self.betas) == self.steps, "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"

            self.calculate_for_diffusion()
    
        self.tau_max = 10.0
        self.lamda = 3.0
        self.beta = 2.0
    
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
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
    # def p_sample(self, model, x_start, steps, sampling_noise=False):
    #     assert steps <= self.steps, "Too much steps in inference."
    #     if steps == 0:
    #         x_t = x_start
    #     else:
    #         t = torch.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
    #         x_t = self.q_sample(x_start, t)

    #     indices = list(range(self.steps))[::-1]

    #     if self.noise_scale == 0.: # No variance but only mean
    #         for i in indices:
    #             t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
    #             x_t = model(x_t, t)
    #         return x_t

    #     for i in indices:
    #         t = torch.tensor([i] * x_t.shape[0]).to(x_start.device) # [batch_size]
    #         out = self.p_mean_variance(model, x_t, t)
    #         if sampling_noise:
    #             noise = torch.randn_like(x_t)
    #             nonzero_mask = (
    #                 (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
    #             )  # no noise when t == 0
    #             x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
    #         else:
    #             x_t = out["mean"]
    #     return x_t

    def p_sample(self, model, x_start):
        n_row, n_col = x_start.size(0), x_start.size(1)
        x_t = torch.full((n_row, n_col), 1/n_col, device=self.device)

        for i in range(self.steps):
            t = torch.full((n_row,), 1/self.steps, device=self.device)
            tau_t = self.tau_max * torch.exp(-self.lamda * t)
            f_theta = model(x_t, t)
            tau_t = tau_t.unsqueeze(1).repeat((1, n_col))
            score = -tau_t + tau_t * n_col * nn.Softmax(dim=1)(f_theta)
            x_t += score / self.steps
        return x_t
        
    
    def training_losses(self, model, x_start, neg_x_start, reweight=False, denoise=True):
        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, 'uniform')
        dense_x_start = x_start.to_dense()
        noise = -torch.log(-torch.log(torch.rand_like(dense_x_start) + 1e-10) + 1e-10) #[batch, item_num]
        x_t, tau_t = self.q_sample(dense_x_start, ts, noise)

        terms = {}
        model_output = model(x_t, ts)
        target = {
            ModelMeanType.START_X: dense_x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_t.shape

        #mse = mean_flat((target - model_output) ** 2)
        score = mean_flat((model_output - (dense_x_start+torch.mul(tau_t, x_t))) ** 2)
        loss = score

        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = torch.where((ts == 0), 1.0, weight)
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / ((1-self.alphas_cumprod_prev[ts])**2 * (1-self.betas[ts]))
                weight = torch.where((ts == 0), 1.0, weight)
                likelihood = mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output))**2 / 2.0)
                loss = torch.where((ts == 0), likelihood, loss)
        else:
            weight = torch.tensor([1.0] * len(target)).to(device)

        terms["loss"] = weight * loss
        
        # update Lt_history & Lt_count
        for t, loss in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            else:
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                    self.Lt_count[t] += 1
                except:
                    print(t)
                    print(self.Lt_count[t])
                    print(loss)
                    raise ValueError

        terms["loss"] /= pt
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
    
    def q_sample(self, x_start, t, noise):
        tau_t = self.tau_max * torch.exp(-self.lamda*(1- t/self.steps)).unsqueeze(dim=1) # [batch,1]
        x_t  = torch.exp((x_start + (noise/self.beta)) / tau_t) # [batch, item_num]
        return nn.functional.normalize(x_t, dim=1, p=1), tau_t
    
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
    
    def p_mean_variance(self, model, x, t):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        B, C = x.shape[:2]
        assert t.shape == (B, )
        model_output = model(x, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped
        
        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        
        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)
        
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
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
    
    
        


