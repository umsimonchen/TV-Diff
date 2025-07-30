import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss,l2_reg_loss
import numpy as np
import os 
import random

# paper: Neural Collaborative Filtering, WWW'17
# https://github.com/hexiangnan/neural_collaborative_filtering

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

class NeuMF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(NeuMF, self).__init__(conf, training_set, test_set)
        self.model = NeuMF_Encoder(self.data, self.emb_size)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        while not early_stopping:
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                label = torch.cat([torch.ones(len(pos_idx)), torch.zeros(len(neg_idx))]).cuda()
                output = model(user_idx+user_idx, pos_idx+neg_idx)
                batch_loss = self.loss(output, label)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                prediction_list = []
                for user in range(self.data.user_num):
                    output = self.sigmoid(model([user]*self.data.item_num, list(range(self.data.item_num)))).unsqueeze(0)
                    prediction_list.append(output.detach().cpu())
                    if user % 1000==0 and n>0:
                        print("Finished user %d / %d"%(user, self.data.user_num))
                del output
                self.prediction = np.concatenate(prediction_list, axis=0)
            _, early_stopping = self.fast_evaluation(epoch)
            epoch += 1
        self.prediction = self.best_prediction
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")

    def save(self):
        with torch.no_grad():
            self.best_prediction = self.prediction

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = self.prediction[u]
            return score

class MLPLayers(nn.Module):
    r"""MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """

    def __init__(
        self,
        layers,
        dropout=0.0,
        activation="relu",
        bn=False,
        init_method=None,
        last_activation=True,
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)
        if self.activation is not None and not last_activation:
            mlp_modules.pop()
        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            if self.init_method == "norm":
                nn.initi.normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)

def activation_layer(activation_name="relu", emb_dim=None):
    """Construct activation layers

    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation

    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation

class NeuMF_Encoder(nn.Module):
    def __init__(self, data, emb_size):
        super(NeuMF_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.embedding_dict = self._init_model()
        self.mlp_layers = MLPLayers([2*self.latent_size]+[self.latent_size], 0.5)
        self.predict_layer = nn.Linear(self.latent_size+self.latent_size, 1)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_mf': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_mf': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
            'user_mlp': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_mlp': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self, user_idx, pos_idx):
        user_mf_e = self.embedding_dict['user_mf'][user_idx]
        item_mf_e = self.embedding_dict['item_mf'][pos_idx]
        user_mlp_e = self.embedding_dict['user_mlp'][user_idx]
        item_mlp_e = self.embedding_dict['item_mlp'][pos_idx]
        
        mf_output = torch.mul(user_mf_e, item_mf_e)
        mlp_output = self.mlp_layers(torch.cat([user_mlp_e,item_mlp_e], -1))
        output = self.predict_layer(torch.cat((mf_output, mlp_output), -1))
        return output.squeeze(-1)


