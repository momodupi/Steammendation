import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

import pickle

class Model(object):
    def __init__(self, Sl,Sf, A, time_horizon):
        self.slow_state = tc.zeros(Sl)
        self.fast_state = tc.zeros(Sf)
        self.action = tc.zeros(A)
        self.time_horizon = time_horizon
        self.Sl_d, self.Sf_d, self.A_d = Sl, Sf, A

        with open('data/epsilon.pickle', 'rb') as pk:
            self.epsilon = pickle.load(pk)
        self.f2_nn_dict = tc.load('data/f2_nn')

        with open('data/km_model.pickle', 'rb') as pk:
            self.kmeans = pickle.load(pk)

        self.net_dict = {}
        f2_input_layer_dim = Sl + Sf + A
        f2_output_layer_dim = Sf
        f2_hidden_width = f2_input_layer_dim
        W_SCALE = 5
        for u in self.f2_nn_dict:
            net = nn.Sequential(nn.Linear(f2_input_layer_dim, W_SCALE*f2_hidden_width), 
                    nn.ReLU(), nn.Linear(W_SCALE*f2_hidden_width, W_SCALE*f2_hidden_width),
                    nn.ReLU(), nn.Linear(W_SCALE*f2_hidden_width, f2_output_layer_dim),
                    nn.Softmax())
            net.load_state_dict(self.f2_nn_dict[u])
            self.net_dict[u] = net
        
        self.users_cnt = len(self.f2_nn_dict)

    def update(self, Sl, Sf, A, u):
        Sl_next = Sl + self.epsilon[u]*tc.dot(Sf, A) 
        input_data = tc.hstack([tc.tensor([Sl]), Sf, A])
        Sf_next = self.net_dict[u](input_data)

        return Sl_next, Sf_next

    def reward(self, Sl, Sf, A):
        return Sl + tc.dot(Sf, A) 

    def initial_state(self, u, seed=0):
        Sf = tc.tensor(self.kmeans.cluster_centers_[u], dtype=tc.float)
        return 0, Sf

if __name__ == '__main__':
    with open('data/dimension.pickle', 'rb') as pk:
        dim_info = pickle.load(pk)
    Sl_d, Sf_d, A_d = 1, dim_info['tages'], dim_info['tages']
    T = 100
    user_class = 39
    model = Model(Sl_d, Sf_d, A_d, T)
    sl, sf = model.initial_state(user_class)
    A = sf
    
    
    for _ in range(model.time_horizon):
        A = sf
        sl, sf = model.update(sl, sf, A, user_class)
        print(sl, sf, model.reward(sl, sf, A))