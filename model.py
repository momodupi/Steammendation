import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

import pickle

DEVICE = tc.device("cuda:1" if tc.cuda.is_available() else "cpu")

class Model(object):
    def __init__(self, Sl,Sf, A, time_horizon):
        self.slow_state = tc.zeros(Sl)
        self.fast_state = tc.zeros(Sf)
        self.action = tc.zeros(A)
        self.time_horizon = time_horizon
        self.Sl_d, self.Sf_d, self.A_d = Sl, Sf, A
        self.scale_f = 50

        with open('data/km_model.pickle', 'rb') as pk:
            self.kmeans = pickle.load(pk)

        self.epsilon = 0.2
        self.y_0 = tc.tensor(self.kmeans.cluster_centers_, dtype=tc.float).to(DEVICE)

    def update(self, Sl, Sf, A, u):
        Sl_next = Sl + self.epsilon*tc.dot(Sf, A)
        Sl_next = tc.clip(Sl_next, min=0, max=1)
        # Sf_next = (Sf + Sl*A)/(Sf + Sl*A).sum()
        Sf_next = F.softmax((Sf + Sl*A)*self.scale_f, dim=0)
        # print(tc.exp(Sf + Sl*A))
        # Sf_next = tc.exp(Sf + Sl*A) / tc.sum(tc.exp(Sf + Sl*A))
        return Sl_next, Sf_next, Sl == 0

    def reward(self, Sl, Sf, A):
        return tc.dot(Sf, A) if Sl != 0 else -100

    def reset(self, u, seed=0):
        return 0.5, tc.ones(len(self.y_0[u])).to(DEVICE)


if __name__ == '__main__':
    with open('data/dimension.pickle', 'rb') as pk:
        dim_info = pickle.load(pk)
    Sl_d, Sf_d, A_d = 1, dim_info['tages'], dim_info['tages']
    T = 100
    user_class = 9
    model = Model(Sl_d, Sf_d, A_d, T)
    sl, sf = model.reset(user_class)
    A = model.y_0[user_class]
    
    for _ in range(model.time_horizon):
        
        sl, sf, terminal = model.update(sl, sf, A, user_class)
        if terminal: break
        
    print(sl, sf, A, model.reward(sl, sf, A))