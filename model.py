import numpy as np
from numpy.linalg import norm
from scipy.special import softmax
from scipy.spatial.distance import cosine
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
        self.scale_f = 15
        self.cos_bias = 0.5

        with open('data/km_model.pickle', 'rb') as pk:
            self.kmeans = pickle.load(pk)
            
        self.cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)

        self.epsilon = 0.01
        self.y_0 = self.kmeans.cluster_centers_

    def update(self, Sl, Sf, A, u):
        Sl_next = Sl + self.epsilon * ( Sf.dot(A)/(norm(Sf)*norm(A)) - self.cos_bias )
        Sl_next = np.clip(Sl_next, a_min=0, a_max=1)
        # Sf_next = (Sf + Sl*A)/ max((Sf + Sl*A).sum(), 1e-2)
        # Sf_next = Sf
        Sf_next = softmax((Sf + Sl*A)*self.scale_f)
        return Sl_next, Sf_next, Sl == 0 or Sl == 1


    def reward(self, Sl, Sf, A):
        return Sl + Sf.dot(A)/(norm(Sf)*norm(A))
        # if Sl == 0.:
        #     return -1000.
        # elif Sl == 1.:
        #     return 1000.
        # else:
        #     return (self.cos_sim(Sf, A)-self.cos_bias).detach().cpu().item()

    def reset(self, u, seed=0):
        return 0.5, self.y_0[u]


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