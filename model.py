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
    def __init__(self, Sl,Sf, A, time_horizon, bias=0.5, normal_scale=0.05):
        self.slow_state = tc.zeros(Sl)
        self.fast_state = tc.zeros(Sf)
        self.action = tc.zeros(A)
        self.time_horizon = time_horizon
        self.Sl_d, self.Sf_d, self.A_d = Sl, Sf, A
        self.scale_f = 20
        self.cos_bias = bias
        self.normal_scale = normal_scale

        with open('data/km_model.pickle', 'rb') as pk:
            self.kmeans = pickle.load(pk)
            
        self.cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)

        self.epsilon = 0.01
        self.y_0 = self.kmeans.cluster_centers_

    def update(self, Sl, Sf, A, u):
        Sl_next = Sl + self.epsilon * ( Sf.dot(A)/(norm(Sf)*norm(A)) - self.cos_bias )
        Sl_next = np.clip(Sl_next, a_min=0., a_max=1.)
        # Sf_next = (Sf + Sl*A)/ max((Sf + Sl*A).sum(), 1e-2)
        # Sf_next = Sf
        w = np.random.normal(0, self.normal_scale, size=len(Sf))
        # w = 0
        Sf_next = softmax((Sf + Sl*A + w)*self.scale_f)
        # y = np.random.choice(len(Sf), 1, p=softmax((Sf + Sl*A)*self.scale_f))
        # Sf_next = np.zeros(len(Sf))
        # Sf_next[y] = 1
        self.t -= 1
        return Sl_next, Sf_next, self.t==0


    def reward(self, Sl, Sf, A):
        if Sl <= 0.:
            return -100.
        elif Sl >= 1.:
            return 100.
        else:
            return (Sl+Sf.dot(A)/(norm(Sf)*norm(A))) - self.cos_bias
        # if Sl == 0.:
        #     return -1000.
        # elif Sl == 1.:
        #     return 1000.
        # else:
        #     return (self.cos_sim(Sf, A)-self.cos_bias).detach().cpu().item()

    def reset(self, u, seed=0):
        self.t = self.time_horizon
        return 0.5, self.y_0[u]

