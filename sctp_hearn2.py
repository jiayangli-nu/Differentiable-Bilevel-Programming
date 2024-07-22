# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:25:46 2024

@author: Jiayang Li
"""
import torch
import pickle
import numpy as np
import time as tm
import networkx as nx
from numba import jit
from sctp_load import Instance
from scipy.optimize import minimize, brute
from read import read_notoll
from graph import Network


@jit(nopython=True)
def update_b(c, b, path2od):
    for k, w in enumerate(path2od):
        bmin = b[w]
        if c[k] < bmin:
            b[w] = c[k]
    return b    
    

def total_travel_time(x):
    t = tfree * (1 + 0.15 * (x / cap) ** 4)
    return np.dot(x, t)


NETWORK = 'hearn'
    

instance = Instance(NETWORK)
    
Capacity = Instance.Capacity
Length = Instance.Length
Fftime = Instance.Fftime
Demand = Instance.Demand
Link = Instance.Link

demand = Instance.demand.numpy()
cap = Instance.cap.numpy()
tfree = Instance.tfree.numpy()
toll_link = Instance.toll_link
epsilon = Instance.epsilon
xi = Instance.xi
eps_eva = Instance.eps_eva


with open('result_sctp/net_' + NETWORK + '.pkl', 'rb') as f:
    net = pickle.load(f)
    
G = nx.DiGraph()
for e in net.Link:
    G.add_edge(*e)
    
edge2num = dict()
path2num = dict()
for i, edge in enumerate(net.Link):
    edge2num[edge] = i
    
Path_set = list()
path_number = 0
for od in Demand.keys():
    o = od[0]
    d = od[1]
    paths = list(nx.all_simple_paths(G, o, d))
    path_number += len(paths)
    Path_set.append(paths)

path_edge = np.zeros((len(net.Link), path_number), dtype=np.double)
path_demand = np.zeros((len(demand), path_number), dtype=np.double)


k = 0
for ii, paths in enumerate(Path_set):
    for path in paths:
        path2num[tuple(path)] = k
        path_demand[ii, k] = 1
        for i in range(len(path) - 1):
            e = (path[i], path[i + 1])
            path_edge[edge2num[e], k] = 1
        k += 1

# torch.manual_seed(0)

x_ue = torch.load('result_sctp/ue_' + NETWORK + '.pt').numpy()
x_so = torch.load('result_sctp/so_' + NETWORK + '.pt').numpy()
TT_ue = total_travel_time(x_ue)
TT_so = total_travel_time(x_so)
dtime = lambda x: 0.6 * tfree * x ** 3 / cap ** 4
toll_so = dtime(x_so) * x_so

non_link = [i for i in range(len(tfree)) if i not in toll_link]

    
NumberofOD = len(demand)


q = path_demand.T @ demand

# toll_add = np.zeros(len(toll_link), dtype=np.double)



# toll = np.zeros(len(tfree), dtype=np.double)
def obj(toll_add): 
    print(toll_add)
    toll = np.zeros(len(tfree), dtype=np.double)
    toll[toll_link] = toll_add

    Toll = dict()
    for kk, ii in enumerate(net.Link):
        Toll[ii] = float(toll[kk])
    net.solve_ue(Demand, Fftime, Capacity, 10000, 1e-10, warm_start=True, toll=Toll, delete=False)
    x = np.array([net.flow[ii] for ii in net.Link], dtype=np.double)    
    t = tfree * (1 + 0.15 * (x / cap) ** 4)
    TT = np.dot(x, t)
    return TT

tic = tm.time()
bounds = [(0, None) for _ in range(len(toll_link))]
ranges = [(0, 10) for _ in range(len(toll_link))]
result = brute(obj, ranges=ranges)
result = minimize(obj, result, bounds=bounds, method='Powell', tol=1e-50)

toll_add = result.x
toll = np.zeros(len(tfree), dtype=np.double)
toll[toll_link] = toll_add
Toll = dict()
for kk, ii in enumerate(net.Link):
    Toll[ii] = float(toll[kk])
                     
net.solve_ue(Demand, Fftime, Capacity, 1000, 1e-10, warm_start=True, toll=Toll)
x = np.array([net.flow[ii] for ii in net.Link], dtype=np.double)
t = tfree * (1 + 0.15 * (x / cap) ** 4)
TT = np.dot(x, t)
obj = (TT - TT_so) / (TT_ue - TT_so)

A = np.zeros((1, 1 + len(toll_link)), dtype=np.double)

    
A[0, 0] = obj
A[0, 1:] = toll_add
                