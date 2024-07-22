# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:25:46 2024

@author: Jiayang Li
"""
import pickle
from read import read_notoll
from graph import Network
from copy import deepcopy
import numpy as np
import torch
import scipy
from scipy.optimize import minimize
import time as tm
from numba import jit


@jit(nopython=True)
def update_b(c, b, path2od):
    for k, w in enumerate(path2od):
        bmin = b[w]
        if c[k] < bmin:
            b[w] = c[k]
    return b    


device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")


NETWORK = 'bar'

node_count, link_count, flow_count, \
Innode, Outnode, Link, Odtree, Nodeloc, \
Capacity, Length, Fftime, Demand = read_notoll(NETWORK)
net = Network(NETWORK,Innode, Outnode, Link, Odtree)

if NETWORK == 'sf':
    
    demand = torch.tensor(list(Demand.values()), dtype=torch.double) / 100 * 0.11
    Demand = dict(zip(Demand.keys(), [float(d) for d in demand]))
    tfree = torch.tensor(list(Fftime.values()), dtype=torch.double) / 0.1 * 0.06
    Fftime = dict(zip(Link, [float(t) for t in tfree]))
    cap_original = torch.tensor(list(Capacity.values()), dtype=torch.double) / 1000
    Capacity = dict(zip(Link, [float(c) for c in cap_original]))

    enh_link = [15, 18, 16, 19, 24, 25, 28, 47, 38, 73]
    cash = torch.tensor([26, 26, 40, 40, 25, 25, 48, 48, 34, 34], dtype=torch.double)
    
    eta = 0.001
    cap_scale = 1
    
    ratio = 2
    
    epsilon = pow(10, -5)
    
elif NETWORK == "bar":
    demand = torch.tensor(list(Demand.values()), dtype=torch.double) * 2 / 1000
    Demand = dict(zip(Demand.keys(), [float(d) for d in demand]))
    tfree = torch.tensor(list(Fftime.values()), dtype=torch.double)
    cap_original = torch.tensor(list(Capacity.values()), dtype=torch.double) / 1000
    Capacity = dict(zip(Link, [float(c) for c in cap_original]))
    
    enh_link = [1416, 2143, 1120, 2484, 2298,  415, 1418, 2269,  410,  616, 2006, 2118,
                1897, 1456,  540, 1514, 2165, 2281, 2033,  537, 2208, 1106, 1459, 1917,
                1661, 2517, 2433,  630, 2508, 1548, 2287, 1457, 2276, 2151, 1460, 2274,
                463, 2009, 1316, 1635, 2109, 1448, 1331,  704,  677, 2482, 1888, 1652,
                1531,  471]
    length = torch.tensor(list(Length.values()), dtype=torch.double)
    cash = length[enh_link]

    
    eta = 1
    cap_scale = 1000
    
    ratio = 2
    
    epsilon = pow(10, -5)
    
elif NETWORK == "cs":
    demand = torch.tensor(list(Demand.values()), dtype=torch.double) * 2 / 1000
    Demand = dict(zip(Demand.keys(), [float(d) for d in demand]))
    tfree = torch.tensor(list(Fftime.values()), dtype=torch.double)
    cap_original = torch.tensor(list(Capacity.values()), dtype=torch.double) / 1000
    Capacity = dict(zip(Link, [float(c) for c in cap_original]))
    
    enh_link = [964, 2949, 1214, 2126,  944,  594, 2170,  433,  519,  954, 2138, 2778,
                462,  946,  528, 1798,  961, 1314, 2380,  925, 2925,  760,  530, 1086,
                1734, 2927, 2173,  447,  575,  750, 1688,  424, 2193,  522,  804,  806,
                448, 2192,  587,  445,  443,  736, 1227, 1402, 1031, 2119,  955, 1311,
                2109, 2180]
    
    epsilon = pow(10, -4)
    
    eta = 1
    ratio = 2
    
    length = torch.tensor(list(Length.values()), dtype=torch.double)


with open('result_cndp/net_' + NETWORK + '.pkl', 'rb') as f:
    net = pickle.load(f)


method = 'sa0'
NETWORK = 'bar'
path_to_pickle = 'result_cndp/' + method + '_' + NETWORK + '.pkl'
with open(path_to_pickle, 'rb') as f:
    Result = pickle.load(f)
    
cap_add = Result['solution']

# cap_add = torch.zeros_like(cap_original)
cap = cap_original
cap[enh_link] += cap_add
Capacity = dict(zip(Link, [float(c) for c in cap]))

net.solve_ue(Demand, Fftime, Capacity, 1000, 1e-3, warm_start=True)

x = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double)

t = tfree * (1 + 0.15 * (x / cap) ** 4)
TT = np.dot(x, t)
obj = TT + eta * np.dot(cash, cap_add ** ratio)
    
    

# path_edge = net.path_edge.to(device)
# path_demand = net.path_demand.to(device)
# path_number = net.path_number


# path_number = path_edge.size()[1]
# q = path_demand.t() @ demand


# cap_add.requires_grad_()
# iter_num = 0
# gap_old = 1

# tic = tm.time()
# time_traj = list()
# solution_traj = list()
# cap_old = cap_add * 1.0
# r = 1
    
# cap = cap_original + cap_add   
    

# p_0 = torch.ones(path_number, dtype=torch.double).to(device)
# p_0 /= path_demand.t() @ (path_demand @ p_0)
# p = p_0 * 1.0

# kk = 0
# gap = torch.inf

# if NETWORK == 'cs':
#     with torch.no_grad():
#         while True:
#             f = q * p
#             x = path_edge @ f
#             t = tfree * (1 + 0.15 * (x / cap) ** 4)
#             c = path_edge.t() @ t
#             p *= torch.exp(-r * c)
#             p /= path_demand.t() @ (path_demand @ p)
            
            
#             b = torch.zeros(len(demand), dtype=torch.double) + torch.inf
#             path2od = np.array(net.path2od)
#             b = update_b(c.detach().cpu().numpy(), b.numpy(), path2od)    
#             b = torch.tensor(b, dtype=torch.double).to(device)                   
#             gt = torch.dot(b, demand)
#             tt = torch.dot(c, f)
#             gap = (tt - gt) / tt
#             # print(gap)
#             if gap < 1e-3:
#                 break

#     p[p < 1e-20] = 0
# while True:

#     f = q * p
#     x = path_edge @ f
#     t = tfree * (1 + 0.15 * (x / cap) ** 4)
#     c = path_edge.t() @ t
#     p *= torch.exp(-r * c)
#     p /= path_demand.t() @ (path_demand @ p)
#     # p[p < 1e-20] = 0
#     with torch.no_grad():
        
#         b = torch.zeros(len(demand), dtype=torch.double) + torch.inf
#         path2od = np.array(net.path2od)
#         b = update_b(c.detach().cpu().numpy(), b.numpy(), path2od)    
#         b = torch.tensor(b, dtype=torch.double).to(device)                   
#         gt = torch.dot(b, demand)
#         tt = torch.dot(c, f)
#         gap = (tt - gt) / tt
#         print(gap)
#         # gap = compute_gap(c, f)
#         if gap < epsilon:
#             break
#         # if gap > gap_old:
#         #     r *= 0.5
#         gap_old = gap
#         kk += 1

# f = q * p
# x = path_edge @ f
# t = tfree * (1 + 0.15 * (x / cap) ** 4)

# TT = torch.dot(x, t) 
# objective = TT
# objective.backward()

# # enh_link = torch.sort(cap_add.grad, descending=False)[1][:50]

