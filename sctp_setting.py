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
import networkx as nx


@jit(nopython=True)
def update_b(c, b, path2od):
    for k, w in enumerate(path2od):
        bmin = b[w]
        if c[k] < bmin:
            b[w] = c[k]
    return b    


device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")


NETWORK = 'hearn'

node_count, link_count, flow_count, \
Innode, Outnode, Link, Odtree, Nodeloc, \
Capacity, Length, Fftime, Demand = read_notoll(NETWORK)
net = Network(NETWORK,Innode, Outnode, Link, Odtree)

if NETWORK == 'hearn':
    demand = torch.tensor(list(Demand.values()), dtype=torch.double)
    tfree = torch.tensor(list(Fftime.values()), dtype=torch.double)
    cap_original = torch.tensor(list(Capacity.values()), dtype=torch.double)
    toll_link = [5, 11, 2, 0]
    epsilon = 1e-6
    xi = 1e-10
    
    

if NETWORK == 'sf':
    
    
    demand = torch.tensor(list(Demand.values()), dtype=torch.double) / 1000
    Demand = dict(zip(Demand.keys(), [float(d) for d in demand]))
    tfree = torch.tensor(list(Fftime.values()), dtype=torch.double) / 0.1 * 0.06
    Fftime = dict(zip(Link, [float(t) for t in tfree]))
    cap_original = torch.tensor(list(Capacity.values()), dtype=torch.double) / 1000
    Capacity = dict(zip(Link, [float(c) for c in cap_original]))

    toll_link = [11, 14, 21, 32, 35, 38, 41, 45, 46, 48, 51, 52, 57, 64, 66, 68, 70, 73]
    
    
    epsilon = pow(10, -5)
    
elif NETWORK == "bar":
    demand = torch.tensor(list(Demand.values()), dtype=torch.double) * 2 / 1000
    Demand = dict(zip(Demand.keys(), [float(d) for d in demand]))
    tfree = torch.tensor(list(Fftime.values()), dtype=torch.double)
    cap_original = torch.tensor(list(Capacity.values()), dtype=torch.double) / 1000
    Capacity = dict(zip(Link, [float(c) for c in cap_original]))
    
    toll_link = [1514, 1464, 2265, 2517, 2274, 2062,  677,  704,  644,  674,  649, 2176,
                 1752, 1635, 2484, 1652, 1531, 1782, 2007, 1892]
    length = torch.tensor(list(Length.values()), dtype=torch.double)
    cash = length[toll_link]

    
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
    
    toll_link = [658,  566,  800, 1082,  810,  646,  654,  849, 1053,  807,  869,  857,
                 771,  789,  805,  847,  775,  650,  966,  801,  865,  796,  852,  621,
                 630,  853,  755,  861,  642,  638,  634,  782,  813,  856,  783,  561,
                 655,  662,  779,  860]
    
    epsilon = pow(10, -4)
    
    eta = 1
    ratio = 2
    
    length = torch.tensor(list(Length.values()), dtype=torch.double)





# net.solve_so(Demand, Fftime, Capacity, 1000, pow(10, -6), warm_start=False)
# x_so = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double)


net.solve_ue(Demand, Fftime, Capacity, 1000, pow(10, -6), warm_start=False)
x_ue = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double)
t_ue = tfree * (1 + 0.15 * (x_ue / cap_original) ** 4)
print(torch.dot(x_ue, t_ue))
net.path_enumeration()
net.generate_sparse_matrix()

G = nx.DiGraph()
for e in net.Link:
    G.add_edge(*e)
    
edge2num = dict()
for i, edge in enumerate(net.Link):
    edge2num[edge] = i
    
path13 = list(nx.all_simple_paths(G, 1, 3))
path14 = list(nx.all_simple_paths(G, 1, 4))
path23 = list(nx.all_simple_paths(G, 2, 3))
path24 = list(nx.all_simple_paths(G, 2, 4))

path_number = len(path13) + len(path14) + len(path23) + len(path24)
path_edge = np.zeros((len(net.Link), path_number), dtype=np.double)
path_demand = np.zeros((len(demand), path_number), dtype=np.double)

k = 0
for path in path13:
    path_demand[0, k] = 1
    for i in range(len(path) - 1):
        e = (path[i], path[i + 1])
        path_edge[edge2num[e], k] = 1
    k += 1
    
for path in path14:
    path_demand[1, k] = 1
    for i in range(len(path) - 1):
        e = (path[i], path[i + 1])
        path_edge[edge2num[e], k] = 1
    k += 1
    
for path in path23:
    path_demand[2, k] = 1
    for i in range(len(path) - 1):
        e = (path[i], path[i + 1])
        path_edge[edge2num[e], k] = 1
    k += 1
    
for path in path24:
    path_demand[3, k] = 1
    for i in range(len(path) - 1):
        e = (path[i], path[i + 1])
        path_edge[edge2num[e], k] = 1
    k += 1
        


# path_edge = net.path_edge
# path_demand = net.path_demand
# path_number = net.path_number

# bar = 1.05

# edge_toll = torch.where(x_ue / x_so > bar)[0]



# with open('net_' + NETWORK + '.pkl', 'rb') as f:
#     net = pickle.load(f)



# path_edge = net.path_edge.to(device)
# path_demand = net.path_demand.to(device)
# path_number = net.path_number


# path_number = path_edge.size()[1]
# q = path_demand.t() @ demand

# toll = torch.zeros_like(cap_original)

# toll.requires_grad_()
# iter_num = 0
# gap_old = 1

# r = 0.01
    
    

# p_0 = torch.ones(path_number, dtype=torch.double).to(device)
# p_0 /= path_demand.t() @ (path_demand @ p_0)
# p = p_0 * 1.0

# kk = 0
# gap = torch.inf

# # if NETWORK == 'cs':
# #     with torch.no_grad():
# #         while True:
# #             f = q * p
# #             x = path_edge @ f
# #             t = tfree * (1 + 0.15 * (x / cap) ** 4)
# #             c = path_edge.t() @ t
# #             p *= torch.exp(-r * c)
# #             p /= path_demand.t() @ (path_demand @ p)
            
            
# #             b = torch.zeros(len(demand), dtype=torch.double) + torch.inf
# #             path2od = np.array(net.path2od)
# #             b = update_b(c.detach().cpu().numpy(), b.numpy(), path2od)    
# #             b = torch.tensor(b, dtype=torch.double).to(device)                   
# #             gt = torch.dot(b, demand)
# #             tt = torch.dot(c, f)
# #             gap = (tt - gt) / tt
# #             # print(gap)
# #             if gap < 1e-3:
# #                 break

# #     p[p < 1e-20] = 0
# while True:

#     f = q * p
#     x = path_edge @ f
#     u = tfree * (1 + 0.15 * (x / cap_original) ** 4) + toll
#     c = path_edge.t() @ u
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
# t = tfree * (1 + 0.15 * (x / cap_original) ** 4)

# TT = torch.dot(x, t) 
# TT.backward()

# toll_link = torch.sort(toll.grad, descending=False)[1][:4]

