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
from scipy.optimize import minimize
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

toll_add = np.array([0, 0], dtype=np.double)
toll = np.zeros(len(tfree), dtype=np.double)
toll[toll_link] = toll_add

Toll = dict()
for kk, ii in enumerate(net.Link):
    Toll[ii] = float(toll[kk])
net.solve_ue(Demand, Fftime, Capacity, 1000, 1e-10, warm_start=True, toll=Toll, delete=False)
f = np.zeros(path_number, dtype=np.double)
for o in net.Odtree.keys():
    for d in net.Odtree[o]:
        flows = net.pathflow[(o, d)]
        for path in flows.keys():
            k = path2num[path]
            f[k] = flows[path]

# toll = np.zeros(len(tfree), dtype=np.double)
def obj(decision): 
    
    f = decision[len(toll_link):]
    x = path_edge @ f
    t = tfree * (1 + 0.15 * (x / cap) ** 4)
    TT = np.dot(x, t)
    return TT

def link_time(decision): 
        
    f = decision[len(toll_link):]
    x = path_edge @ f
    t = tfree * (1 + 0.15 * (x / cap) ** 4)

    return t

def link_cost(decision):
    toll = np.zeros(len(tfree), dtype=np.double)
    toll_add = decision[:len(toll_link)]
    toll[toll_link] = toll_add
    t = link_time(decision)
    return t + toll

def eq_constraint(decision):
    
    f = decision[len(toll_link):]
    
    return path_demand @ f - demand


tic = tm.time()

cuts = [x_so]
stop = False

kkk = 0
eq_constraints = [{'type': 'eq', 'fun': eq_constraint}]

toll_init = toll_add * 1.0
f_init = f * 1.0
while True:
    kkk += 1
    
    initial_guess = np.concatenate([toll_add, f])
    # initial_guess = np.concatenate([toll_init, f_init])
    
    bounds = [(0, None) for _ in range(len(initial_guess))]

    non_constraints = [{'type': 'eq', 'fun': 
                        lambda decision, cut=cut: np.dot(link_cost(decision), cut - path_edge @ decision[len(toll_link):]) + 1e-6}
                       for cut in cuts]
        
    constraints = eq_constraints + non_constraints

    # for constraint in constraints:
    #     print(constraint['fun'](initial_guess))
                   
    result = minimize(obj, initial_guess, bounds=bounds, constraints=constraints)
    print(result.message)
    decision = result.x
    toll_add = decision[:len(toll_link)]
    # print(toll_add)
    print('lower bound:', result.fun)
    
    f_sub = decision[len(toll_link):]
    x_sub = path_edge @ f_sub
    toll = np.zeros(len(tfree), dtype=np.double)
    toll[toll_link] = toll_add
    u_sub = tfree * (1 + 0.15 * (x_sub / cap) ** 4) + toll
    cut = net.all_or_nothing(dict(zip(Link, [float(t) for t in u_sub]))).numpy()
    
    # cuts.append(cut)
    same = False
    for cut_o in cuts:
        if np.array_equal(cut_o, cut):
            same = True
            break
    if not same:
        cuts.append(cut * 1.0)
    else:
        stop = True
    
    
    
    toll = np.zeros(len(tfree), dtype=np.double)
    toll[toll_link] = toll_add
    Toll = dict()
    for kk, ii in enumerate(net.Link):
        Toll[ii] = float(toll[kk])
    net.solve_ue(Demand, Fftime, Capacity, 1000, 1e-10, warm_start=True, toll=Toll, delete=False)
    
    f = np.zeros(path_number, dtype=np.double)
    for o in net.Odtree.keys():
        for d in net.Odtree[o]:
            flows = net.pathflow[(o, d)]
            for path in flows.keys():
                k = path2num[path]
                f[k] = flows[path]
                
    x_current = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double).numpy()
   
    # cuts.append(x_current)
    
    TT = total_travel_time(x_current)
    print('upper bound:', TT)
    
    if stop:
        break
    

    
    
# toll_add = np.array([0, 0], dtype=np.double)
# toll = np.zeros(len(tfree), dtype=np.double)
# toll[toll_link] = toll_add

# Toll = dict()
# for kk, ii in enumerate(net.Link):
#     Toll[ii] = float(toll[kk])
# net.solve_ue(Demand, Fftime, Capacity, 1000, 1e-10, warm_start=True, toll=Toll, delete=False)
# f = np.zeros(path_number, dtype=np.double)
# for o in net.Odtree.keys():
#     for d in net.Odtree[o]:
#         flows = net.pathflow[(o, d)]
#         for path in flows.keys():
#             k = path2num[path]
#             f[k] = flows[path]  
# initial_guess = np.concatenate([toll_add, f])
# # initial_guess = np.concatenate([toll_init, f_init])

# bounds = [(0, None) for _ in range(len(initial_guess))]

# non_constraints = [{'type': 'eq', 'fun': 
#                     lambda decision, cut=cut: np.dot(link_cost(decision), cut - path_edge @ decision[len(toll_link):]) + 1e-6}
#                    for cut in cuts]
    
# constraints = eq_constraints + non_constraints

               
# result = minimize(obj, initial_guess, bounds=bounds, constraints=constraints, options={'maxiter': 1e+5})

# decision = result.x
# toll_add = decision[:len(toll_link)]


# toc = tm.time()

# #solution evaluation
# with torch.no_grad():
#     toll = torch.zeros_like(tfree)
#     toll[toll_link] = toll_add
    
# Toll = dict()
# for kk, ii in enumerate(net.Link):
#     Toll[ii] = float(toll[kk])

# net.solve_ue(Demand, Fftime, Capacity, 1000, eps_eva, warm_start=False, toll=Toll)
# x = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double).to(device)
# t = tfree * (1 + 0.15 * (x / cap) ** 4)
# TT = torch.dot(x, t)
# obj = (TT - TT_so) / (TT_ue - TT_so)
# print(obj)

# Result = dict()
# Result['solution'] = toll_add.detach()
# Result['obj'] = obj
# # Result['time_traj'] = time_traj
# # Result['solution_traj'] = solution_traj
# Result['time'] = toc - tic


# # with open('result_sctp/' + 'dolmd_' + NETWORK + '.pkl', 'wb') as f:
# #     pickle.dump(Result, f)
                