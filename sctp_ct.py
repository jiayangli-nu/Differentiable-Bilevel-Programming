# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:25:46 2024

@author: 84091
"""
import torch
import pickle
import time as tm
from sctp_load import Instance
from numba import jit
import numpy as np


@jit(nopython=True)
def update_b(c, b, path2od):
    for k, w in enumerate(path2od):
        bmin = b[w]
        if c[k] < bmin:
            b[w] = c[k]
    return b    


def total_travel_time(x):
    t = tfree * (1 + 0.15 * (x / cap) ** 4)
    return torch.dot(x, t)


for NETWORK in ['hearn1']:
   
    instance = Instance(NETWORK)
        
    Capacity = Instance.Capacity
    Length = Instance.Length
    Fftime = Instance.Fftime
    Demand = Instance.Demand
    Link = Instance.Link
    
    demand = Instance.demand
    cap = Instance.cap
    tfree = Instance.tfree
    toll_link = Instance.toll_link
    epsilon = Instance.epsilon
    xi = Instance.xi
    eps_eva = Instance.eps_eva
    
    Numberoflink = len(Link)
    
    non_link = [i for i in range(len(tfree)) if i not in toll_link]

    dtime = lambda x: 0.6 * tfree * x ** 3 / cap ** 4
    
    
    with open('result_sctp/net_' + NETWORK + '.pkl', 'rb') as f:
        net = pickle.load(f)


    
    # delta = 0.1
    
    x_ue = torch.load('result_sctp/ue_' + NETWORK + '.pt')
    x_so = torch.load('result_sctp/so_' + NETWORK + '.pt')
    
    TT_ue = total_travel_time(x_ue)
    TT_so = total_travel_time(x_so)
    
    toll_so = dtime(x_so) * x_so
    
    if NETWORK == 'hearn1' or NETWORK == 'hearn2':
        Delta_list = [0.001]
    
    if NETWORK == 'sf':
        Delta_list = [0.001, 0.01]
    if NETWORK == 'bar' or NETWORK == 'cs':
        Delta_list = [0.01, 0.1]
    for delta in Delta_list:
        tic = tm.time()
        
        time_traj = list()
        solution_traj = list()
    
        toll = torch.zeros_like(cap)
        
        tollable = torch.zeros_like(toll, dtype=torch.bool)
        tollable[toll_link] = True
        
        while torch.any(tollable):
            
            
            Toll = dict()
            for kk, ii in enumerate(net.Link):
                Toll[ii] = float(toll[kk])
            
            net.solve_ue(Demand, Fftime, Capacity, 1000, epsilon, warm_start=True, toll=Toll)
            
            x_ue = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double)
            
            TT = total_travel_time(x_ue)
            print(float(TT))
            
            mtoll = dtime(x_ue) * x_ue
            
            subset_indices = torch.where(tollable)[0]
            subset_values = mtoll[tollable]
        
            max_value, max_idx_in_subset = torch.max(subset_values, dim=0)
        
            e = subset_indices[max_idx_in_subset]
            
            if x_ue[e] > x_so[e]:
                toll[e] += delta
            else:
                tollable[e] = False
                
                print("delete:", int(e))
            solution_traj.append(toll[toll_link])
            time_traj.append(tm.time() - tic)
                
                
            
                
        
        toc = tm.time()
            
        with torch.no_grad():
            Toll = dict()
            for kk, ii in enumerate(net.Link):
                Toll[ii] = float(toll[kk])
                                 
            net.solve_ue(Demand, Fftime, Capacity, 1000, eps_eva, warm_start=True, toll=Toll)
            x = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double)
            t = tfree * (1 + 0.15 * (x / cap) ** 4)
            TT = torch.dot(x, t)
            obj = (TT - TT_so) / (TT_ue - TT_so)
            
            print("objective: ", obj)
        
        Result = dict()
        Result['solution'] = toll[toll_link]
        Result['obj'] = obj
        Result['time_traj'] = time_traj
        Result['solution_traj'] = solution_traj
        Result['time'] = toc - tic
        
        with open('result_sctp/' + 'ct_' + NETWORK + '.pkl', 'wb') as f:
            pickle.dump(Result, f)
            
    # A = np.zeros((1, len(toll_link) + 1), dtype=np.double)

        
    # A[0, 0] = obj.numpy()
    # A[0, 1:] = toll[toll_link].numpy()
    
    




