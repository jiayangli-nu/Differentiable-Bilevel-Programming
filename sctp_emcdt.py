# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:25:46 2024

@author: 84091
"""
import torch
import pickle
import numpy as np
import time as tm
from sctp_load import Instance



def total_travel_time(x):
    t = tfree * (1 + 0.15 * (x / cap) ** 4)
    return torch.dot(x, t)


for NETWORK in ['sf']:
    

    
    with open('result_sctp/net_' + NETWORK + '.pkl', 'rb') as f:
        net = pickle.load(f)
   
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
    NumberofOD = len(net.pathflow)
    
    non_link = [i for i in range(len(tfree)) if i not in toll_link]

    dtime = lambda x: 0.6 * tfree * x ** 3 / cap ** 4


    delta = 1e-4
    
    x_ue = torch.load('result_sctp/ue_' + NETWORK + '.pt')
    x_so = torch.load('result_sctp/so_' + NETWORK + '.pt')
    
    TT_ue = total_travel_time(x_ue)
    TT_so = total_travel_time(x_so)
    
    toll_so = dtime(x_so) * x_so
    
    toll = toll_so * 1.0
    toll[toll < delta] = delta
    toll[non_link] = 0
    
    
    tic = tm.time()
    
    time_traj = list()
    solution_traj = list()
    
    i = 1
    c = 1
    toll_old = toll * 1.0
    while True:
        # print(toll[edge_toll])
        
        Toll = dict()
        for kk, ii in enumerate(net.Link):
            Toll[ii] = float(toll[kk])
            
            
        net.solve_ue(Demand, Fftime, Capacity, 1000, epsilon, warm_start=True, toll=Toll)
        
        x_ue = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double)
        # print(x_ue)
        TT = total_travel_time(x_ue)
        print(TT)
        
        mtoll = dtime(x_ue) * x_ue
        
        alpha = torch.max(mtoll)
        for e in toll_link:
            ratio = c / max(1, alpha)
            
            diff = mtoll[e] - toll_so[e]
            
            toll[e] *= torch.exp(ratio * diff)
        
        change = float(torch.norm(toll - toll_old, torch.inf))
        if change < xi:
            break

        toll_old = toll * 1.0
            
        i += 1
        c *= 0.9
        
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
    
    # A = np.zeros((1, len(toll_link) + 1), dtype=np.double)

        
    # A[0, 0] = obj.numpy()
    # A[0, 1:] = toll[toll_link].numpy()
    with open('result_sctp/' + 'emcdt_' + NETWORK + '.pkl', 'wb') as f:
        pickle.dump(Result, f)
        
        
        
    




