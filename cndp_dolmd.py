# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:25:46 2024

@author: Jiayang Li
"""
import torch
import pickle
import numpy as np
import time as tm
from numba import jit
from cndp_load import Instance

# from scipy.optimize import minimize_scalar


@jit(nopython=True)
def update_b(c, b, path2od):
    for k, w in enumerate(path2od):
        bmin = b[w]
        if c[k] < bmin:
            b[w] = c[k]
    return b    
    

for NETWORK in ['sf']:
    
    
    
    if NETWORK == 'sf':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    
    with open('result_cndp/net_' + NETWORK + '.pkl', 'rb') as f:
        net = pickle.load(f)
        
   
    instance = Instance(NETWORK)
        
    Capacity = Instance.Capacity
    Length = Instance.Length
    Fftime = Instance.Fftime
    Demand = Instance.Demand
    Link = Instance.Link
    
    demand = Instance.demand.to(device)
    cap_original = Instance.cap_original.to(device)
    tfree = Instance.tfree.to(device)
    enh_link = Instance.enh_link
    cash = Instance.cash.to(device)
    eta = Instance.eta
    ratio = Instance.ratio
    # epsilon = Instance.epsilon * 10
    epsilon = Instance.epsilon
    xi = Instance.xi
    eps_eva = Instance.eps_eva
    
    if NETWORK == 'sf':
        r = 5        
        alpha = 2
        beta = 10
                
        pro_de = True
        truncation = False
        gap_freq = 1

    if NETWORK == 'bar':
        r = 2
        alpha = 0.1
        beta = 20
        
        pro_de = True
        truncation = False
        gap_freq = 1
                
        
    if NETWORK == 'cs':
        r = 1
        alpha = 0.05
        beta = 10
        
        
        pro_de = True
        truncation = False
        gap_freq = 20
        
    if pro_de:
        epsilon *= 10
        
    
    cap_add = torch.zeros(len(cash), dtype=torch.double).to(device)

    cap = cap_original * 1.0
    cap[enh_link] += cap_add   
    Capacity = dict(zip(Link, [float(c) for c in cap]))
    
    
    path_edge = net.path_edge.to(device)
    path_demand = net.path_demand.to(device)
    path_number = net.path_number

    
    path_number = path_edge.size()[1]
    q = path_demand.t() @ demand
    
    
    cap_add.requires_grad_()
    iter_num = 0
    gap_old = 1
    
    
    time_traj = list()
    solution_traj = list()
    cap_old = cap_add * 1.0
    
    inverting_time = 0
    we_time = 0
    
    accurancy_increase = 0

    
    tic = tm.time()
    while iter_num <= 1000:
        

        
        cap = cap_original * 1.0
        cap[enh_link] += cap_add   
            
        
        p_0 = torch.ones(path_number, dtype=torch.double).to(device)
        p_0 /= path_demand.t() @ (path_demand @ p_0)
        p = p_0 * 1.0
        
        kk = 0
        gap = torch.inf
        
        
        if truncation:
            with torch.no_grad():
                kkk = 0
                while True:
                    
                    f = q * p
                    x = path_edge @ f
                    t = tfree * (1 + 0.15 * (x / cap) ** 4)
                    c = path_edge.t() @ t
                    p *= torch.exp(-r * c)
                    p /= path_demand.t() @ (path_demand @ p)
                    
                    if kkk % gap_freq == 0:
                        b = torch.zeros(len(demand), dtype=torch.double) + torch.inf
                        path2od = np.array(net.path2od)
                        b = update_b(c.detach().cpu().numpy(), b.numpy(), path2od)    
                        b = torch.tensor(b, dtype=torch.double).to(device)                   
                        gt = torch.dot(b, demand)
                        tt = torch.dot(c, f)
                        gap = (tt - gt) / tt
                    else:
                        gap = torch.inf
                    # print(gap)
                    if gap < epsilon * 10:
                        break
                    kkk += 1
    
            p[p < 1e-20] = 0
        while True:
            f = q * p
            x = path_edge @ f
            t = tfree * (1 + 0.15 * (x / cap) ** 4)
            c = path_edge.t() @ t
            p *= torch.exp(-r * c)
            p /= path_demand.t() @ (path_demand @ p)
            p[p < 1e-20] = 0
            
            with torch.no_grad():
                if kk % gap_freq == 0:
                    b = torch.zeros(len(demand), dtype=torch.double) + torch.inf
                    path2od = np.array(net.path2od)
                    b = update_b(c.detach().cpu().numpy(), b.numpy(), path2od)    
                    b = torch.tensor(b, dtype=torch.double).to(device)                   
                    gt = torch.dot(b, demand)
                    tt = torch.dot(c, f)
                    gap = (tt - gt) / tt
                    # print(kk, gap)
                else:
                    gap = torch.inf
                
                # if iter_num == 0:
                #     if kk >= 1000:
                #         break
                # else:
                if gap < epsilon:
                    break
                gap_old = gap
                kk += 1
        
        f = q * p
        x = path_edge @ f
        t = tfree * (1 + 0.15 * (x / cap) ** 4)
        
        TT = torch.dot(x, t) 
        objective = TT + eta * torch.dot(cash, cap_add ** 2)
        
        tic2 = tm.time()
        objective.backward()
        inverting_time += tm.time() - tic2
    

        with torch.no_grad():
            
            # cap_add -= alpha * cap_add.grad
            cap_add -= alpha * beta / (iter_num + beta) * cap_add.grad
            
            cap_add.clamp_(0)
            cap_add.grad.zero_()
            
        change = float(torch.norm(cap_add - cap_old, torch.inf))
        
        # torch.cuda.empty_cache()
        
        print(iter_num, kk, net.path_number, objective, change)
        # print(iter_num, alpha, net.path_number, obj, change)
        with torch.no_grad():
            cap_old = cap_add * 1.0
        
        if pro_de:
            if change < 10 * xi and accurancy_increase == 0:
                epsilon /= 2
                accurancy_increase += 1
            if change < 2 * xi and accurancy_increase == 1:
                epsilon /= 5
                accurancy_increase += 1  
        if change < xi:
            break
            
        iter_num += 1
        
        
        with torch.no_grad():
            
            
            update = False
            # update = True
            if iter_num < 10 and iter_num % 1 == 0:
                update = True
            if iter_num >= 10 and iter_num < 50 and  iter_num % 5 == 0:
                update = True
            if iter_num >= 50 and iter_num % 25 == 0:
                update = True
                
            if update:
                link_time = tfree * (1 + 0.15 * (x / cap) ** 4)   
                # link_time += 0.01 * torch.randn_like(link_time)
                net.path_update(dict(zip(Link, [float(t) for t in link_time])))                
                path_edge = net.path_edge.to(device)    
                path_demand = net.path_demand.to(device)    
                path_number = net.path_number
                q = path_demand.t() @ demand
                
        solution_traj.append(cap_add.detach().cpu() * 1.0)
        time_traj.append(tm.time() - tic)

    toc = tm.time()
    
    #solution evaluation
    cap = cap_original * 1.0
    with torch.no_grad():
        cap[enh_link] += cap_add 
        Capacity = dict(zip(Link, [float(c) for c in cap]))
        net.solve_ue(Demand, Fftime, Capacity, 1000, eps_eva, warm_start=False)
        x = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double).to(device)
        t = tfree * (1 + 0.15 * (x / cap) ** 4)
        TT = torch.dot(x, t)
        obj = TT + eta * torch.dot(cash, cap_add ** ratio)
        
        print("objective: ", obj)
        
    
    Result = dict()
    Result['solution'] = cap_add.detach().cpu()
    Result['obj'] = obj.cpu()
    Result['time_traj'] = time_traj
    Result['solution_traj'] = solution_traj
    Result['time'] = toc - tic
    Result['inverting_time'] = inverting_time
    
    with open('result_cndp/' + 'dolmd_' + NETWORK + '.pkl', 'wb') as f:
        pickle.dump(Result, f)
                