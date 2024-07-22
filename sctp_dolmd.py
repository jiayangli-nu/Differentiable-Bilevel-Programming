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
from sctp_load import Instance


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


for NETWORK in ['sf']:
    
    if NETWORK == 'sf':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
    
    instance = Instance(NETWORK)
        
    Capacity = Instance.Capacity
    Length = Instance.Length
    Fftime = Instance.Fftime
    Demand = Instance.Demand
    Link = Instance.Link
    
    demand = Instance.demand.to(device)
    cap = Instance.cap.to(device)
    tfree = Instance.tfree.to(device)
    toll_link = Instance.toll_link
    epsilon = Instance.epsilon
    xi = Instance.xi
    eps_eva = Instance.eps_eva
    
    torch.manual_seed(0)
    
    x_ue = torch.load('result_sctp/ue_' + NETWORK + '.pt').to(device)
    x_so = torch.load('result_sctp/so_' + NETWORK + '.pt').to(device)
    TT_ue = total_travel_time(x_ue)
    TT_so = total_travel_time(x_so)
    dtime = lambda x: 0.6 * tfree * x ** 3 / cap ** 4
    toll_so = dtime(x_so) * x_so
    
    non_link = [i for i in range(len(tfree)) if i not in toll_link]
    
    
    
    if NETWORK == 'hearn1' or NETWORK == 'hearn2':
        r = 0.025
        # alpha = 0.05
        alpha = 0.01
        beta = 20
        truncation = False
        pro_de = True
        gap_freq = 1
    
    if NETWORK == 'sf':
        r = 10
        
        alpha = 0.005
        beta = 10
        truncation = False
        pro_de = True
        gap_freq = 1
        
    if NETWORK == 'bar':
        # r = 2
        r = 1.5
        alpha = 0.01
        beta = 20
        truncation = False
        pro_de = True
        
        xi *= 5
        
        gap_freq = 10
        
    if NETWORK == 'cs':
        r = 1
        
        r_pre = 1
        alpha = 0.05
        
        beta = 10
        pro_de = True
        truncation = False
        
        gap_freq = 20
        
    if pro_de:
        epsilon *= 10


    path_to_pickle = 'result_sctp/initialization_' + NETWORK + '.pkl'
    with open(path_to_pickle, 'rb') as f:
        initializations = pickle.load(f)
        
        
    for n_ini, initialization in enumerate(initializations):
        

        
        
        if NETWORK == 'sf':
            if n_ini == 0:
                alpha = 0.005
                beta = 5
                
            if n_ini == 1:
                alpha = 0.01
                beta = 1
                
            if n_ini == 2:
                alpha = 0.005
                beta = 5
                
            if n_ini == 3:
                alpha = 0.005
                beta = 5
                
                
        if NETWORK == 'bar':
            if n_ini == 0:
                alpha = 0.005
                beta = 20
                r = 1
                
            if n_ini == 1:
                alpha = 0.025
                beta = 20
                r = 1
                
            if n_ini == 2:
                alpha = 0.005
                beta = 20
                r = 1.5
            if n_ini == 3:
                alpha = 0.01
                beta = 20
                r = 1.5
                
        if NETWORK == 'cs':
            if n_ini == 0:
                alpha = 0.005
                beta = 20
                r = 1.5
                
                r_pre = 1
                
            if n_ini == 1:
                alpha = 0.005
                beta = 40
                r = 1.5
                
                r_pre = 1
                
            if n_ini == 2:
                alpha = 0.0025
                beta = 20
                r = 1.5
                
                r_pre = 1
                
            if n_ini == 3:
                alpha = 0.005
                beta = 20
                r = 1.5
                
                r_pre = 1

        
        with open('result_sctp/net_' + NETWORK + '.pkl', 'rb') as f:
            net = pickle.load(f)
      
        
        toll_add = initialization[0].to(device)
        
        
        iter_num = 0
        gap_old = 1
        
        tic = tm.time()
        time_traj = list()
        solution_traj = list()
        toll_old = toll_add * 1.0
        
        inverting_time = 0
        we_time = 0
        
        toll = torch.zeros_like(tfree).to(device)
        toll[toll_link] = toll_add
        with torch.no_grad():
            Toll = dict()
            for kk, ii in enumerate(net.Link):
                Toll[ii] = float(toll[kk])
        
        net.solve_ue(Demand, Fftime, Capacity, 1000, epsilon, warm_start=True, toll=Toll)
        x = torch.tensor(list(net.flow.values()), dtype=torch.double).to(device)
        net.path_enumeration()
        net.generate_sparse_matrix()
        path_edge = net.path_edge.to(device)
        path_demand = net.path_demand.to(device)
        path_number = net.path_number

        
        path_number = path_edge.size()[1]
        q = path_demand.t() @ demand
        
        accurancy_increase = 0
        toll_add.requires_grad_()
        while iter_num <= 1000:

                
            toll = torch.zeros_like(tfree).to(device)
            toll[toll_link] = toll_add  
                
            
            p_0 = torch.ones(path_number, dtype=torch.double).to(device)
            p_0 /= path_demand.t() @ (path_demand @ p_0)
            p = p_0 * 1.0
            
            kk = 0
            gap = torch.inf
            
            
            if truncation:
                with torch.no_grad():
                    while True:
                        f = q * p
                        x = path_edge @ f
                        t = tfree * (1 + 0.15 * (x / cap) ** 4)
                        u = t + toll
                        c = path_edge.t() @ u
                        p *= torch.exp(-r_pre * c)
                        p /= path_demand.t() @ (path_demand @ p)
                        
                        
                        b = torch.zeros(len(demand), dtype=torch.double) + torch.inf
                        path2od = np.array(net.path2od)
                        b = update_b(c.detach().cpu().numpy(), b.numpy(), path2od)    
                        b = torch.tensor(b, dtype=torch.double).to(device)                   
                        gt = torch.dot(b, demand)
                        tt = torch.dot(c, f)
                        gap = (tt - gt) / tt
                        # print(gap)
                        if gap < epsilon:
                            break 
                    p[p < 1e-20] = 0
                    
            while True:
            
                f = q * p
                x = path_edge @ f
                t = tfree * (1 + 0.15 * (x / cap) ** 4)
                u = t + toll
                c = path_edge.t() @ u
                
                p *= torch.exp(-r * c)
                p /= path_demand.t() @ (path_demand @ p)
       
                
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
                    if not truncation:
                        if gap < epsilon and gap > 0:
                            break
                    else:
                        if iter_num == 0:
                            if kk >= 400:
                                break
                        else:
                            if kk >= 150:
                                break
                                

                    kk += 1

            
            
            f = q * p
            x = path_edge @ f
            t = tfree * (1 + 0.15 * (x / cap) ** 4)
            
            TT = torch.dot(x, t) 
            objective = TT
  
            
            
            tic2 = tm.time()
            objective.backward()
            inverting_time += tm.time() - tic2
        
            with torch.no_grad():
                # print(cap_add.grad)
                
                toll_add -= alpha * beta / (iter_num + beta) * toll_add.grad
                
                toll_add.clamp_(0)
                toll_add.grad.zero_()
                
            change = float(torch.norm(toll_add - toll_old, torch.inf))
            print(n_ini, iter_num, r, net.path_number, kk, float((TT - TT_so) / (TT_ue - TT_so)), change)
            with torch.no_grad():
                toll_old = toll_add * 1.0
                
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
                if iter_num < 10 and iter_num % 1 == 0:
                    update = True
                if iter_num >= 10 and iter_num < 50 and  iter_num % 5 == 0:
                    update = True
                if iter_num >= 50 and iter_num % 25 == 0:
                    update = True
                    
                # if NETWORK == 'cs' and n_ini == 1 and iter_num < 20:
                #     update = True
                    
                if update:
                    link_time = tfree * (1 + 0.15 * (x / cap) ** 4) + toll
                    # link_time += 0.01 * torch.randn_like(link_time)
                    net.path_update(dict(zip(Link, [float(t) for t in link_time])))                
                    path_edge = net.path_edge.to(device)    
                    path_demand = net.path_demand.to(device)    
                    path_number = net.path_number
                    q = path_demand.t() @ demand
                    
            solution_traj.append(toll_add.detach().cpu())
            time_traj.append(tm.time() - tic)
    
        toc = tm.time()
        
        #solution evaluation
        with torch.no_grad():
            toll = torch.zeros_like(tfree).to(device)
            toll[toll_link] = toll_add
            
        Toll = dict()
        for kk, ii in enumerate(net.Link):
            Toll[ii] = float(toll[kk])
        
        net.solve_ue(Demand, Fftime, Capacity, 1000, eps_eva, warm_start=False, toll=Toll)
        x = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double).to(device)
        t = tfree * (1 + 0.15 * (x / cap) ** 4)
        TT = torch.dot(x, t)
        obj = (TT - TT_so) / (TT_ue - TT_so)
        # print(toll_add.detach())
        print(obj)
        
        Result = dict()
        Result['solution'] = toll_add.detach()
        Result['obj'] = obj
        Result['time_traj'] = time_traj
        Result['solution_traj'] = solution_traj
        Result['time'] = toc - tic
        
        
        
        with open('result_sctp/' + 'dolmd' + str(n_ini)  + '_' + NETWORK + '.pkl', 'wb') as f:
            pickle.dump(Result, f)
                    