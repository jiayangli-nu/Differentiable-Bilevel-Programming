# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:25:46 2024

@author: Jiayang Li
"""
import torch
import pickle
import time as tm
import numpy as np
from numba import jit
from cndp_load import Instance


@jit(nopython=True)
def update_b(c, b, path2od):
    for k, w in enumerate(path2od):
        bmin = b[w]
        if c[k] < bmin:
            b[w] = c[k]
    return b    
    

for NETWORK in ['bar']:
    
    if NETWORK == 'sf':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        
    Results = dict()
    T_list = [10, 40]
    
    Results['T_list'] = T_list
    for T in T_list:
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
        epsilon = Instance.epsilon
        xi = Instance.xi
        eps_eva = Instance.eps_eva
        
        if NETWORK == 'sf':
            r = 15
            alpha = 2.5
            beta = 20
            
            gap_freq = 1
            
        if NETWORK == 'bar':
            r = 2.5
            alpha = 0.1
            beta = 5
            
            gap_freq = 1
            
        if NETWORK == 'cs':
            r = 2.5
            
            alpha = 0.02
            beta = 10
            xi = 1e-4
            
            gap_freq = 10
            
        
            
        
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
        
        
        cap_old = cap_add * 1.0
        
        net.generate_flow()
        f = net.f.to(device)
        p0 = f / q
        
        Result = dict()
        tic = tm.time()
        time_traj = list()
        solution_traj = list()
        
        while iter_num <= 1000:
            
            cap = cap_original * 1.0
            cap[enh_link] += cap_add
            
            if iter_num % gap_freq == 0:
            
                with torch.no_grad():
                    f = q * p0
                    x = path_edge @ f
                    t = tfree * (1 + 0.15 * (x / cap) ** 4)
                    c = path_edge.t() @ t
                    
                    b = torch.zeros_like(demand) + torch.inf
                    path2od = np.array(net.path2od)
                    b = update_b(c.detach().cpu().numpy(), b.cpu().numpy(), path2od)    
                    b = torch.tensor(b, dtype=torch.double).to(device)                   
                    gt = torch.dot(b, demand)
                    tt = torch.dot(c, f)
                    gap = (tt - gt) / tt
    
            p = p0 * 1.0
            for _ in range(T):
            
                f = q * p
                x = path_edge @ f
                t = tfree * (1 + 0.15 * (x / cap) ** 4)
                c = path_edge.t() @ t
                p *= torch.exp(-r * c)
                p /= path_demand.t() @ (path_demand @ p)
    
            f = q * p
            x = path_edge @ f
            t = tfree * (1 + 0.15 * (x / cap) ** 4)
            
            TT = torch.dot(x, t) 
            objective = TT + eta * torch.dot(cash, cap_add ** 2)
            objective.backward()
        
            with torch.no_grad():
                # print(cap_add.grad)
                
                cap_add -= alpha * beta / (iter_num + beta) * cap_add.grad
                
                cap_add.clamp_(0)
                cap_add.grad.zero_()
                
                
            change = float(torch.norm(cap_add - cap_old, torch.inf))
            print(iter_num, net.path_number, float(objective), float(change), float(gap))
            with torch.no_grad():
                cap_old = cap_add * 1.0
            if change < xi and gap < epsilon:
                break
            if gap > epsilon:
                with torch.no_grad():
                    p0 = p * 1.0
            
            
            with torch.no_grad():
                
                cap = cap_original * 1.0
                cap[enh_link] += cap_add
                
                update = False
                if iter_num < 10 and iter_num % 1 == 0:
                    update = True
                if iter_num >= 10 and iter_num < 50 and  iter_num % 5 == 0:
                    update = True
                if iter_num >= 50 and iter_num % 50 == 0:
                    update = True
    
        # 
                if update:
                    f = p0 * q
                    net.load_flow(f)
                    
                    link_time = tfree * (1 + 0.15 * (x / cap) ** 4)            
                    net.path_update(dict(zip(Link, [float(t) for t in link_time])))                
                    path_edge = net.path_edge.to(device)    
                    path_demand = net.path_demand.to(device)    
                    path_number = net.path_number
                    q = path_demand.t() @ demand
                    
                    net.generate_flow()
                    f = net.f.to(device)
                    p0 = f / q

            iter_num += 1
            
            solution_traj.append(cap_add.detach().cpu())
            time_traj.append(tm.time() - tic)
        toc = tm.time()
                                 
                    
        cap = cap_original * 1.0
        with torch.no_grad():
            cap[enh_link] += cap_add 
        Capacity = dict(zip(Link, [float(c) for c in cap]))
        net.solve_ue(Demand, Fftime, Capacity, 1000, eps_eva, warm_start=False)
        x = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double).to(device)
        t = tfree * (1 + 0.15 * (x / cap) ** 4)
        TT = torch.dot(x, t)
        obj = TT + eta * torch.dot(cash, cap_add ** ratio)
        print(obj)
        
        
        Result['solution'] = cap_add.detach().cpu()
        Result['obj'] = obj.cpu()
        Result['time_traj'] = time_traj
        Result['solution_traj'] = solution_traj
        Result['time'] = toc - tic
        
        Results[T] = Result
        
        print(Result['obj'])
        
        with open('result_cndp/' + 'silmd' + str(T) + '_' + NETWORK + '.pkl', 'wb') as f:
            pickle.dump(Result, f)
        
                
