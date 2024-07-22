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
    

for NETWORK in ['cs']:
    
    
    if NETWORK == 'hearn1' or NETWORK ==  'hearn2' or NETWORK == 'sf':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
        
        
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
    
    
        
    Results = dict()
    
    
    
    N = 1
    
    initial_list = list()
    
    # toll_add = torch.tensor([0.0390, 0.0390, 0.0210, 0.0600, 0.0600, 0.0450, 0.0550, 0.0620, 0.0210,
    #                          0.0320, 0.0370, 0.0360, 0.0310, 0.0000, 0.0610, 0.0000, 0.0530, 0.0440], dtype=torch.double).to(device)
    path_to_pickle = 'result_sctp/initialization_' + NETWORK + '.pkl'
    with open(path_to_pickle, 'rb') as f:
        initializations = pickle.load(f)
        
        
    for n_ini, initialization in enumerate(initializations):
        
        if n_ini in [0, 2, 3]:
            continue
        
        toll_add = initialization[0].to(device)
        toll = torch.zeros(len(tfree), dtype=torch.double).to(device)
        toll[toll_link] = toll_add
        
        with open('result_sctp/net_' + NETWORK + '.pkl', 'rb') as f:
            net = pickle.load(f)
       
        if NETWORK == 'hearn1' or NETWORK == 'hearn1':
            r = 0.05
            alpha = 0.1
            beta = 20
            gap_freq = 1
        if NETWORK == 'sf':
            r = 10
            alpha = 0.01
            beta = 10
            gap_freq = 1
            
            T = 40
            
        if NETWORK == 'bar':
            r = 2.5
            alpha = 0.01
            beta = 10
            gap_freq = 10
            T = 20
            
            xi *= 5
            
            
        if NETWORK == 'cs':
            r = 1.5
            
            r_pre = 1
            alpha = 0.0075
            
            beta = 20
            
            if n_ini == 1:
                beta = 40
            gap_freq = 20
            
            T = 20
            
        
        
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
        
        # toll = torch.zeros_like(tfree)

        # path_edge = net.path_edge.to(device)
        # path_demand = net.path_demand.to(device)
        # path_number = net.path_number
    
        
        # path_number = path_edge.size()[1]
        # q = path_demand.t() @ demand
        
        
        toll_add.requires_grad_()
        iter_num = 0
        gap_old = 1
        
        
        toll_old = toll_add * 1.0
        
        net.generate_flow()
        f = net.f.to(device)
        p0 = f / q
        
        Result = dict()
        tic = tm.time()
        time_traj = list()
        solution_traj = list()
        
        while iter_num <= 1000:
            toll = torch.zeros_like(tfree)
            toll[toll_link] = toll_add
            
            if iter_num % gap_freq == 0:
                with torch.no_grad():
                    f = q * p0
                    x = path_edge @ f
                    t = tfree * (1 + 0.15 * (x / cap) ** 4)
                    u = t + toll
                    c = path_edge.t() @ u
                    
                    b = torch.zeros_like(demand) + torch.inf
                    path2od = np.array(net.path2od)
                    b = update_b(c.detach().cpu().numpy(), b.cpu().numpy(), path2od)    
                    b = torch.tensor(b, dtype=torch.double).to(device)                   
                    gt = torch.dot(b, demand)
                    tt = torch.dot(c, f)
                    gap = (tt - gt) / tt
            else:
                gap = torch.inf
    
            p = p0 * 1.0
            for kkk in range(T):
                # print(kkk)
                f = q * p
                x = path_edge @ f
                t = tfree * (1 + 0.15 * (x / cap) ** 4)
                u = t + toll
                c = path_edge.t() @ u
                p *= torch.exp(-r * c)
                p /= path_demand.t() @ (path_demand @ p)
    
            f = q * p
            x = path_edge @ f
            t = tfree * (1 + 0.15 * (x / cap) ** 4)
            
            TT = torch.dot(x, t) 
            objective = TT
            objective.backward()
        
            with torch.no_grad():
                # print(cap_add.grad)
                
                toll_add -= alpha * beta / (iter_num + beta) * toll_add.grad
                
                toll_add.clamp_(0)
                toll_add.grad.zero_()
                
                
            change = float(torch.norm(toll_add - toll_old, torch.inf))
            print(n_ini, iter_num, net.path_number, float((TT - TT_so) / (TT_ue - TT_so)), float(change), float(gap))
            with torch.no_grad():
                toll_old = toll_add * 1.0
            if change < xi and gap < epsilon:
                break
            
            if gap > epsilon:
                with torch.no_grad():
                    p0 = p * 1.0
            
            
            with torch.no_grad():
                
                toll[toll_link] = toll_add
                
                update = False
                if iter_num < 10 and iter_num % 1 == 0:
                    update = True
                if iter_num >= 10 and iter_num < 50 and  iter_num % 5 == 0:
                    update = True
                if iter_num >= 50 and iter_num % 25 == 0:
                    update = True
                    
                # if NETWORK == 'cs' and n_ini == 1 and iter_num < 20:
                #     update = True
        # 
                if update:
                    f = p0 * q
                    net.load_flow(f)
                    
                    link_time = tfree * (1 + 0.15 * (x / cap) ** 4) + toll         
                    net.path_update(dict(zip(Link, [float(t) for t in link_time])))                
                    path_edge = net.path_edge.to(device)    
                    path_demand = net.path_demand.to(device)    
                    path_number = net.path_number
                    q = path_demand.t() @ demand
                    
                    net.generate_flow()
                    f = net.f.to(device)
                    p0 = f / q
                # if iter_num > 10:
                #     p0[p0 < 1e-20] = 0
            iter_num += 1
            
            solution_traj.append(toll_add.detach().cpu())
            time_traj.append(tm.time() - tic)
        toc = tm.time()
                                 
                    
        with torch.no_grad():
            toll = torch.zeros_like(tfree)
            toll[toll_link] = toll_add
            
        Toll = dict()
        for kk, ii in enumerate(net.Link):
            Toll[ii] = float(toll[kk])
        
        net.solve_ue(Demand, Fftime, Capacity, 1000, eps_eva, warm_start=False, toll=Toll)
        x = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double).to(device)
        t = tfree * (1 + 0.15 * (x / cap) ** 4)
        TT = torch.dot(x, t)
        obj = (TT - TT_so) / (TT_ue - TT_so)
        print(obj)
        
        
        Result['solution'] = toll_add.detach().cpu()
        Result['obj'] = obj.cpu()
        Result['time_traj'] = time_traj
        Result['solution_traj'] = solution_traj
        Result['time'] = toc - tic
        
        
        # A = np.zeros((1, len(toll_link) + 1), dtype=np.double)

            
        # A[0, 0] = obj.detach()
        # A[0, 1:] = toll_add.detach()
                        
  
        with open('result_sctp/' + 'silmd' + str(T) + '_' + str(n_ini) + NETWORK + '.pkl', 'wb') as f:
            pickle.dump(Result, f)
        
    


# for ini_num, toll_add in enumerate(initial_list):
#     print(Results[ini_num]['obj'])
                
