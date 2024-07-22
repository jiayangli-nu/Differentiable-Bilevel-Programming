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
from scipy.optimize import minimize


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


for NETWORK in ['sf']:
    
    torch.cuda.empty_cache()
    
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
    
    
    
    torch.manual_seed(0)
    
    x_ue = torch.load('result_sctp/ue_' + NETWORK + '.pt').numpy()
    x_so = torch.load('result_sctp/so_' + NETWORK + '.pt').numpy()
    TT_ue = total_travel_time(x_ue)
    TT_so = total_travel_time(x_so)
    dtime = lambda x: 0.6 * tfree * x ** 3 / cap ** 4
    toll_so = dtime(x_so) * x_so
    
    non_link = [i for i in range(len(tfree)) if i not in toll_link]
    
    if NETWORK == 'sf':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    
    with open('result_sctp/net_' + NETWORK + '.pkl', 'rb') as f:
        net = pickle.load(f)
        
    NumberofOD = len(net.pathflow)
    
    toll_add = torch.tensor([0.0390, 0.0390, 0.0210, 0.0600, 0.0600, 0.0450, 0.0550, 0.0620, 0.0210,
                              0.0320, 0.0370, 0.0360, 0.0310, 0.0000, 0.0610, 0.0000, 0.0530, 0.0440], dtype=torch.double)
    toll = torch.zeros(len(tfree), dtype=torch.double)
    toll[toll_link] = toll_add
    Toll = dict()
    for kk, ii in enumerate(net.Link):
        Toll[ii] = float(toll_so[kk])
    net.solve_ue(Demand, Fftime, Capacity, 1000, 1e-10, warm_start=True, toll=Toll, delete=False)
    net.path_enumeration()
    net.generate_sparse_matrix()
    net.generate_flow()
    f = net.f.numpy()
    
    
    toll_so = dtime(x_so) * x_so
    Toll = dict()
    for kk, ii in enumerate(net.Link):
        Toll[ii] = float(toll_so[kk])
    net.solve_ue(Demand, Fftime, Capacity, 1000, 1e-10, warm_start=True, toll=Toll, delete=False)
    net.path_enumeration()
    net.generate_sparse_matrix()

    path_edge = net.path_edge.to_dense().numpy()
    path_demand = net.path_demand.to_dense().numpy()
    path_number = net.path_number
    
    path_number = path_edge.shape[1]
    q = path_demand.T @ demand

    
    # toll_add = torch.zeros(len(toll_link), dtype=torch.double).to(device)
    
    # toll_add = torch.tensor([0.0390, 0.0390, 0.0210, 0.0600, 0.0600, 0.0450, 0.0550, 0.0620, 0.0210,
    #                           0.0320, 0.0370, 0.0360, 0.0310, 0.0000, 0.0610, 0.0000, 0.0530, 0.0440], dtype=torch.double)
    
    toll = np.zeros(len(tfree), dtype=np.double)
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
        
        toll_add = decision[:len(toll_link)]
        toll[toll_link] = toll_add
        t = link_time(decision)
        return t + toll

    def eq_constraint(decision):
        
        f = decision[len(toll_link):]
        
        return path_demand @ f - demand
    
    
    toll_add = toll_add.numpy()
    tic = tm.time()
    
    cuts = [x_so]
    while True:
        initial_guess = np.concatenate([toll_add, f])
        
        bounds = [(0, None) for _ in range(len(initial_guess))]
    
        eq_constraints = [{'type': 'eq', 'fun': eq_constraint}]
        
        non_constraints = [{'type': 'eq', 'fun': 
                            lambda decision, cut=cut: np.dot(link_cost(decision), cut - path_edge @ decision[len(toll_link):])}
                           for cut in cuts]
            
        constraints = eq_constraints + non_constraints
                       
    
        result = minimize(obj, initial_guess, bounds=bounds, constraints=constraints)
        
        decision = result.x
        toll_add = decision[:len(toll_link)]
        print('lower bound:', result.fun)
        f_sub = decision[len(toll_link):]
        x_sub = path_edge @ f_sub
        toll = np.zeros(len(tfree), dtype=np.double)
        toll[toll_link] = toll_add
        u_sub = tfree * (1 + 0.15 * (x_sub / cap) ** 4) + toll
        cut = net.all_or_nothing(dict(zip(Link, [float(t) for t in u_sub]))).numpy()
        cuts.append(cut)
        
        
        
        toll = np.zeros(len(tfree), dtype=np.double)
        toll[toll_link] = toll_add
        # Toll = dict()
        # for kk, ii in enumerate(net.Link):
        #     Toll[ii] = float(toll[kk])
        # net.solve_ue(Demand, Fftime, Capacity, 1000, 1e-10, warm_start=True, toll=Toll, delete=False)
        # x_current = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double).numpy()
        p_0 = np.ones(path_number, dtype=np.double)
        p_0 /= path_demand.T @ (path_demand @ p_0)
        p = p_0 * 1.0
        
        kk = 0
        r = 2.5
        while True:
        
            f = q * p
            x = path_edge @ f
            t = tfree * (1 + 0.15 * (x / cap) ** 4)
            u = t + toll
            c = path_edge.T @ u
            p *= np.exp(-r * c)
            p /= path_demand.T @ (path_demand @ p)
            with torch.no_grad():
                
                b = np.zeros(len(demand), dtype=np.double) + np.inf
                path2od = np.array(net.path2od)
                b = update_b(c, b, path2od)    
                gt = np.dot(b, demand)
                tt = np.dot(c, f)
                gap = (tt - gt) / tt
                

                if gap < 1e-6 and gap > 0:
                    print('UE found')
                    break
                kk += 1
                
        
        
        TT = total_travel_time(x)
        print('upper bound:', TT)
    
        
        
    

    toc = tm.time()
    
    #solution evaluation
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
    
    Result = dict()
    Result['solution'] = toll_add.detach()
    Result['obj'] = obj
    # Result['time_traj'] = time_traj
    # Result['solution_traj'] = solution_traj
    Result['time'] = toc - tic
    
    break
    
    # with open('result_sctp/' + 'dolmd_' + NETWORK + '.pkl', 'wb') as f:
    #     pickle.dump(Result, f)
                