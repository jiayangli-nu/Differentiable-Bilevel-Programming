# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:25:46 2024

@author: Jiayang Li
"""
import torch
import pickle
import time as tm
import numpy as np
from cndp_load import Instance
from scipy.optimize import minimize


for NETWORK in ['sf']:
    
    with open('result_cndp/net_' + NETWORK + '.pkl', 'rb') as f:
        net = pickle.load(f)
    
    instance = Instance(NETWORK)
        
    Capacity = Instance.Capacity
    Length = Instance.Length
    Fftime = Instance.Fftime
    Demand = Instance.Demand
    Link = Instance.Link
    
    cap_original = Instance.cap_original
    tfree = Instance.tfree
    enh_link = Instance.enh_link
    cash = Instance.cash
    eta = Instance.eta
    ratio = Instance.ratio
    epsilon = Instance.epsilon
    xi = Instance.xi
    eps_eva = Instance.eps_eva
        
    cap_original = cap_original.numpy()
    cash = cash.numpy()
    
    cap_add = np.zeros(len(enh_link), dtype=np.double)
        
    tic = tm.time()
    
    cap_old = cap_add * 1.0
    time_traj = list()
    solution_traj = list()
    
    xi = 1e-4
    
    while True:
        
        cap = cap_original * 1.0
        cap[enh_link] += cap_add 
        Capacity = dict(zip(Link, [float(c) for c in cap]))
        net.solve_so(Demand, Fftime, Capacity, 1000, epsilon, warm_start=True)
        x = np.array([net.flow[ii] for ii in net.Link], dtype=np.double)
    
        def obj_so(cap_add): 
            
            cap = cap_original * 1.0
            cap[enh_link] += cap_add 
            t = tfree.numpy() * (1 + 0.15 * (x / cap) ** 4)
            TT = np.dot(x, t) 
            obj = TT + eta * np.dot(cash, cap_add ** ratio)

            return obj
        
        initial_guess = cap_add * 1.0
        bounds = [(0, None) for _ in range(len(initial_guess))]  
        result = minimize(obj_so, initial_guess, bounds=bounds, tol=1e-5)
        cap_add = result.x
        
        change = np.linalg.norm(cap_add - cap_old, np.inf)
        if change < xi:
            break
        cap_old = cap_add * 1.0
                
        solution_traj.append(cap_add)
        time_traj.append(tm.time() - tic)
        
    toc = tm.time()
    
    # solution evaluation
    cap_add = result.x
    cap = cap_original * 1.0
    with torch.no_grad():
        cap[enh_link] += cap_add 
    Capacity = dict(zip(Link, [float(c) for c in cap]))
    net.solve_ue(Demand, Fftime, Capacity, 1000, eps_eva, warm_start=False)
    x = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double)
    t = tfree * (1 + 0.15 * (x / cap) ** 4)
    TT = np.dot(x, t)
    obj = TT + eta * np.dot(cash, cap_add ** ratio)
    print(obj)
    
    Result = dict()
    Result['solution'] = cap_add
    Result['obj'] = obj
    Result['time_traj'] = time_traj
    Result['solution_traj'] = solution_traj
    Result['time'] = toc - tic
    Result['lower-bound'] = result.fun
    
    with open('result_cndp/' + 'so_' + NETWORK + '.pkl', 'wb') as f:
        pickle.dump(Result, f)
        
