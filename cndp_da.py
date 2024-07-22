"""
Created on Thu Feb 22 12:25:46 2024

@author: Jiayang Li
"""
import torch
import pickle
import numpy as np
import time as tm
from cndp_load import Instance
from scipy.optimize import dual_annealing



def cost_function(cap_add):
    global best_solution
    global best_value
    global n_ue
    global tic
    global time_traj
    
    cap = cap_original * 1.0
    with torch.no_grad():
        cap[enh_link] += cap_add 
    Capacity = dict(zip(Link, [float(c) for c in cap]))
    net.solve_ue(Demand, Fftime, Capacity, 1000, eps_eva, warm_start=True)
    x = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double)
    t = tfree * (1 + 0.15 * (x / cap) ** 4)
    TT = np.dot(x, t)
    obj = TT + eta * np.dot(cash, cap_add ** ratio)
    
    
    if n_ue == 0:
        tic = tm.time()
        best_solution = [cap_add]
        best_value = [obj]
        time_traj = [0]
    elif obj < best_value[-1]:
        best_solution.append(cap_add)
        best_value.append(obj)
        print(obj)

        time_traj.append(tm.time() - tic)
    
    n_ue += 1
    
    if n_ue > N_max + 1:
        raise Exception("Termination condition met")

    return obj




for NETWORK in ['sf']:
       
    instance = Instance(NETWORK)
        
    Capacity = Instance.Capacity
    Length = Instance.Length
    Fftime = Instance.Fftime
    Demand = Instance.Demand
    Link = Instance.Link
    
    demand = Instance.demand
    cap_original = Instance.cap_original
    tfree = Instance.tfree
    enh_link = Instance.enh_link
    cash = Instance.cash
    eta = Instance.eta
    ratio = Instance.ratio
    # epsilon = Instance.epsilon * 10
    epsilon = Instance.epsilon
    xi = Instance.xi
    eps_eva = Instance.eps_eva
    
    
    non_link = [i for i in range(len(tfree)) if i not in enh_link]

    dtime = lambda x: 0.6 * tfree * x ** 3 / cap_original ** 4
    
    N_trial = 1
    if NETWORK == 'sf':
        N_max = 100000
        bounds = [(0, 10) for _ in range(len(enh_link))]
    # if NETWORK == 'bar':
    #     N_max = 1000
    #     bounds = [(0, 4) for _ in range(len(toll_link))]
    # if NETWORK == 'cs':
    #     N_max = 2000
    #     bounds = [(0, 5) for _ in range(len(toll_link))]

    
    with open('result_cndp/net_' + NETWORK + '.pkl', 'rb') as f:
        net = pickle.load(f)
        
    for i in range(N_trial):
        
        np.random.seed(i)
        # minimizer_kwargs = {'method': 'Powell'}
        n_ue = 0
        
        result = dual_annealing(cost_function, bounds=bounds, maxfun=N_max, no_local_search=False)
        # try:
        #     result = dual_annealing(cost_function, bounds=bounds, maxfun=N_max, no_local_search=True)
        # except Exception as e:
        #     print(str(e))
    
    
        
        # result.time = toc - tic
        
        print('finish')
      
        cap_add = result.x
        cap = cap_original * 1.0
        with torch.no_grad():
            cap[enh_link] += cap_add
        Capacity = dict(zip(Link, [float(c) for c in cap]))
        net.solve_ue(Demand, Fftime, Capacity, 1000, eps_eva, warm_start=True)
        x = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double)
        t = tfree * (1 + 0.15 * (x / cap) ** 4)
        TT = np.dot(x, t)
        obj = TT + eta * np.dot(cash, cap_add ** ratio)
        # result.obj = obj
        print("objective: ", obj)
        print(result.fun)
        
        Result = dict()
        Result['obj'] = best_value[-1]
        Result['solution'] = best_solution[-1]
        
        Result['time_traj'] = time_traj
        Result['solution_traj'] = best_solution
        Result['obj_traj'] = best_value
            
            
        with open('result_cndp/' + 'da' + str(i) +  '_' + NETWORK + '.pkl', 'wb') as f:
            pickle.dump(Result, f)
