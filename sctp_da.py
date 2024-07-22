"""
Created on Thu Feb 22 12:25:46 2024

@author: Jiayang Li
"""
import torch
import pickle
import numpy as np
import time as tm
from sctp_load import Instance
from scipy.optimize import dual_annealing



def total_travel_time(x):
    t = tfree * (1 + 0.15 * (x / cap) ** 4)
    return torch.dot(x, t)


def cost_function(toll_add):
    global best_solution
    global best_value
    global n_ue
    global tic
    global time_traj
    
    toll = np.zeros(len(tfree), dtype=np.double)
    toll[toll_link] = toll_add
    Toll = dict()
    for kk, ii in enumerate(net.Link):
        Toll[ii] = float(toll[kk])
    
    net.solve_ue(Demand, Fftime, Capacity, 10000, epsilon, warm_start=True, toll=Toll)
    x = np.array([net.flow[ii] for ii in net.Link], dtype=np.double)
    t = tfree * (1 + 0.15 * (x / cap) ** 4)
    TT = np.dot(x, t)
    
    
    if n_ue == 0:
        tic = tm.time()
        best_solution = [toll_add]
        best_value = [TT]
        time_traj = [0]
    elif TT < best_value[-1]:
        best_solution.append(toll_add)
        best_value.append(TT)
        print(TT)

        time_traj.append(tm.time() - tic)
    
    n_ue += 1
    
    if n_ue > N_max + 1:
        raise Exception("Termination condition met")

    return TT




for NETWORK in ['sf']:
       
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
    
    N_trial = 1
    if NETWORK == 'sf':
        N_max =  100000
        bounds = [(0, 0.1) for _ in range(len(toll_link))]
    if NETWORK == 'bar':
        N_max = 1000
        bounds = [(0, 4) for _ in range(len(toll_link))]
    if NETWORK == 'cs':
        N_max = 2000
        bounds = [(0, 5) for _ in range(len(toll_link))]

    
    with open('result_sctp/net_' + NETWORK + '.pkl', 'rb') as f:
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
      
        
        toll = np.zeros(len(tfree), dtype=np.double)
        toll[toll_link] = result.x * 1.0
        Toll = dict()
        for kk, ii in enumerate(net.Link):
            Toll[ii] = float(toll[kk])
                             
        net.solve_ue(Demand, Fftime, Capacity, 10000, eps_eva, warm_start=True, toll=Toll)
        x = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double)
        obj = np.dot(x, toll)
        # result.obj = obj
        print("objective: ", obj)
        print(result.fun)
        
        Result = dict()
        Result['obj'] = best_value[-1]
        Result['solution'] = best_solution[-1]
        
        Result['time_traj'] = time_traj
        Result['solution_traj'] = best_solution
        Result['obj_traj'] = best_value
            
            
        with open('result_sctp/' + 'da' + str(i) +  '_' + NETWORK + '.pkl', 'wb') as f:
            pickle.dump(Result, f)
