# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:25:46 2024

@author: Jiayang Li
"""
# import sympy
import scipy
import torch
import pickle
import time as tm
import numpy as np
from cndp_load import Instance
from torch.autograd.functional import jacobian


def find_linearly_independent_columns(A):
    
    Q, R, P = scipy.linalg.qr(A, pivoting=True)
    
    independent_cols = np.abs(np.diag(R)) > 1e-10
    rank = np.sum(independent_cols)


    independent_columns_indices = P[:rank]
    return independent_columns_indices


for NETWORK in ['bar']:
    
    if NETWORK == 'sf' or 'bar':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    
    with open('result_cndp/net_' + NETWORK + '.pkl', 'rb') as f:
        net = pickle.load(f)
   
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
    epsilon = Instance.epsilon
    xi = Instance.xi
    eps_eva = Instance.eps_eva
    
    Numberoflink = len(Link)
    NumberofOD = len(net.pathflow)
    
    non_link = [i for i in range(len(tfree)) if i not in enh_link]
    
    
    if NETWORK == 'sf':
        alpha = 1
        beta = 10
                
        
    if NETWORK == 'bar':
        # alpha = 0.02
        # beta = 10
        
        alpha = 0.1
        beta = 20
        
    if NETWORK == 'cs':
        alpha = 0.05
        beta = 20


    cap_add = torch.zeros(len(tfree), dtype=torch.double)

    # cap_add = torch.tensor(Result['solution'], dtype=torch.double)

    # cap_add = torch.tensor([5.2860, 5.3097, 1.8356, 1.8078, 2.7194, 2.7553, 3.2275, 3.2968, 4.8218, 4.8024], dtype=torch.double)
    cap_old = cap_add * 1.0
    cap_add.requires_grad_()
    
    tic = tm.time()
    
    time_traj = list()
    solution_traj = list()
    inverting_time = 0
    iter_num = 0
        
    first = True
    A_old = None
    while True:
        with torch.no_grad():
            cap = cap_original + cap_add
            Capacity = dict(zip(Link, [float(c) for c in cap]))
        
        net.solve_ue(Demand, Fftime, Capacity, 1000, epsilon, warm_start=True)
        
        
        net.path_enumeration()
        net.generate_sparse_matrix()
        
        
        path_edge = net.path_edge.to_dense()
        path_demand = net.path_demand.to_dense()
        path_number = net.path_number
        
        
        tic2 = tm.time()
        """Finding a maximum set of linearly independent columns"""
        
        q = path_demand.T @ demand
        x = torch.tensor(list(net.flow.values()), dtype=torch.double)
        
        A_eq = torch.cat([path_edge, path_demand]).cpu().numpy()
        
        # print(A_eq.size())
        if first:
            not_same = True
        elif A_eq.shape != A_old.shape:
            not_same = True
        elif np.array_equal(A_eq, A_old):
            not_same = False
        else:
            not_same = True

     
        if not_same:
            obj = torch.zeros(path_number, dtype=torch.double)
            b_eq = torch.cat([x, demand.cpu()])
            bounds = (0, None)
            res = scipy.optimize.linprog(obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            p = torch.tensor(res.x, dtype=torch.double)
            inds = p > 0
            n_i = int(torch.sum(p > 0))
            
            # inds = find_linearly_independent_columns(A_eq)
            # n_i = len(inds)
        first = False
        A_old = A_eq * 1.0
        

        print('start')
        tic_inner = tm.time()
        """Performing sensivity analysis"""
        path_demand_i = path_demand[:, inds].to(device)
        path_edge_i = path_edge[:, inds].to(device)
        

        
        
        u_x = 0.15 * tfree * 4 * (x / cap) ** 3 / cap
        # u_cap = -0.15 * tfree * 4 * (x / cap) ** 3 * x / cap ** 2
        # u_cap = u_cap[[enh_link]]
        c_f = torch.mm(path_edge_i.t(), torch.diag(u_x).to(device))
        c_f = torch.mm(c_f, path_edge_i)
        
        
        
        J_up = torch.cat([c_f, -path_demand_i.t()], 1)
        J_down = torch.cat([path_demand_i, torch.zeros(len(demand), len(demand)).to(device)], 1)
        J = torch.cat([J_up, J_down], 0)
        
        def func_c_i(cap_var):
            cap_new = cap_original + cap_var
            t = tfree * (1 + 0.15 * (x / cap_new) ** 4)
            c_i = path_edge_i.t() @ t
            return c_i
        
        c_cap = jacobian(func_c_i, cap_add.detach())
            
            
            
        # c_cap = torch.mm(path_edge_i[enh_link, :].t(), torch.diag(u_cap).to(device))
        Right = torch.cat([-c_cap, torch.zeros(len(demand), len(tfree)).to(device)], 0)
        
        S = torch.tensor(np.linalg.solve(J, Right))
        f_cap = S[:n_i, :]
        
        # f_cap = torch.mm(invJ, Right)[:n_i, :]
        x_cap = torch.mm(path_edge_i, f_cap)
        
        cap_new = cap_original + cap_add
        x_vir = x_cap @ cap_add - (x_cap @ cap_add - x).detach()
        t_vir = tfree * (1 + 0.15 * (x_vir / cap_new) ** 4)
        TT = torch.dot(x_vir, t_vir) 
        objective = TT + eta * torch.dot(cash, cap_add[enh_link] ** 2)
        objective.backward()
        print(tm.time() - tic_inner)
        inverting_time += tm.time() - tic2
        
        
        print(tm.time() - tic2)
        
        # print(cap_add.grad)
        
        with torch.no_grad():
        


            cap_add -= alpha * beta / (iter_num + beta) * cap_add.grad
            # cap_add -= alpha * cap_add.grad
            cap_add.clamp_(0)
            cap_add.grad.zero_()
            cap_add[non_link] = 0
            
        change = float(torch.norm(cap_add - cap_old, torch.inf))
        print(iter_num, alpha, objective, change)

        if change < xi:
            break
        
        if NETWORK == 'bar' and iter_num > 10 and change > 1:
            break
        cap_old = cap_add.detach() * 1.0
        
        solution_traj.append(cap_add[enh_link].detach())
        time_traj.append(tm.time() - tic)
            
        iter_num += 1
        
        # if iter_num >= 50:
        #     break
    
    toc = tm.time()
    
    #solution evaluation
    
    with torch.no_grad():
        cap = cap_original + cap_add 
        Capacity = dict(zip(Link, [float(c) for c in cap]))
        net.solve_ue(Demand, Fftime, Capacity, 1000, eps_eva, warm_start=True)
        x = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double).to(device)
        t = tfree * (1 + 0.15 * (x / cap) ** 4)
        TT = torch.dot(x, t)
        obj = TT + eta * torch.dot(cash, cap_add[enh_link] ** ratio)
        
        print("objective: ", obj)
    
    Result = dict()
    Result['solution'] = cap_add[enh_link].detach()
    Result['obj'] = obj
    Result['time_traj'] = time_traj
    Result['solution_traj'] = solution_traj
    Result['time'] = toc - tic
    Result['inverting_time'] = inverting_time
    
    with open('result_cndp/' + 'sab_' + NETWORK + '.pkl', 'wb') as f:
        pickle.dump(Result, f)
        

