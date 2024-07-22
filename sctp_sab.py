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
from sctp_load import Instance
from torch.autograd.functional import jacobian


def total_travel_time(x):
    t = tfree * (1 + 0.15 * (x / cap) ** 4)
    return torch.dot(x, t)


def find_linearly_independent_columns(A):
    
    Q, R, P = scipy.linalg.qr(A, pivoting=True)
    
    independent_cols = np.abs(np.diag(R)) > 1e-20
    rank = np.sum(independent_cols)


    independent_columns_indices = P[:rank]
    return independent_columns_indices


for NETWORK in ['hearn1', 'hearn2']:
    
    if NETWORK == 'hearn1' or NETWORK ==  'hearn2' or NETWORK == 'sf' or NETWORK ==  'bar':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    
    with open('result_sctp/net_' + NETWORK + '.pkl', 'rb') as f:
        net = pickle.load(f)
   
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
    
    Numberoflink = len(Link)
    NumberofOD = len(net.pathflow)
    
    # torch.manual_seed(0)
    
    x_ue = torch.load('result_sctp/ue_' + NETWORK + '.pt')
    x_so = torch.load('result_sctp/so_' + NETWORK + '.pt')
    TT_ue = total_travel_time(x_ue)
    TT_so = total_travel_time(x_so)
    dtime = lambda x: 0.6 * tfree * x ** 3 / cap ** 4
    toll_so = dtime(x_so) * x_so
    
    non_link = [i for i in range(len(tfree)) if i not in toll_link]
    
    if NETWORK == 'hearn1' or NETWORK == 'hearn2':
        # alpha = 0.1
        # alpha = 0.05
        # beta = 10
        alpha = 0.05
        beta = 20
        
        epsilon = 1e-4
    if NETWORK == 'sf':
        alpha = 0.001
        beta = 100
                
        
    if NETWORK == 'bar':
        alpha = 0.02
        beta = 10
        
    if NETWORK == 'cs':
        alpha = 0.05
        beta = 20



    
    path_to_pickle = 'result_sctp/initialization_' + NETWORK + '.pkl'
    with open(path_to_pickle, 'rb') as f:
        initializations = pickle.load(f)
        
        
    for n_ini, initialization in enumerate(initializations):
        # if n_ini == 0 or n_ini == 1 or n_ini == 2:
        #     continue
        
        
        
        with open('result_sctp/net_' + NETWORK + '.pkl', 'rb') as f:
            net = pickle.load(f)
        # path_edge = net.path_edge.to(device)
        # path_demand = net.path_demand.to(device)
        # path_number = net.path_number

        
        # path_number = path_edge.size()[1]
        # q = path_demand.t() @ demand
        
        toll_add = initialization[0].to(device)
        toll = torch.zeros(len(tfree), dtype=torch.double)
        toll[toll_link] = toll_add
    
       
        toll_old = toll * 1.0
        toll.requires_grad_()
        
        tic = tm.time()
        
        time_traj = list()
        solution_traj = list()
        inverting_time = 0
        iter_num = 0
        while True:
            
            with torch.no_grad():
                Toll = dict()
                for kk, ii in enumerate(net.Link):
                    Toll[ii] = float(toll[kk])
            
            net.solve_ue(Demand, Fftime, Capacity, 1000, epsilon, warm_start=True, toll=Toll)
            # net.solve_ue(Demand, Fftime, Capacity, 1000, epsilon, toll=Toll, warm_start=True)
            
            
            net.path_enumeration()
            net.generate_sparse_matrix()
            
            
            path_edge = net.path_edge.to_dense()
            path_demand = net.path_demand.to_dense()
            path_number = net.path_number
            
            
            tic2 = tm.time()
            """Finding a maximum set of linearly independent columns"""
            
            q = path_demand.T @ demand
            x = torch.tensor(list(net.flow.values()), dtype=torch.double)
            
    
            # A_eq = torch.cat([path_edge, path_demand]).cpu().numpy()
            # inds = find_linearly_independent_columns(A_eq)
            # n_i = len(inds)
            A_eq = torch.cat([path_edge, path_demand]).cpu().numpy()
            obj = torch.zeros(path_number, dtype=torch.double)
            b_eq = torch.cat([x, demand.cpu()])
            bounds = (0, None)
            res = scipy.optimize.linprog(obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            p = torch.tensor(res.x, dtype=torch.double)
            inds = p > 0
            n_i = int(torch.sum(p > 0))
            
            
            """Performing sensivity analysis"""
            path_demand_i = path_demand[:, inds].to(device)
            path_edge_i = path_edge[:, inds].to(device)
            
            
            u_x = 0.15 * tfree * 4 * (x / cap) ** 3 / cap
            # u_toll = torch.ones_like(tfree)
            # u_toll = u_toll[[toll_link]]
            c_f = torch.mm(path_edge_i.t(), torch.diag(u_x).to(device))
            c_f = torch.mm(c_f, path_edge_i)
            # c_toll = torch.mm(path_edge_i[toll_link, :].t(), torch.diag(u_toll).to(device))
            
            
            J_up = torch.cat([c_f, -path_demand_i.t()], 1)
            J_down = torch.cat([path_demand_i, torch.zeros(len(demand), len(demand)).to(device)], 1)
            J = torch.cat([J_up, J_down], 0)
            
            def func_c_i(toll_var):
                u = tfree * (1 + 0.15 * (x / cap) ** 4) + toll_var
                c_i = path_edge_i.t() @ u
                return c_i
            
            c_toll = jacobian(func_c_i, toll.detach())
                    
            Right = torch.cat([-c_toll, torch.zeros(len(demand), len(tfree)).to(device)], 0)
            
            S = torch.linalg.solve(J, Right)
            f_toll = S[:n_i, :]
            
            x_toll = torch.mm(path_edge_i, f_toll)
            
            
            x_vir = x_toll @ toll - (x_toll @ toll - x).detach()
            
            
            
            t_vir = tfree * (1 + 0.15 * (x_vir / cap) ** 4)
            TT = torch.dot(x_vir, t_vir) 
            objective = TT
            objective.backward()
            inverting_time += tm.time() - tic2
            
            with torch.no_grad():
    
                toll -= alpha * beta / (iter_num + beta) * toll.grad
                # toll -= alpha * toll.grad
                toll.clamp_(0)
                toll.grad.zero_()
                toll[non_link] = 0
                
            change = float(torch.norm(toll - toll_old, torch.inf))
            # print(iter_num, net.path_number, alpha, objective, change)
    
            if change < xi:
                break
            
            toll_old = toll.detach() * 1.0
            
            solution_traj.append(toll[toll_link].detach())
            time_traj.append(tm.time() - tic)
                
            iter_num += 1
        
        toc = tm.time()
        
        #solution evaluation
        
        with torch.no_grad():
            Toll = dict()
            for kk, ii in enumerate(net.Link):
                Toll[ii] = float(toll[kk])
                                 
            net.solve_ue(Demand, Fftime, Capacity, 1000, eps_eva, warm_start=True, toll=Toll)
            x = torch.tensor([net.flow[ii] for ii in net.Link], dtype=torch.double).to(device)
            t = tfree * (1 + 0.15 * (x / cap) ** 4)
            TT = torch.dot(x, t)
            obj = (TT - TT_so) / (TT_ue - TT_so)
            
            print("objective: ", obj)
        
        Result = dict()
        Result['solution'] = toll.detach()
        Result['obj'] = obj
        Result['time_traj'] = time_traj
        Result['solution_traj'] = solution_traj
        Result['time'] = toc - tic
        Result['inverting_time'] = inverting_time
        
        with open('result_sctp/' + 'sab' + str(n_ini) + '_' + NETWORK + '.pkl', 'wb') as f:
            pickle.dump(Result, f)
            
        
            
        # A = np.zeros((1, len(toll_link) + 1), dtype=np.double)
    
            
        # A[0, 0] = obj
        # A[0, 1:] = toll[toll_link].detach()
        # break
            
        
    
