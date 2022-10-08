#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
import torch.sparse as sparse
from torch.autograd.functional import jacobian
from prettytable import PrettyTable


"""Testing CPU"""
#device = torch.device("cpu")

"""Testing GPU"""
device = torch.device("cuda:0")

N_link_list = [10, 10, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]
M = 1

ID_time = torch.zeros(len(N_link_list), M)
AD_time = torch.zeros(len(N_link_list), M)
FP_time = torch.zeros(len(N_link_list), M)
Iteration = torch.zeros(len(N_link_list), M)
for j, N_link in enumerate(N_link_list):
    if j > 0:
        torch.manual_seed(1)
    for k in range(M):
        tfree = 1 + torch.rand(int(N_link), dtype=torch.double).to(device)
        cap = 0.8 * (1 + 0.2 * torch.rand(int(N_link), dtype=torch.double)).to(device)
        demand = torch.tensor([N_link], dtype=torch.double).to(device)
        i = torch.arange(int(N_link)).to(device)
        ii = torch.stack([i, i])
        v = torch.ones(int(N_link), dtype=torch.double).to(device)
        path_edge = sparse.FloatTensor(ii, v).to(device)
        path_demand = torch.ones((1, int(N_link)), dtype=torch.double).to(device)
        path_number = int(N_link)
        
        q = path_demand.t() @ demand
        
        
        
        cap.requires_grad_() 
        p = torch.ones(path_number, dtype=torch.double).to(device)
        p /= path_demand.t() @ (path_demand @ p)
        i = 0
        r = 0.4
        tic = time.time()
        while True:
            f = q * p
            x = path_edge @ f
            t = tfree * (1 + 0.15 * (x / cap) ** 4)
            c = path_edge.t() @ t
            p *= torch.exp(-r * c)
            p /= path_demand.t() @ (path_demand @ p)
            i += 1
            with torch.no_grad():
                TT = torch.dot(x, t) 
                gap = (TT - torch.min(c) * demand[0]) / TT
            if gap < 1e-3:
                Iteration[j, k] = i
                break
        p_star = p * 1.0
        f = q * p_star
        x = path_edge @ f
        t = tfree * (1 + 0.15 * (x / cap) ** 4)
        TT = torch.dot(x, t)
        FP_time[j, k] = time.time() - tic
        """Automatic differentiation"""
        tic = time.time()
        TT.backward()
        AD_time[j, k] = time.time() - tic
        cap.grad.zero_()
        torch.cuda.empty_cache()
        
        if N_link > 10000:
            continue
        """Implicit differentiation"""
        tic = time.time()
        def h(p, cap):
            f = q * p
            x = path_edge @ f
            t = tfree * (1 + 0.15 * (x / cap) ** 4)
            c = path_edge.t() @ t
            p_new = p * torch.exp(-r * c)
            p_new /= path_demand.t() @ (path_demand @ p_new)
            return p_new
        
        h_p, h_cap = jacobian(h, (p_star.detach(), cap.detach()))
        p_cap = torch.linalg.solve(torch.eye(len(p)).to(device) - h_p, h_cap)
        p_vir = p_cap @ cap - (p_cap @ cap - p_star).detach()
        f = demand * p_vir
        x = path_edge @ f
        t = tfree * (1 + 0.15 * (x / cap) ** 4)    
        TT = torch.dot(t, x)
        TT.backward()
        ID_time[j, k] = time.time() - tic
        cap.grad.zero_()
        del(p_cap)
        torch.cuda.empty_cache()

FP_time = FP_time.numpy()
FP_time = np.round(FP_time, 4)
AD_time = AD_time.numpy()
AD_time = np.round(AD_time, 4)
ID_time = ID_time.numpy()
ID_time = np.round(ID_time, 4)
Table = PrettyTable()
Table.field_names = ['|A| = |K|', '10^1', '10^2', '10^3', '10^4', '10^5', '10^6']
Table.add_rows(
        [
                ["Number of iteration"] + list(Iteration.squeeze().numpy()[1:]),
                ["FP time"] + list(FP_time.squeeze()[1:]),
                ["BP time"] + list(AD_time.squeeze()[1:]),
                ["ID time"] + list(ID_time.squeeze()[1:-2]) + ['-', '-']
                ]
        )
print('Computational performance of the proposed BP algorithm and ID')
print(Table)
