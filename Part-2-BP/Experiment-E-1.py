#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
import torch.sparse as sparse
from prettytable import PrettyTable

""""
This script only tests ILD
"""

"""Testing CPU"""
device = torch.device("cpu")
testing = 'cpu'

"""Testing GPU"""
#device = torch.device("cuda:0")
testing = 'gpu'


city_size = [10, 10, 20, 30, 40, 50]
Path_num = torch.zeros(len(city_size), dtype=torch.double).to(device)
N_link = torch.zeros(len(city_size), dtype=torch.double).to(device)
FP_time = torch.zeros(len(city_size), dtype=torch.double).to(device)
BP_time = torch.zeros(len(city_size), dtype=torch.double).to(device)
Iteration_num = torch.zeros(len(city_size), dtype=torch.double).to(device)
for jj, N in enumerate(city_size):
    if jj == 0:
        print('This sample is used to initialize ' + 'testing')
    if jj > 0:
        print('Testing EP with ' + str(N) + ' x ' + str(N) + ' network using ' + testing)
    city = 'Grid_' + str(N) + '_0.125_1'
    if N == 50:
        city = 'Grid_' + str(N) + '_0.1252_1'
    
    with open('Network/' + city + '.inputtrp', 'r') as f:
        od_data = f.readlines()
    od_data = [line.split() for line in od_data]
    a_node = [int(od[0]) for od in od_data]
    b_node = [int(od[1]) for od in od_data]
    a_number = torch.tensor(a_node, dtype=torch.int64)
    b_number = torch.tensor(b_node, dtype=torch.int64)
    demand = torch.tensor([float(od[2]) for od in od_data], dtype=torch.double).to(device)
       
    with open('Network/' + city +  '.inputnet', 'r') as f:
        link_data = f.readlines()
    link_data = [line.split() for line in link_data]
    s_node = [int(link[0]) for link in link_data]
    t_node = [int(link[1]) for link in link_data]
    
    
    cap = torch.tensor([float(l[4]) for l in link_data], dtype=torch.double).to(device)
    cap.requires_grad_()
    length = torch.tensor([float(l[2]) for l in link_data], dtype=torch.double).to(device)
    vmax = torch.tensor([float(l[3]) for l in link_data], dtype=torch.double).to(device)
    tfree = length / vmax
    s_number = torch.tensor(s_node, dtype=torch.int64)
    t_number = torch.tensor(t_node, dtype=torch.int64)
    
    with open('Network/' + city + '.edgepath', 'r') as f:
        epmatrix_data = f.readlines()
    epmatrix_data = [line.split() for line in epmatrix_data]
    e_loc = [int(link[0]) for link in epmatrix_data]
    p_loc = [int(link[1]) for link in epmatrix_data]
    i = torch.tensor([e_loc, p_loc])
    v = torch.ones(len(e_loc), dtype=torch.double)
    m = len(cap)
    l = max(p_loc) + 1
    path_edge = sparse.FloatTensor(i, v, torch.Size([m, l])).to(device)
    
    with open('Network/' + city + '.demandpath', 'r') as f:
        dpmatrix_data = f.readlines()
    dpmatrix_data = [line.split() for line in dpmatrix_data]
    
    d_loc = [int(link[0]) for link in dpmatrix_data]
    p_loc = [int(link[1]) for link in dpmatrix_data]
    
    i = torch.tensor([d_loc, p_loc])
    v = torch.ones(len(d_loc), dtype=torch.double)
    m = len(od_data)
    l = max(p_loc) + 1
    path_demand = sparse.FloatTensor(i, v, torch.Size([m, l])).to(device)
    
    path_number = path_edge.size()[1]
    Path_num[jj] = path_number
    N_link[jj] = len(e_loc) / path_number
    
    
    d_node = list()
    p_node = list()
    path_count = torch.zeros_like(demand)
    for j in range(len(d_loc)):
        d_node.append(d_loc[j])
        p_node.append(int(path_count[d_loc[j]]))
        path_count[d_loc[j]] += 1
    d_number = torch.tensor(d_node).to(device)
    p_number = torch.tensor(p_node).to(device)
    c_mat = 1e+7 + torch.zeros((len(demand), torch.max(p_number) + 1), dtype=torch.double).to(device)
    
    gamma = 0.20
    demand_a = gamma * demand
    demand_h = (1 - gamma) * demand
    q_a = path_demand.t() @ demand_a
    q_h = path_demand.t() @ demand_h
    
    with torch.no_grad():
        p_a = torch.ones(path_number, dtype=torch.double).to(device)
        p_a /= path_demand.t() @ (path_demand @ p_a)
        p_h = torch.ones(path_number, dtype=torch.double).to(device)
        p_h /= path_demand.t() @ (path_demand @ p_h)
        
        f_a = q_a * p_a
        x_a = path_edge @ f_a
        f_h = q_h * p_h
        x_h = path_edge @ f_h
        x = x_a + x_h
        r_a = torch.nan_to_num(x_a / x)
        cap_eff = cap * (1 + r_a ** 2)
        t = tfree * (1 + 0.15 * (x / cap_eff) ** 4)
        c = path_edge.t() @ t
        r = 25 / torch.max(c)
    
    i = 0
    T_max = 1000
    tic = time.time()
    
    time_gap = 0
    while i < T_max:
        f_a = q_a * p_a
        x_a = path_edge @ f_a
        f_h = q_h * p_h
        x_h = path_edge @ f_h
        x = x_a + x_h
        r_a = torch.nan_to_num(x_a / x)
        cap_eff = cap * (1 + r_a ** 2)
        t = tfree * (1 + 0.15 * (x / cap_eff) ** 4)
        c = path_edge.t() @ t
        
        TT_a = torch.dot(x_a, t)
        TT_h = torch.dot(x_h, t)
        
        tic_gap = time.time()
        with torch.no_grad():
            c_mat[d_number, p_number] = c
            GT_a = torch.dot(torch.min(c_mat, 1)[0], demand_a)
            c_mat[d_number, p_number] = c
            GT_h = torch.dot(torch.min(c_mat, 1)[0], demand_h)
            gap_a = (TT_a - GT_a) / TT_a
            gap_h = (TT_h - GT_h) / TT_h
            gap = max(gap_a, gap_h)
            if gap < 1e-4:
                break
        time_gap += time.time() - tic_gap
 
        p_h *= torch.exp(-r * c)
        p_h /= path_demand.t() @ (path_demand @ p_h)
        p_a *= torch.exp(-r * c)
        p_a /= path_demand.t() @ (path_demand @ p_a)
        i += 1
    Iteration_num[jj] = i
    f_a = q_a * p_a
    x_a = path_edge @ f_a
    f_h = q_h * p_h
    x_h = path_edge @ f_h
    x = x_a + x_h
    r_a = torch.nan_to_num(x_a / x)
    cap_eff = cap * (1 + r_a ** 2)
    t = tfree * (1 + 0.15 * (x / cap_eff) ** 4)
    TT = torch.dot(x, t)
    FP_time[jj] = time.time() - tic - time_gap
    
    tic = time.time()
    TT.backward()
    BP_time[jj] = time.time() - tic
    torch.cuda.empty_cache()

Path_num = Path_num.numpy()
Path_num = np.round(Path_num, 0)
Path_num = Path_num.astype(int)

N_link = N_link.numpy()
N_link = np.round(N_link, 0)
N_link = N_link.astype(int)

Iteration_num = Iteration_num.numpy()
Iteration_num = np.round(Iteration_num, 0)
Iteration_num = Iteration_num.astype(int)

FP_time = FP_time.numpy()
FP_time = np.round(FP_time, 4)

BP_time = BP_time.numpy()
BP_time = np.round(BP_time, 4)

Table = PrettyTable()
Table.field_names = ['|N|', '10 x 10', '20 x 20', '30 x 30', '40 x 40', '50 x 50']
Table.add_rows(
        [
                ["|K^*|"] + list(Path_num[1:]),
                ["N_link"] + list(N_link[1:]),
                ["T_iter"] + list(Iteration_num[1:]),
                ["FP time"] + list(FP_time[1:]),
                ["BP time"] + list(BP_time[1:])
                ]
        )
print('Computational performance of FP and BP by ILD ')
print(Table)

    
    
