#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The script "Experiment-H-1.py" must be run first before this script to
generate two files: 'chicagosketch.decrease.pt' and "chicagosketch.grad_pa.pt"
"""
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

city = 'chicagosketch'
 
torch.manual_seed(0)


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
length = torch.tensor([float(l[2]) for l in link_data], dtype=torch.double).to(device)
vmax = torch.tensor([float(l[3]) for l in link_data], dtype=torch.double).to(device)
tfree = length / vmax
s_number = torch.tensor(s_node, dtype=torch.int64)
t_number = torch.tensor(t_node, dtype=torch.int64)

path_edge = torch.load('Network/' + city + '.path_edge.pt', map_location=device)
path_demand = torch.load('Network/' + city + '.path_demand.pt', map_location=device)
c_mat = torch.load('Network/' + city + '.c_mat.pt', map_location=device)
d_number = torch.load('Network/' + city + '.d_number.pt', map_location=device)
p_number = torch.load('Network/' + city + '.p_number.pt', map_location=device)
path_number = path_edge.size()[1]
pool = torch.load('Network/' + city + '.pool.pt', map_location=device)

gamma = 0.2
demand_a0 = gamma * demand
demand_h = (1 - gamma) * demand
q_h = path_demand.t() @ demand_h


T_ue = 4211543.634645345
T_so = 4087371.7178332736


Decrease = torch.load(city + '.decrease.pt')
        

setting_number = 2
T_list = [0, 1, 3, 6, 10]
Results = [[] for _ in range(setting_number)]
for set_num in range(setting_number):
    
    if set_num == 0:
        K = 4000
        print('Testing Scenario A')
    if set_num == 1:
        K = 8000
        print('Testing Scenario B')
    top = torch.topk(Decrease, K)[1]
    
    
    demand_c = torch.zeros_like(demand)
    demand_c[top] = demand_a0[top]
    q_c = path_demand.t() @ demand_c
    
    demand_a = demand_a0 - demand_c
    q_a = path_demand.t() @ demand_a
    
    
    
    alpha = 0.25
    beta = 5000
    obj = 1e+20
    
    Result = torch.zeros(5, len(T_list), dtype=torch.double)
    tic = time.time()
    for jj, T in enumerate(T_list):
        path_c = torch.cat([pool[k] for k in top])
        p_c = torch.ones(len(path_c), dtype=torch.double).to(device)
        p_c_full = torch.zeros(len(q_c), dtype=torch.double).to(device)
        p_c_full[path_c] =  p_c
        for i in top:
            p_c_full[pool[i]] /= torch.sum(p_c_full[pool[i]])
        p_c = p_c_full[path_c]
                
        p_c.requires_grad_()
        
    
        p_a0 = torch.ones(path_number, dtype=torch.double).to(device)
        p_a0 /= path_demand.t() @ (path_demand @ p_a0)
        p_h0 = torch.ones(path_number, dtype=torch.double).to(device)
        p_h0 /= path_demand.t() @ (path_demand @ p_h0)
        descent_num = 0
        while descent_num <= 5000:
            iter_num = 0
        
            p_c_full = torch.zeros(len(q_c), dtype=torch.double).to(device)
            p_c_full[path_c] = p_c
            
            f_c = q_c * p_c_full
            x_c = path_edge @ f_c
            
            p_a = p_a0 * 1.0
            p_h = p_h0 * 1.0
            
            with torch.no_grad():
                f_a = q_a * p_a
                x_a = path_edge @ f_a
                f_h = q_h * p_h
                x_h = path_edge @ f_h
                x = x_a + x_h + x_c
                r_a = torch.nan_to_num((x_a + x_c) / x)
                cap_eff = cap * (1 + 1 * r_a ** 2)
                t = tfree * (1 + 0.15 * (x / cap_eff) ** 4)
                c_a = path_edge.t() @ t
                c_h = path_edge.t() @ t
                
                """compute_gap"""
                TT_a = torch.dot(x_a, t)
                TT_h = torch.dot(x_h, t)
                c_mat[d_number, p_number] = c_a
                GT_a = torch.dot(torch.min(c_mat, 1)[0], demand_a)
                c_mat[d_number, p_number] = c_h
                GT_h = torch.dot(torch.min(c_mat, 1)[0], demand_h)
                gap_a = (TT_a - GT_a) / TT_a
                gap_h = (TT_h - GT_h) / TT_h
                equilibrium_gap = max(gap_a, gap_h)
            
            while iter_num < T:
                f_a = q_a * p_a
                x_a = path_edge @ f_a
                f_h = q_h * p_h
                x_h = path_edge @ f_h
                x = x_a + x_h + x_c
                r_a = torch.nan_to_num((x_a + x_c) / x)
                cap_eff = cap * (1 + 1 * r_a ** 2)
                t = tfree * (1 + 0.15 * (x / cap_eff) ** 4)
                c_a = path_edge.t() @ t
                c_h = path_edge.t() @ t
                p_a *= torch.exp(-alpha * c_a)
                p_a += 1e-30
                p_a /= path_demand.t() @ (path_demand @ p_a)
                p_h *= torch.exp(-alpha * c_h)
                p_h += 1e-30
                p_h /= path_demand.t() @ (path_demand @ p_h)
                iter_num += 1
                
            f_a = q_a * p_a
            x_a = path_edge @ f_a
            f_h = q_h * p_h
            x_h = path_edge @ f_h
            x = x_a + x_h + x_c
            r_a = torch.nan_to_num((x_a + x_c) / x)
            cap_eff = cap * (1 + 1 * r_a ** 2)
            t = tfree * (1 + 0.15 * (x / cap_eff) ** 4)
            c = path_edge.t() @ t
            f = f_a + f_h
            TT = torch.dot(x, t)
            AT = TT / torch.sum(demand)            
            AT.backward()
            
            with torch.no_grad():
                grad = p_c.grad * 1.0
                p_c_previous = p_c * 1.0
                p_c *= torch.exp(-beta * grad)
                p_c_full[path_c] = p_c
                for i in top:
                    p_c_full[pool[i]] /= torch.sum(p_c_full[pool[i]])
                p_c = p_c_full[path_c]
                direction = p_c - p_c_previous
                decrease = torch.dot(direction, -grad)
                p_c.requires_grad_()
                if descent_num % 10 == 0:
                    print('T = ' + str(T) + ',',
                          "iter_num = " + str(descent_num))
                    
            if decrease < 1e-6 and equilibrium_gap < 1e-3:
                Value = TT.detach()
                GPU_time = time.time() - tic
                Iteration_num = descent_num
                Timepiter = GPU_time / Iteration_num
                Result[0, jj] = T
                Result[1, jj] = Value
                Result[2, jj] = GPU_time
                Result[3, jj] = Iteration_num
                Result[4, jj] = Timepiter
                break
            
            with torch.no_grad():
                f_a = q_a * p_a0
                x_a = path_edge @ f_a
                f_h = q_h * p_h0
                x_h = path_edge @ f_h
                x = x_a + x_h + x_c
                r_a = torch.nan_to_num((x_a + x_c) / x)
                cap_eff = cap * (1 + 1 * r_a ** 2)
                t = tfree * (1 + 0.15 * (x / cap_eff) ** 4)
                c_a = path_edge.t() @ t
                c_h = path_edge.t() @ t  
                p_a0 *= torch.exp(-alpha * c_a)
                p_a0 += 1e-30
                p_a0 /= path_demand.t() @ (path_demand @ p_a0)
                p_h0 *= torch.exp(-alpha * c_h)
                p_h0 += 1e-30
                p_h0 /= path_demand.t() @ (path_demand @ p_h0)
            descent_num += 1
    Results[set_num] = Result

"""Visualization"""
T_list = [0, 1, 3, 6, 10]
A = Results[0].numpy()
B = Results[1].numpy()
A[1, :] = (A[1, :] - T_so) / (T_ue - T_so) 
A[2, :] = A[2, :] / 60
B[1, :] = (B[1, :] - T_so) / (T_ue - T_so)
B[2, :] = B[2, :] / 60

for set_num in range(2):
    if set_num == 0:
        Result = A
        print('Scenario A')
    if set_num == 1:
        Result = B
        print('Scenario B')
    Algorithm = torch.arange(len(Result[0, :]))
    Value = Result[1, :]
    Value = np.around(Value, decimals=2)

    GPU_time = Result[2, :]
    GPU_time = np.around(GPU_time, decimals=2)
    
    Iteration = Result[3, :]
    Iteration = Iteration.astype(int)

    Timepiter = Result[4, :]
    Timepiter = np.around(Timepiter, decimals=2)
    
    Table = PrettyTable()
    Table.field_names = ["", 'S-0', 'S-1', 'S-3', 'S-6', 'S-10']
    Table.add_rows(
            [
                    ["Remaining gap"] + list(Value),
                    ["Total GPU time"] + list(GPU_time),
                    ["Iteration number"] + list(Iteration),
                    ["Time per iteration"] + list(Timepiter)
                    ]
            )
    print(Table)
    print('')

    fig = plt.figure(figsize=(14.5, 2.8))
    
    for plot_num in range(4):
        ax = plt.subplot(1, 4, plot_num + 1)
        ax.spines['top'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        
        if plot_num == 0:
            Y = Value
            plt.bar(Algorithm[:], Y[:], 0.8, color='firebrick')
            ax.set_title('(a) Remaining gap', fontsize=16)
            plt.ylim([0.3, 0.85])
            for i, v in enumerate(Y):
                
                if v > 0:
                    y = v + 0.014
                    ax.text(i - 0.36, y, str(format(float(v), '.2f')), fontsize=12)
                if v < 0:
                    y = v - 0.042
                    ax.text(i - 0.42, y, str(format(float(v), '.2f')), fontsize=12)        
            
        if plot_num == 1:
            Y = GPU_time
            plt.bar(Algorithm[:], Y[:], 0.8, color='firebrick')
            ax.set_title('(b) Total GPU time (min)', fontsize=16)
            
            plt.ylim([0, 110])
            for i, v in enumerate(Y):
                y = v + 2
                
                if v >= 100:
                    ax.text(i - 0.34, y, str(format(float(v), '.0f')), fontsize=12)
                if v < 100 and v >= 10:
                    ax.text(i - 0.34, y, str(format(float(v), '.1f')), fontsize=12)
                if v < 10:
                    ax.text(i - 0.34, y, str(format(float(v), '.2f')), fontsize=12)
            
            
        if plot_num == 2:
            Y = Iteration
            plt.bar(Algorithm[:], Y[:], 0.8, color='firebrick')
            ax.set_title('(c) Itertation number', fontsize=16)
            
            plt.ylim([0, 1300])
            for i, v in enumerate(Y):
                y = v + 30
                if v >= 100 and v < 1000:
                    alpha = 0.3
                if v >= 1000:
                    alpha = 0.35
                if v < 100:
                    alpha = 0.22
                ax.text(i - alpha, y, str(int(v)), fontsize=12)
                
        if plot_num == 3:
            Y = Timepiter
            plt.bar(Algorithm[:], Y[:], 0.8, color='firebrick')
            ax.set_title('(d) Time per iteration (s)', fontsize=16)
            plt.ylim([0, 7.75])
            for i, v in enumerate(Y):
                y = v + 0.24
                
                if v >= 100:
                    ax.text(i - 0.34, y, str(format(float(v), '.0f')), fontsize=12)
                if v < 100 and v >= 10:
                    ax.text(i - 0.34, y, str(format(float(v), '.1f')), fontsize=12)
                if v < 10 and v >= 1:
                    ax.text(i - 0.34, y, str(format(float(v), '.2f')), fontsize=12)
                if v < 1:
                    ax.text(i - 0.34, y, str(format(float(v), '.2f')), fontsize=12)
        
        Algorithm_name = list()
        for T in T_list:
            Algorithm_name.append(r'S-' + str(T))
        plt.xticks(Algorithm, Algorithm_name, fontsize=13)
        plt.yticks(fontsize=13)
    plt.subplots_adjust(wspace=0.25)
    
    if set_num == 0:
        plt.savefig('Result-H-2.pdf', bbox_inches='tight', dpi=10000)
    if set_num == 1:
        plt.savefig('Result-H-3.pdf', bbox_inches='tight', dpi=10000)
    plt.close()