#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:41:10 2021

@author: jiayang Li
"""
import time
import torch
import torch.sparse as sparse
import matplotlib.pyplot as plt


def compute_gap(c, f, demand):
    c_max = torch.max(c) + 1
    c_min = c_max - torch.max((c_max - path_demand * c) * path_demand, 1)[0]
    gt = torch.dot(c_min, demand)
    tt = torch.dot(c, f)
    gap = tt - gt
    return gap / tt

device = torch.device("cpu")

torch.manual_seed(0)

"""Load Network Data"""

with open('Network/sf.inputtrp', 'r') as f:
    od_data = f.readlines()
od_data = [line.split() for line in od_data]
a_node = [int(od[0]) for od in od_data]
b_node = [int(od[1]) for od in od_data]
a_number = torch.tensor(a_node, dtype=torch.int64)
b_number = torch.tensor(b_node, dtype=torch.int64)
demand = torch.tensor([float(od[2]) for od in od_data], dtype=torch.float64)
demand = demand / 1000
    
with open('Network/sf.inputnet', 'r') as f:
    link_data = f.readlines()
link_data = [line.split() for line in link_data]
s_node = [int(link[0]) for link in link_data]
t_node = [int(link[1]) for link in link_data]
s_number = torch.tensor(s_node, dtype=torch.int64)
t_number = torch.tensor(t_node, dtype=torch.int64)
cap = torch.tensor([float(l[4]) for l in link_data], dtype=torch.float64)
length = torch.tensor([float(l[2]) for l in link_data], dtype=torch.float64)
vmax = torch.tensor([float(l[3]) for l in link_data], dtype=torch.float64)
tfree = length / vmax
B = tfree * 0.09
K = cap / 1000
A = tfree * 0.6


with open('Network/sf.edgepath', 'r') as f:
    epmatrix_data = f.readlines()
epmatrix_data = [line.split() for line in epmatrix_data]
e_loc = [int(link[0]) for link in epmatrix_data]
p_loc = [int(link[1]) for link in epmatrix_data]
i = torch.tensor([e_loc, p_loc])
v = torch.ones(len(e_loc), dtype=torch.float64)
m = len(cap)
l = max(p_loc) + 1
path_edge = sparse.FloatTensor(i, v, torch.Size([m, l])).to_dense()

with open('Network/sf.demandpath', 'r') as f:
    dpmatrix_data = f.readlines()
dpmatrix_data = [line.split() for line in dpmatrix_data]

d_loc = [int(link[0]) for link in dpmatrix_data]
p_loc = [int(link[1]) for link in dpmatrix_data]

i = torch.tensor([d_loc, p_loc])
v = torch.ones(len(d_loc), dtype=torch.float64)
m = len(od_data)
l = max(p_loc) + 1
path_demand = sparse.FloatTensor(i, v, torch.Size([m, l])).to_dense()

path_number = path_edge.size()[1]
 

q = path_demand.t() @ demand

T_ue = 74.80225318717638
T_so = 71.94256096965475

x_ue = torch.tensor(
        [4.4947,  8.1191,  4.5191,  5.9673,  8.0947, 14.0064, 10.0223, 14.0306,
        18.0064,  5.2000, 18.0306,  8.7983, 15.7808,  5.9918,  8.8065, 12.4929,
        12.1015, 15.7940, 12.5256, 12.0409,  6.8827,  8.3887, 15.7967,  6.8367,
        21.7441, 21.8141, 17.7266, 23.1258, 11.0471,  8.1000,  5.3000, 17.6042,
         8.3653,  9.7761,  9.9737,  8.4049, 12.2876, 12.3786, 11.1214,  9.8141,
         9.0363,  8.4004, 23.1923,  9.0798, 19.0833, 18.4099,  8.4067, 11.0730,
        11.6950, 15.2783,  8.1000, 11.6838,  9.9530, 15.8546, 15.3334, 18.9768,
        19.1167,  9.9419,  8.6884, 18.9925,  8.7106,  6.3020,  7.0000,  6.2400,
         8.6195, 10.3094, 18.3865,  7.0000,  8.6074,  9.6618,  8.3949,  9.6262,
         7.9030, 11.1124, 10.2595,  7.8618], dtype=torch.float64)
x_so = torch.tensor(
        [7.6200, 11.2393,  7.6393,  6.6200, 11.2200, 17.5014, 14.3135, 17.4742,
        18.7324,  6.0805, 18.8002,  6.9944, 16.8956,  6.6393,  7.0140, 12.5567,
        13.2722, 16.3244, 12.5957, 13.2244,  7.5655,  7.9680, 16.9439,  7.5357,
        21.7647, 21.8831, 17.4678, 23.3610, 10.7448,  8.3203,  6.0855, 17.3875,
         7.3250,  9.4430, 14.3214,  7.3239, 15.5385, 15.6452, 10.5385,  9.4688,
         8.9171,  7.6143, 23.4207,  8.9551, 18.5560, 16.1717,  7.9889, 10.7667,
        10.5989, 18.4613,  8.3374, 10.5823,  8.2572, 16.3722, 18.5208, 20.7003,
        18.5700,  8.2577,  8.6983, 20.7076,  8.7128,  6.3588,  7.1172,  6.3145,
         7.9379,  9.8376, 16.1554,  7.0833,  7.9475,  9.3000,  7.6021,  9.2593,
         7.7390, 10.5452,  9.7837,  7.6862], dtype=torch.float64)
     

r = 5
N_max = 10000
rho = 0.0025

tol_up = 0.05
tol_lw = 1e-4

for setting_num in range(2):
    if setting_num == 0:
        edge_toll = torch.where(x_ue / x_so > 1.05)[0]
        print('Testing Scenario A')
    if setting_num == 1:
        edge_toll = torch.where(x_ue / x_so > 1.15)[0]
        print('Testing Scenario B')
    
    """Dol-MD"""
    Traj_ad = list()
    Time_ad = list()
    
    toll = torch.zeros(len(edge_toll), dtype=torch.double)
    toll.requires_grad_()
    
    p_0 = torch.ones(path_number, dtype=torch.double)
    p_0 /= path_demand.t() @ (path_demand @ p_0)
    Traj_ad = list()
    Time_ad = list()
    tic = time.time()
    
    iter_num = 0
    q = path_demand.t() @ demand
    while iter_num <= N_max:
        toll_full = torch.zeros_like(cap)
        toll_full[edge_toll] = toll
        
        p = p_0 * 1.0
        i = 0
        while True:
            i += 1
            f = q * p
            x = path_edge @ f
            t = A + B * (x / K) ** 4
            u = t + toll_full
            c = path_edge.t() @ u
            with torch.no_grad():
                gap = compute_gap(c, f, demand)
                if gap < tol_lw:
                    break
            
            p *= torch.exp(-r * c)
            p /= path_demand.t() @ (path_demand @ p)
           
        f = q * p
        x = path_edge @ f
        t = A + B * (x / K) ** 4
        tt = torch.dot(x, t)
        obj = tt
        Traj_ad.append(obj.detach())
        Time_ad.append(time.time() - tic)
        obj.backward()
        grad = toll.grad * 1.0
        grad[toll == 0] = 0
        if iter_num > 0 and torch.norm(grad) < tol_up:
            break
        
        with torch.no_grad():
            toll -= rho * toll.grad
            toll.clamp_(0)
        toll.grad.zero_()
        
        iter_num += 1
        Traj_ad.append(obj.detach())
        Time_ad.append(time.time() - tic)
     
    Traj_ad = torch.tensor(Traj_ad)
    Time_ad = torch.tensor(Time_ad)
    
    
    """Sil-MD"""
    T_list = [1, 3, 6, 10, 15]
    
    Traj_c_data = list()
    Time_c_data = list()
    for jj, T in enumerate(T_list):
        Traj_c = list()
        Time_c = list()
        toll = torch.zeros(len(edge_toll), dtype=torch.double) + 0
        toll.requires_grad_()
        
        toll_full = torch.zeros_like(cap)
        toll_full[edge_toll] = toll
        
        p_0 = torch.ones(path_number, dtype=torch.double)
        p_0 /= path_demand.t() @ (path_demand @ p_0)
        f = q * p_0
        x = path_edge @ f
        t = A + B * (x / K) ** 4
        tt = torch.dot(x, t)
        obj = tt
        Traj_c.append(obj.detach())
        Time_c.append(0)    
        tic = time.time()
        
        iter_num = 0
        while iter_num <= N_max:
            
            toll_full = torch.zeros_like(cap)
            toll_full[edge_toll] = toll

            p = p_0 * 1.0
            i = 0
            while i < T:
                f = q * p
                x = path_edge @ f
                t = A + B * (x / K) ** 4
                u = t + toll_full
                c = path_edge.t() @ u
                p *= torch.exp(-r * c)
                p /= path_demand.t() @ (path_demand @ p)         
                i += 1
                
            f = q * p
            x = path_edge @ f
            t = A + B * (x / K) ** 4
            tt = torch.dot(x, t)
            obj = tt
            Traj_c.append(obj.detach())
            Time_c.append(time.time() - tic)
            obj.backward()
            
            grad = toll.grad * 1.0
            with torch.no_grad():
                toll -= rho * toll.grad
                toll.clamp_(0)
            toll.grad.zero_()
            
            with torch.no_grad():
                
                f = q * p_0
                x = path_edge @ f
                t = A + B * (x / K) ** 4
                c = path_edge.t() @ (t + toll_full)
                
                equilibrium_gap = compute_gap(c, f, demand)
                    
                p_0 *= torch.exp(-r * c)
                p_0 /= path_demand.t() @ (path_demand @ p_0)
                if iter_num < 50:
                    p_0 += 0.01 * torch.ones_like(p_0) / (iter_num + 1) ** 2
                    p_0 /= path_demand.t() @ (path_demand @ p_0)
                
            iter_num += 1
            grad[toll == 0] = 0      
            if torch.norm(grad) < tol_up and equilibrium_gap < tol_lw:
                break
        i = 0
        while i < T:
            f = q * p_0
            x = path_edge @ f
            t = A + B * (x / K) ** 4
            u = t + toll_full
            c = path_edge.t() @ u
            p_0 *= torch.exp(-r * c)
            p_0 /= path_demand.t() @ (path_demand @ p_0)
            i += 1
    
        f = q * p_0
        x = path_edge @ f
        t = A + B * (x / K) ** 4
        tt = torch.dot(x, t)
        obj = tt
        Traj_c.append(obj.detach())
        Time_c.append(time.time() - tic)
        Traj_c_data.append(torch.tensor(Traj_c))
        Time_c_data.append(torch.tensor(Time_c))
        
        
    N_algorithm = len(T_list) + 1
    Algorithm = torch.arange(N_algorithm)
    Value = torch.zeros(N_algorithm)
    Value[-1] = Traj_ad[-1]
    
    CPU_time = torch.zeros(N_algorithm)
    CPU_time[-1] = Time_ad[-1]
    
    Iteration = torch.zeros(N_algorithm)
    Iteration[-1] = len(Time_ad)
    
    for ii in range(len(T_list)):
        Value[ii] = Traj_c_data[ii][-1]
        CPU_time[ii] = Time_c_data[ii][-1]
        Iteration[ii] = len(Time_c_data[ii])  
    Timepiter = CPU_time / Iteration
    
    fig = plt.figure(figsize=(14.5, 2.8))

    for plot_num in range(4):
        ax = plt.subplot(1, 4, plot_num + 1)
        ax.spines['top'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        
        if plot_num == 0:
            if setting_num == 0:
                Y = (Value - 72.6238) / 72.6238 * 100
            if setting_num == 1:
                Y = (Value - 73.8787) / 73.8787 * 100
            plt.bar(Algorithm[: -1], Y[: -1], 0.8, color='firebrick')
            plt.bar(Algorithm[-1], Y[-1], 0.8, color='darkblue')
            ax.set_title('(a) Relative gap (%)', fontsize=16)
            if setting_num == 0:
                plt.ylim([-0.25, 0.2])
            if setting_num == 1:
                plt.ylim([-0.25, 0.05])
                
            for i, v in enumerate(Y):
                
                if v > 0:
                    y = v + 0.012
                    ax.text(i - 0.40, y, str(format(float(v), '.2f')), fontsize=12)
                if v < 0:
                    y = v - 0.042
                    ax.text(i - 0.45, y, str(format(float(v), '.2f')), fontsize=12)
            
        if plot_num == 1:
            Y = CPU_time
            plt.bar(Algorithm[: -1], Y[: -1], 0.8, color='firebrick')
            plt.bar(Algorithm[-1], Y[-1], 0.8, color='darkblue')
            ax.set_title('(b) Total CPU time (s)', fontsize=16)
            
            ax.set_yscale('log')
            ax.set_ylim(top=60)
            for i, v in enumerate(Y):
                y = v * 1.15
                
                if v >= 100:
                    ax.text(i - 0.37, y, str(format(float(v), '.0f')), fontsize=12)
                if v < 100 and v >= 10:
                    ax.text(i - 0.37, y, str(format(float(v), '.1f')), fontsize=12)
                if v < 10:
                    ax.text(i - 0.37, y, str(format(float(v), '.2f')), fontsize=12)
            
            
        if plot_num == 2:
            Y = Iteration
            plt.bar(Algorithm[: -1], Y[: -1], 0.8, color='firebrick')
            plt.bar(Algorithm[-1], Y[-1], 0.8, color='darkblue')
            ax.set_title('(c) Itertation number', fontsize=16)
            
            plt.ylim([0, 300])
            for i, v in enumerate(Y):
                y = v + 10
                if v >= 100 and v < 1000:
                    alpha = 0.33
                if v >= 1000:
                    alpha = 0.55
                if v < 100:
                    alpha = 0.22
                ax.text(i - alpha, y, str(int(v)), fontsize=12)
                
        if plot_num == 3:
            Y = Timepiter * 1000
            plt.bar(Algorithm[: -1], Y[: -1], 0.8, color='firebrick')
            plt.bar(Algorithm[-1], Y[-1], 0.8, color='darkblue')
            ax.set_yscale('log')
            ax.set_title('(d) Time per iteration (ms)', fontsize=16)
            ax.set_ylim(top=350)
            for i, v in enumerate(Y):
                y = v * 1.15
                
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
        Algorithm_name.append(r'D')
        plt.xticks(Algorithm, Algorithm_name, fontsize=13)
        plt.yticks(fontsize=13)
    plt.tight_layout(pad=2, h_pad=0.5, w_pad=0.5)
    plt.subplots_adjust(wspace=0.18)
    
    if setting_num == 0:
        plt.savefig('Result-G-1.pdf', bbox_inches='tight', dpi=1000)
    if setting_num == 1:
        plt.savefig('Result-G-2.pdf', bbox_inches='tight', dpi=1000)
    
    plt.close()
