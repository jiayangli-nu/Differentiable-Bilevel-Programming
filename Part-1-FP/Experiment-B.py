#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt


device = torch.device("cpu")

torch.manual_seed(0)

TT_city = [[], []]
Entropy_city = [[], []]
for city_num in range(2):
    if city_num == 0:
        print('Testing Braess')
        r = 0.1
        path_edge = torch.tensor([[1, 0, 1, 0, 0],
                      [1, 0, 0, 1, 1],
                      [0, 1, 0, 0, 1]], dtype=torch.double).t().to(device)
        path_demand = torch.tensor([[1, 1, 1]], dtype=torch.double).to(device)
        tfree = torch.tensor([1, 3, 3, 0.5, 1], dtype=torch.double).to(device)
        cap = torch.tensor([2, 4, 4, 1, 2], dtype=torch.double).to(device)
        demand = torch.tensor([6], dtype=torch.double).to(device)   
        path_number = path_edge.size()[1]
        toll_a = 0
        toll_h = 0
        T_max = 500
    elif city_num == 1:
        print('Testing 3N4L')
        r = 1e-4
        path_edge = torch.tensor([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [1, 0, 0, 1],
                      [0, 1, 1, 0]], dtype=torch.double).t().to(device)
        path_demand = torch.tensor([[1, 1, 1, 1]], dtype=torch.double).to(device)
        A = torch.tensor([4, 20, 1, 30], dtype=torch.double).to(device)
        B = torch.tensor([1, 5, 30, 1], dtype=torch.double).to(device)
        tfree = A
        cap = (0.15 * tfree / B) ** 0.25
        time = lambda x: A + B * x ** 4
        demand = torch.tensor([10], dtype=torch.double).to(device)
        path_number = path_edge.size()[1]
        toll_a = 0
        toll_h = 0
        T_max = 500

    gamma = 0.2
    demand_a = gamma * demand
    demand_h = (1 - gamma) * demand
    q_a = path_demand.t() @ demand_a
    q_h = path_demand.t() @ demand_h
    N = 5000
    TT_list = torch.zeros(N)
    Entropy_list = torch.zeros(N)
    
    
    p_a = torch.ones(path_number, dtype=torch.double).to(device)
    p_a /= path_demand.t() @ (path_demand @ p_a)
    
    p_h = torch.ones(path_number, dtype=torch.double).to(device)
    p_h /= path_demand.t() @ (path_demand @ p_h)
        
    for k in range(N):
        if k % 500 == 0:
            print('Completed ', str(k) + ' / 5000')
        
        if k == 0:
            p_a = torch.ones(path_number, dtype=torch.double).to(device)
            p_h = torch.ones(path_number, dtype=torch.double).to(device)
        else:
            p_a = torch.rand(path_number, dtype=torch.double).to(device)
            p_h = torch.rand(path_number, dtype=torch.double).to(device)
        p_a /= path_demand.t() @ (path_demand @ p_a)
        p_h /= path_demand.t() @ (path_demand @ p_h)
        
        i = 0
        while i < T_max:
            f_a = q_a * p_a
            x_a = path_edge @ f_a
            f_h = q_h * p_h
            x_h = path_edge @ f_h
            x = x_a + x_h
            r_a = torch.nan_to_num(x_a / x)
            cap_eff = cap * (1 + r_a ** 2)
            t = tfree * (1 + 0.15 * (x / cap_eff) ** 4)
            u_a = t + toll_a
            u_h = t + toll_h
            c_a = path_edge.t() @ u_a
            c_h = path_edge.t() @ u_h
            
            TT_a = torch.dot(x_a, u_a)
            TT_h = torch.dot(x_h, u_h)
            GT_a = torch.min(c_a) * demand_a[0]
            GT_h = torch.min(c_h) * demand_h[0]
  
            gap_a = (TT_a - GT_a) / TT_a
            gap_h = (TT_h - GT_h) / TT_h
            gap = max(gap_a, gap_h)
            
            
            p_h *= torch.exp(-r * c_h)
            p_h /= path_demand.t() @ (path_demand @ p_h)
            p_a *= torch.exp(-r * c_a)
            p_a /= path_demand.t() @ (path_demand @ p_a)
            i += 1
            
        f_a = q_a * p_a
        x_a = path_edge @ f_a
        f_h = q_h * p_h
        x_h = path_edge @ f_h
        x = x_a + x_h
        r_a = torch.nan_to_num(x_a / x)
        cap_eff = cap * (1 + r_a ** 2)
        t = tfree * (1 + 0.15 * (x / cap_eff) ** 4)
        TT = torch.dot(x, t)
        TT_list[k] = TT
        Entropy_list[k] = torch.dot(f_a, p_a) + torch.dot(f_h, p_h)
    TT_city[city_num] = TT_list
    Entropy_city[city_num] = Entropy_list
    
    
fig = plt.figure(figsize=(9, 2.5))
for city_num in range(2):
    ax = plt.subplot(1, 2, city_num + 1)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    
    if city_num == 0:
        Data = TT_city[city_num]
    if city_num == 1:
        Data = TT_city[city_num] / 1000
        
    plt.grid(linestyle='-', linewidth=1, alpha=0.5)
    freq = plt.hist(Data, 50, density=True, alpha = 0.8)
    if city_num == 0:
        ax.set_title('(a) Braess', fontsize=14)
        plt.vlines(Data[0], 0, 12.5, 'r', linestyle='dashed', linewidth=2)
    if city_num == 1:
        ax.set_title('(b) 3N4L', fontsize=14)
        plt.vlines(Data[0], 0, 1.5, 'r', linestyle='dashed', linewidth=2)
 
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
fig.text(0.5, 0, r'Total travel time at equilbrium', ha='center', fontsize=14)
fig.text(0, 0.5, r'Density', va='center', rotation='vertical', fontsize=14)   
plt.tight_layout(pad=2, h_pad=0.25, w_pad=0.5)
plt.subplots_adjust(wspace=0.2)
plt.savefig('Result-B.pdf', bbox_inches='tight', dpi=1000)
plt.show()
