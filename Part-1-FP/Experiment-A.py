#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.sparse as sparse
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)

Gap_city = [[], [], [], []]
for city_num in range(4):
    if city_num == 0:
        print('Testing Braess')
        r = 0.25
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
        T_max = 60
    elif city_num == 1:
        print('Testing 3N4L')
        r = 1e-5
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
        T_max = 60
    else:
        if city_num == 2:
            print('Testing Sioux-Falls')
            city = 'sf'
            r = 0.5
        if city_num == 3:
            print('Testing Chicagosketch')
            city = 'chicagosketch'
            r = 10

        """Load Network Data"""
        with open('Network/' + city + '.inputtrp', 'r') as f:
            od_data = f.readlines()
        od_data = [line.split() for line in od_data]
        a_node = [int(od[0]) for od in od_data]
        b_node = [int(od[1]) for od in od_data]
        a_number = torch.tensor(a_node, dtype=torch.int64)
        b_number = torch.tensor(b_node, dtype=torch.int64)
        demand = torch.tensor([float(od[2]) for od in od_data], dtype=torch.double).to(device)
        if city_num == 2:
            demand *= 1.5
   
        with open('Network/' + city +  '.inputnet', 'r') as f:
            link_data = f.readlines()
        link_data = [line.split() for line in link_data]
        s_node = [int(link[0]) for link in link_data]
        t_node = [int(link[1]) for link in link_data]
        s_number = torch.tensor(s_node, dtype=torch.int64)
        t_number = torch.tensor(t_node, dtype=torch.int64)
        cap = torch.tensor([float(l[4]) for l in link_data], dtype=torch.double).to(device)
        length = torch.tensor([float(l[2]) for l in link_data], dtype=torch.double).to(device)
        vmax = torch.tensor([float(l[3]) for l in link_data], dtype=torch.double).to(device)
        tfree = length / vmax

        
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
        
        with open('Network/'  + city + '.demandpath', 'r') as f:
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
        toll_a = 0
        toll_h = 0
        T_max = 150

    gamma = 0.20
    demand_a = gamma * demand
    demand_h = (1 - gamma) * demand
    q_a = path_demand.t() @ demand_a
    q_h = path_demand.t() @ demand_h
    if city_num == 0 or city_num == 1:
       N = 1000
    if city_num == 2 or city_num == 3:
       N = 100
    
    p_a = torch.ones(path_number, dtype=torch.double).to(device)
    p_a /= path_demand.t() @ (path_demand @ p_a)
    
    p_h = torch.ones(path_number, dtype=torch.double).to(device)
    p_h /= path_demand.t() @ (path_demand @ p_h)
    Gap = torch.zeros(N, T_max + 1)
    for k in range(N):
        if k % 100 == 0:
            print('Sample ', k)
        
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
            
            c_a = path_edge.t() @ t
            c_h = path_edge.t() @ t
            
            """compute_gap"""
            TT_a = torch.dot(x_a, t)
            TT_h = torch.dot(x_h, t)
            if city_num == 0 or city_num == 1:
                GT_a = torch.min(c_a) * demand_a[0]
                GT_h = torch.min(c_h) * demand_h[0]
            if city_num == 2 or city_num == 3:
                c_mat[d_number, p_number] = c_a
                GT_a = torch.dot(torch.min(c_mat, 1)[0], demand_a)
                c_mat[d_number, p_number] = c_h
                GT_h = torch.dot(torch.min(c_mat, 1)[0], demand_h)
            gap_a = (TT_a - GT_a) / TT_a
            gap_h = (TT_h - GT_h) / TT_h
            gap = max(gap_a, gap_h)
            print(gap)
            Gap[k, i] = gap
            
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
        
        c_a = path_edge.t() @ t
        c_h = path_edge.t() @ t
        
        """compute_gap"""
        TT_a = torch.dot(x_a, t)
        TT_h = torch.dot(x_h, t)
        if city_num == 0 or city_num == 1:
            GT_a = torch.min(c_a) * demand_a[0]
            GT_h = torch.min(c_h) * demand_h[0]
        if city_num == 2 or city_num == 3:
            c_mat[d_number, p_number] = c_a
            GT_a = torch.dot(torch.min(c_mat, 1)[0], demand_a)
            c_mat[d_number, p_number] = c_h
            GT_h = torch.dot(torch.min(c_mat, 1)[0], demand_h)
        gap_a = (TT_a - GT_a) / TT_a
        gap_h = (TT_h - GT_h) / TT_h
        gap = max(gap_a, gap_h)
        Gap[k, i] = gap
    Gap_city[city_num] = Gap

    
fig = plt.figure(figsize=(12, 3))
for city_num in range(4):
    ax = plt.subplot(1, 4, city_num + 1)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.set_yscale('log')
    Data = torch.log10(Gap_city[city_num])
    T_list = torch.arange(len(Data[0]))
    mean = torch.mean(Data[1:, ], 0)
    std = torch.std(Data[1:, ], 0)
    y1 = 10 ** torch.min(Data, 0)[0]
    y2 = 10 ** torch.max(Data, 0)[0]
    y = 10 ** mean
    plt.fill_between(T_list, y1=y1, y2=y2, alpha=0.2)
    plt.plot(T_list, y, lw=1.5, linestyle='dashed', label='averaged')
    plt.plot(T_list, Gap_city[city_num][0, :], 'r', lw=1.5, label='gap')
    
    plt.grid(linestyle='-', linewidth=1, alpha=0.35)
    if city_num == 0:
        ax.set_title('(a) Braess', fontsize=14)
        plt.xlim([0 - 5, 60 + 5])
    if city_num == 1:
        ax.set_title('(b) 3N4L', fontsize=14)
        plt.xlim([0 - 5, 60 + 5])
    if city_num == 2:
        ax.set_title('(c) Sioux Falls', fontsize=14)
        plt.xlim([0 - 5, 150 + 5])
    if city_num == 3:
        ax.set_title('(d) Chicago Sketch', fontsize=14)
        plt.xlim([0 - 5, 150 + 5])
    
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
fig.text(0.5, 0, r'Number of iterations', ha='center', fontsize=14)
fig.text(0, 0.5, r'Equilibrium gap', va='center', rotation='vertical', fontsize=14)   
plt.tight_layout(pad=2, h_pad=0.25, w_pad=0.5)
plt.subplots_adjust(wspace=0.3)
plt.savefig('Figure-A.pdf', bbox_inches='tight', dpi=1000)
plt.show()
