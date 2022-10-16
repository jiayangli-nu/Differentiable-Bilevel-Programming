#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
from prettytable import PrettyTable


def compute_gap(c, f, demand):
    c_max = torch.max(c) + 1
    c_min = c_max - torch.max((c_max - path_demand * c) * path_demand, 1)[0]
    gt = torch.dot(c_min, demand)
    tt = torch.dot(c, f)
    gap = tt - gt
    return gap / tt

  
"""Testing Scenario A"""
edge_enh = torch.tensor([0, 1, 2, 4])

"""Testing Scenario B"""
#edge_enh = torch.tensor([0, 1, 2, 3, 4])

torch.manual_seed(0)

path_edge = torch.tensor([[1, 0, 1, 0, 0],
                      [1, 0, 0, 1, 1],
                      [0, 1, 0, 0, 1]], dtype=torch.double).t()
path_demand = torch.tensor([[1, 1, 1]], dtype=torch.double)
tfree = torch.tensor([1, 3, 3, 0.5, 1], dtype=torch.double)
cap = torch.tensor([2, 4, 4, 1, 2], dtype=torch.double)
demand = torch.tensor([6], dtype=torch.double)
path_number = path_edge.size()[1]


cash = torch.zeros_like(cap)

cash[edge_enh] = tfree[edge_enh]


gamma = 0
q = path_demand.t() @ demand

def path2link(p, enhence):
    f = q * p
    x = path_edge @ f
    t = tfree * (1 + 0.15 * (x / (cap + enhence)) ** 4)
    return x, t


def h(p, enhence):
    t = path2link(p, enhence)[1]
    c = path_edge.t() @ t
    p *= torch.exp(-r * c)
    p /= path_demand.t() @ (path_demand @ p)
    return p

r = 0.5
gamma = 1
N_max = 10000
alpha = 0.005

Enh = torch.zeros(8, len(cap), dtype=torch.double)
P = torch.zeros(8, len(path_edge[0]))

"""System optimum"""
enh = torch.zeros(len(edge_enh), dtype=torch.double)
enh.requires_grad_()

p_0 = torch.ones(path_number, dtype=torch.double)
p_0 /= path_demand.t() @ (path_demand @ p_0)

p_0.requires_grad_()
iter_num = 0
while iter_num <= N_max:
    
    enhence = torch.zeros_like(cap)
    enhence[edge_enh] = enh

    p = p_0 * 1.0
    x, t = path2link(p, enhence)
    tt = torch.dot(x, t)
    obj = tt + gamma * torch.dot(cash, enhence ** 2)
    obj.backward()  
    grad = enh.grad * 1.0
    with torch.no_grad():
        enh -= alpha * enh.grad
        enh.clamp_(0)
        
        p_0 *= torch.exp(-0.05 * p_0.grad)
        p_0 /= path_demand.t() @ (path_demand @ p_0)
    enh.grad.zero_()
    p_0.grad.zero_()
    iter_num += 1


obj_lower = tt + gamma * torch.dot(cash, enhence ** 2)
obj_lower = obj_lower.detach().numpy()
obj_lower = np.around(obj_lower, 4)

Enh[-1, edge_enh] = enh.detach()
P[-1, :] = p_0.detach()


"""IOA"""
Traj_ioa = list()
Time_ioa = list()

enh = torch.zeros(len(edge_enh), dtype=torch.double)
enh.requires_grad_()
p_0 = torch.ones(path_number, dtype=torch.double)
p_0 /= path_demand.t() @ (path_demand @ p_0)

enhence = torch.zeros_like(cap)
enhence[edge_enh] = enh
x, t = path2link(p_0, enhence)
tt = torch.dot(x, t)
obj = tt + gamma * torch.dot(cash, enhence ** 2)
q = path_demand.t() @ demand

Traj_ioa.append(obj.detach())
Time_ioa.append(0)
tic = time.time()
  
iter_num = 0

converge = False
while iter_num <= N_max:
    
    """assignment"""
    with torch.no_grad():
        enhence = torch.zeros_like(cap)
        enhence[edge_enh] = enh
        while True:
            f = q * p_0
            x = path_edge @ f
            t = tfree * (1 + 0.15 * (x / (cap + enhence)) ** 4)
            c = path_edge.t() @ t
            gap = compute_gap(c, f, demand)
            if gap < 1e-8:
                break
            
            p_0 *= torch.exp(-r * c)
            p_0 /= path_demand.t() @ (path_demand @ p_0)
        p = p_0 * 1.0
    
    """optimization"""
    enh_pre = enh.detach() * 1.0
    jjj = 0
    while True:
        enhence = torch.zeros_like(cap)
        enhence[edge_enh] = enh
        x, t = path2link(p, enhence)
        tt = torch.dot(x, t)
        obj = tt + gamma * torch.dot(cash, enhence ** 2)
        obj.backward()
        grad = enh.grad * 1.0
        grad[enh == 0] = 0
        
        with torch.no_grad():
            enh_pre = enh * 1.0
            enh -= alpha * enh.grad
            enh.clamp_(0)
            direction = enh - enh_pre
            descent = torch.dot(direction / alpha, -enh.grad)
        enh.grad.zero_()
        
        if iter_num > 0 and jjj == 0 and torch.norm(grad) ** 2 < 1e-4:
            converge = True
            break
        
        if jjj > 0 and torch.norm(grad) ** 2 < 1e-4:
            break
        jjj += 1
        

    iter_num += 1
    Traj_ioa.append(obj.detach())
    Time_ioa.append(time.time() - tic)
    
    if converge:
        break
    

Enh[-2, edge_enh] = enh.detach()
P[-2, :] = p_0.detach()

    
"""Cournot Models"""
T_list = [0, 1, 3, 6, 10]

Traj_c_data = list()
Time_c_data = list()
for jj, T in enumerate(T_list):
    Traj_c = list()
    Time_c = list()
    enh = torch.zeros(len(edge_enh), dtype=torch.double)
    enh.requires_grad_()
    
    p_0 = torch.ones(path_number, dtype=torch.double)
    p_0 /= path_demand.t() @ (path_demand @ p_0)
    enhence = torch.zeros_like(cap)
    enhence[edge_enh] = enh
    x, t = path2link(p_0, enhence)
    tt = torch.dot(x, t)
    obj = tt + gamma * torch.dot(cash, enhence ** 2)
    Traj_c.append(obj.detach())
    Time_c.append(0)    
    tic = time.time()
    
    iter_num = 0
    while iter_num <= N_max:
        
        enhence = torch.zeros_like(cap)
        enhence[edge_enh] = enh
        
    
        p = p_0 * 1.0
        i = 0
        while i < T:
            p = h(p, enhence)
            i += 1
        x, t = path2link(p, enhence)
        tt = torch.dot(x, t)
        obj = tt + gamma * torch.dot(cash, enhence ** 2)
        Traj_c.append(obj.detach())
        Time_c.append(time.time() - tic)
        obj.backward()
        grad = enh.grad * 1.0
        with torch.no_grad():
            
            f = q * p_0
            x = path_edge @ f
            t = tfree * (1 + 0.15 * (x / (cap + enhence)) ** 4)
            c = path_edge.t() @ t
            
            equilibrium_gap = compute_gap(c, f, demand)
        
        if torch.norm(grad[enh > 0]) ** 2 < 1e-4 and equilibrium_gap < 1e-8:
            break
             
        with torch.no_grad():
            enh_pre = enh * 1.0
            enh -= alpha * enh.grad
            enh.clamp_(0)
            direction = enh - enh_pre
            descent = torch.dot(direction / alpha, -enh.grad)
        enh.grad.zero_()
        
        with torch.no_grad():
            
            
            p_0 *= torch.exp(-r * c)
            p_0 /= path_demand.t() @ (path_demand @ p_0)
                   
        iter_num += 1
        grad[enh == 0] = 0        
    i = 0
    while i < T:
        p_0 = h(p_0, enhence)
        i += 1  
    
    Enh[jj, edge_enh] = enh.detach()
    P[jj, :] = p_0.detach()

    x, t = path2link(p_0, enhence)
    tt = torch.dot(x, t)
    obj = tt + gamma * torch.dot(cash, enhence ** 2)
    Traj_c.append(obj.detach())
    Time_c.append(time.time() - tic)
    Traj_c_data.append(torch.tensor(Traj_c))
    Time_c_data.append(torch.tensor(Time_c))
    

"""Stackelberg Models"""
Traj_ad = list()
Time_ad = list()

enh = torch.zeros(len(edge_enh), dtype=torch.double)
enh.requires_grad_()
p_0 = torch.ones(path_number, dtype=torch.double)
p_0 /= path_demand.t() @ (path_demand @ p_0)

enhence = torch.zeros_like(cap)
enhence[edge_enh] = enh
x, t = path2link(p_0, enhence)
tt = torch.dot(x, t)
obj = tt + gamma * torch.dot(cash, enhence ** 2)

Traj_ad.append(obj.detach())
Time_ad.append(0)
tic = time.time()


iter_num = 0
while iter_num <= N_max:
    enhence = torch.zeros_like(cap)
    enhence[edge_enh] = enh
    
    p = p_0 * 1.0
    q = path_demand.t() @ demand
    while True:
        f = q * p
        x = path_edge @ f
        t = tfree * (1 + 0.15 * (x / (cap + enhence)) ** 4)
        c = path_edge.t() @ t
        with torch.no_grad():
            gap = compute_gap(c, f, demand)
            if gap < 1e-8:
                break
        
        p *= torch.exp(-r * c)
        p /= path_demand.t() @ (path_demand @ p)
       
    x, t = path2link(p, enhence)
    tt = torch.dot(x, t)
    enhence = torch.zeros_like(cap)
    enhence[edge_enh] = enh
    obj = tt + gamma * torch.dot(cash, enhence ** 2)
    Traj_ad.append(obj.detach())
    Time_ad.append(time.time() - tic)
    obj.backward()
    grad = enh.grad * 1.0

    grad[enh == 0] = 0
    if iter_num > 0 and torch.norm(grad) ** 2 < 1e-4:
        break
    
    with torch.no_grad():
        enh -= alpha * enh.grad
        enh.clamp_(0)
    enh.grad.zero_()
    
    iter_num += 1
    Traj_ad.append(obj.detach())
    Time_ad.append(time.time() - tic)
 
Traj_ad = torch.tensor(Traj_ad)
Time_ad = torch.tensor(Time_ad)
Enh[-3, edge_enh] = enh.detach()
P[-3, :] = p_0.detach()


Enh = Enh.numpy()
Enh = np.around(Enh, decimals=3)
P = P.numpy()
P = np.around(P, decimals=4)

N_algorithm = len(T_list) + 2
Algorithm = torch.arange(N_algorithm)
Value = torch.zeros(N_algorithm)
Value[-2] = Traj_ad[-1]
Value[-1] = Traj_ioa[-1]

CPU_time = torch.zeros(N_algorithm)
CPU_time[-1] = Time_ad[-1]
CPU_time[0] = Time_ioa[-1]

Iteration = torch.zeros(N_algorithm)
Iteration[-1] = len(Time_ad)
Iteration[0] = len(Time_ioa)

for ii in range(len(T_list)):
    Value[ii] = Traj_c_data[ii][-1]
    CPU_time[ii + 1] = Time_c_data[ii][-1]
    Iteration[ii + 1] = len(Time_c_data[ii])  
Timepiter = CPU_time / Iteration

Data = torch.stack([CPU_time, Iteration, Timepiter * 1000]).numpy()


Value = Value.numpy()

"Display the Results"
Table1 = PrettyTable()
Table1.field_names = ['Method', 'Objective', 
                         'a = 1', 'a = 2', 'a = 3', 'a = 4', 'a = 5',
                         'k = 1', 'k = 2', 'k = 3']
Table1.add_rows(
        [
                ["S-0"] + [round(Value[0], 4)] + list(Enh[0]) + list(P[0]),
                ["S-1"] + [round(Value[1], 4)] + list(Enh[1]) + list(P[1]),
                ["S-3"] + [round(Value[2], 4)] + list(Enh[2]) + list(P[2]),
                ["S-6"] + [round(Value[3], 4)] + list(Enh[3]) + list(P[3]),
                ["S-10"] + [round(Value[4], 4)] + list(Enh[4]) + list(P[4]),
                ["D"] + [round(Value[5], 4)] + list(Enh[5]) + list(P[5]),
                ["IOA"] + [round(Value[6], 4)] + list(Enh[6]) + list(P[6]),
                ["SO"] + [round(obj_lower, 4)] + list(Enh[7]) + list(P[7])
                ]
        )

print('Capacity enhancement and routing solutions ')
print(Table1)
print('')
Value = np.around(Value, decimals=4)
CPU_time = CPU_time.numpy()
CPU_time = np.around(CPU_time, decimals=3)

Iteration = Iteration.numpy()
Iteration = np.around(Iteration, decimals=0)
Iteration = Iteration.astype(int)

Timepiter = Timepiter.numpy() * 1000
Timepiter = np.around(Timepiter, decimals=2)


Table2 = PrettyTable()
Table2.field_names = ['', 'IOA', 'S-0', 'S-1', 'S-3', 'S-6', 'S-10', 'D']
Table2.add_rows(
        [
                ["Objective value"] + list(Value),
                ["Total CPU Ttme"] + list(CPU_time),
                ["Iteration number"] + ['-'] + list(Iteration[1:]),
                ["Time per iteration"] + ['-'] + list(Timepiter[1:])
                ]
        )
print('Computational performance of the tested algorithms')
print(Table2)
