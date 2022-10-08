#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from torch.autograd.functional import jacobian
from prettytable import PrettyTable


def compute_gap(c, f, demand):
    c_max = torch.max(c) + 1
    c_min = c_max - torch.max((c_max - path_demand * c) * path_demand, 1)[0]
    gt = torch.dot(c_min, demand)
    tt = torch.dot(c, f)
    gap = tt - gt
    return gap / tt


device = torch.device("cpu")

torch.manual_seed(0)

Grad_city = list()
for city_num in range(2):

    if city_num == 0:
        print('Testing Scenario A')
        r = 1
        path_edge = torch.tensor([[1, 0, 1, 0, 0],
                      [1, 0, 0, 1, 1],
                      [0, 1, 0, 0, 1]], dtype=torch.double).t().to(device)
        path_demand = torch.tensor([[1, 1, 1]], dtype=torch.double).to(device)
        tfree = torch.tensor([1, 3, 3, 0.5, 1], dtype=torch.double).to(device)
        cap = torch.tensor([2, 4, 4, 1, 2], dtype=torch.double).to(device)
        demand = torch.tensor([6], dtype=torch.double).to(device)   
        path_number = path_edge.size()[1]
        toll = torch.tensor([0, 0, 0, 0.5, 0], dtype=torch.double).to(device)
        T_max = 275
    elif city_num == 1:
        print('Testing Scenario B')
        r = 1e-4
        path_edge = torch.tensor([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [1, 0, 0, 1],
                      [0, 1, 1, 0]], dtype=torch.double).t()
        path_demand = torch.tensor([[1, 1, 1, 1]], dtype=torch.double)    
        A = torch.tensor([4, 20, 1, 30], dtype=torch.double)
        B = torch.tensor([1, 5, 30, 1], dtype=torch.double)
        tfree = A
        cap = (0.15 * tfree / B) ** 0.25
        time = lambda x: A + B * x ** 4
        demand = torch.tensor([10], dtype=torch.double)
        path_number = path_edge.size()[1]
        toll = torch.zeros_like(tfree)
        T_max = 150
    
    """Exact Equilibrium"""
    q = path_demand.t() @ demand
    if city_num == 0:
        
        p = torch.ones(path_number, dtype=torch.double).to(device)
        p /= path_demand.t() @ (path_demand @ p)
        i = 0
        while True:
            f = q * p
            x = path_edge @ f
            t = tfree * (1 + 0.15 * (x / cap) ** 4)
            u = t + toll
            c = path_edge.t() @ u 
            gap = compute_gap(c, f, demand) 
            p *= torch.exp(-r * c)
            p /= path_demand.t() @ (path_demand @ p)
            i += 1
            if gap < 1e-10:
                break
        p_star = p * 1.0
    if city_num == 1:
        x_star = torch.tensor([6, 4, 3, 7], dtype=torch.double).to(device)
    
    """Exact Gradient"""
    if city_num == 0:
        def h(p, cap):
            f = q * p
            x = path_edge @ f
            t = tfree * (1 + 0.15 * (x / cap) ** 4)
            u = t + toll
            c = path_edge.t() @ u 
            p_new = p * torch.exp(-r * c)
            p_new /= path_demand.t() @ (path_demand @ p_new)
            return p_new
    
        h_p, h_cap = jacobian(h, (p_star, cap))
        p_cap = torch.mm(torch.inverse(torch.eye(len(p)) - h_p), h_cap)
        
        cap.requires_grad_()
        p_vir = p_cap @ cap - (p_cap @ cap - p_star).detach()
        f = demand * p_vir
        x = path_edge @ f
        t = tfree * (1 + 0.15 * (x / cap) ** 4)    
        TT = torch.dot(t, x)
        TT.backward()
        grad_star = cap.grad * 1.0
        cap.grad.zero_()
    if city_num == 1:
        def h(x, cap):
            t = tfree * (1 + 0.15 * (x / cap) ** 4)
            f_var = cp.Variable(len(path_edge))
            x_var = cp.Variable(len(x))
            b_par = cp.Parameter(len(x))
            constraints = [f_var >= 0,
                           path_demand @ f_var == demand,
                           path_edge @ f_var == x_var]
            objective = cp.Minimize(cp.pnorm(x_var - b_par, p=2))
            problem = cp.Problem(objective, constraints)
            assert problem.is_dpp()
            b = x - 0.001 * t
            cvxpylayer = CvxpyLayer(problem, parameters=[b_par], variables=[x_var, f_var])
            return cvxpylayer(b)[0]
        
        h_x, h_cap = jacobian(h, (x_star, cap))
        x_cap = torch.mm(torch.inverse(torch.eye(len(cap)) - h_x), h_cap)
        
        cap.requires_grad_()
        x_vir = x_cap @ cap - (x_cap @ cap - x_star).detach()
        t_vir = tfree * (1 + 0.15 * (x_vir / cap) ** 4)
        TT = torch.dot(x_vir, t_vir)
        TT.backward()
        grad_star = cap.grad * 1.0
        cap.grad.zero_()
    
        
    if city_num == 0:
        T_list = [0, 10, 20]
        Grad = torch.zeros(len(T_list) + 1, len(cap))
    if city_num == 1:
        T_list = [0, 4, 8]
        Grad = torch.zeros(len(T_list) + 1, len(cap))
    for ii, T in enumerate(T_list):
        
        p = torch.ones(path_number, dtype=torch.double).to(device)
        p /= path_demand.t() @ (path_demand @ p)
        
        i = 0
        while i < T:
            f = q * p
            x = path_edge @ f
            t = tfree * (1 + 0.15 * (x / cap) ** 4)
            u = t + toll
            c = path_edge.t() @ u 
            gap = compute_gap(c, f, demand) 
            p *= torch.exp(-r * c)
            p /= path_demand.t() @ (path_demand @ p)
            i += 1
            
        f = q * p
        x = path_edge @ f
        t = tfree * (1 + 0.15 * (x / cap) ** 4)
        TT = torch.dot(t, x)
    
        
        if city_num == 0:
            TT.backward()
            Grad[ii, :] = cap.grad * 1.0
            cap.grad.zero_()
        if city_num == 1:
            TT.backward()  
            Grad[ii, :] = cap.grad * 1.0
            cap.grad.zero_()
    Grad[-1, :] = grad_star    
    Grad_city.append(Grad)

A = Grad_city[0].numpy()
B = Grad_city[1].numpy() / 1e+3
A = np.around(A, decimals=2)
B = np.around(B, decimals=2)

print('')
Table1 = PrettyTable()
Table1.field_names = ["Method", 'link 1', 'link 2', 'link 3', 'link 4', 'link 5']
Table1.add_rows(
        [
                ["AD (T = 0)"] + list(A[0]),
                ["AD (T = 10)"] + list(A[1]),
                ["AD (T = 20)"] + list(A[2]),
                ["ID"] + list(A[3])
                ]
        )
Table2 = PrettyTable()
Table2.field_names = ["Method", 'link 1', 'link 2', 'link 3', 'link 4']
Table2.add_rows(
        [
                ["AD (T = 0)"] + list(B[0]),
                ["AD (T = 4)"] + list(B[1]),
                ["AD (T = 8)"] + list(B[2]),
                ["ID"] + list(B[3])
                ]
        )

print("Scenario A: Braess network")
print(Table1)
print('')
print("Scenario B: 3N4L network")
print(Table2)
