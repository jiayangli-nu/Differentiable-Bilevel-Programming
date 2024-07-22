#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.optimize import minimize, root


def compute_gap(c, f, demand):
    c_max = torch.max(c) + 1
    c_min = c_max - torch.max((c_max - path_demand * c) * path_demand, 1)[0]
    gt = torch.dot(c_min, demand)
    tt = torch.dot(c, f)
    gap = tt - gt
    return gap / tt
    

Result = np.zeros((8, 9), dtype=np.double)

path_edge = torch.tensor([[1, 0, 1, 0, 0],
                          [1, 0, 0, 1, 1],
                          [0, 1, 0, 0, 1]], dtype=torch.double).t()
path_demand = torch.tensor([[1, 1, 1]], dtype=torch.double)
tfree = torch.tensor([1, 3, 3, 0.5, 1], dtype=torch.double)
cap = torch.tensor([2, 4, 4, 1, 2], dtype=torch.double)
demand = torch.tensor([6], dtype=torch.double)
path_number = path_edge.size()[1]
q = path_demand.t() @ demand
cash = tfree
gamma = 1
xi = 1e-10


"""exact"""

r = 0.5

def eq_searc(p1, enh):
    p3 = p1
    p2 = 1 - p1 - p3
    f1 = p1 * 6
    f2 = p2 * 6
    f3 = p3 * 6
    x1 = f1 + f2
    x3 = f1
    x4 = f2
    x5 = f2 + f3
    t1 = 1 * (1 + 0.15 * (x1 / (2 + enh[0])) ** 4)
    t3 = 3 * (1 + 0.15 * (x3 / (4 + enh[2])) ** 4)
    t4 = 0.5 * (1 + 0.15 * (x4 / (1 + enh[3])) ** 4)
    t5 = 1 * (1 + 0.15 * (x5 / (2 + enh[4])) ** 4)
    c1 = t1 + t3
    c2 = t1 + t4 + t5
    return c1 - c2


def equilibrium_solving(enh):
    
    p1 = 0
    p3 = p1
    p2 = 1
    p = torch.tensor([p1, p2, p3], dtype=torch.double)
    x = path_edge @ (p * 6)
    t = tfree * (1 + 0.15 * (x / (cap + enh)) ** 4)
    c = path_edge.t() @ t
    
    if c[1] < c[0]:
        return p
    
    p1 = 0.5
    p3 = p1
    p2 = 0
    p = torch.tensor([p1, p2, p3], dtype=torch.double)
    x = path_edge @ (p * 6)
    t = tfree * (1 + 0.15 * (x / (cap + enh)) ** 4)
    c = path_edge.t() @ t
    
    if c[0] < c[1]:
        
        return p
    
    
    eq_searc_1 = lambda p1: eq_searc(p1, enh)
    p1 = float(root(eq_searc_1, 0.5).x)
    p3 = p1
    p2 = 1 - p1 - p3
    p = torch.tensor([p1, p2, p3], dtype=torch.double)
    print(p)
    return p



enh = np.array([0.93, 0.02, 0.02, 0, 0.93], dtype=np.double)


p = equilibrium_solving(enh)
x_original = path_edge @ (p * 6)


def obj(decision): 
    
    enh = decision[:len(cap)]
    x = decision[len(cap):]
    t = tfree.numpy() * (1 + 0.15 * (x / (cap.numpy() + enh)) ** 4)
    TT = np.dot(x, t)
    return TT + np.dot(cash, enh ** 2)

def link_cost(decision): 
        
    enh = decision[:len(cap)]
    x = decision[len(cap):]
    t = tfree.numpy() * (1 + 0.15 * (x / (cap.numpy() + enh)) ** 4)

    return t

def eq_constraint(decision):
    
    enh1 = decision[0]
    enh2 = decision[1]
    enh3 = decision[2]
    enh5 = decision[4]
    
    x1 = decision[5]
    x2 = decision[6]
    x3 = decision[7]
    x4 = decision[8]
    x5 = decision[9]
    
    l1 = x1 + x2 - 6
    l2 = x1 - x3 - x4
    l3 = x2 + x4 - x5
    l4 = enh1 - enh5
    l5 = enh2 - enh3
    return np.array([l1, l2, l3, l4, l5])


enh = torch.zeros_like(cap)
initial_guess = torch.cat([enh, x_original]).numpy()
bounds = [(0, None) for _ in range(len(initial_guess))]
bounds[3] = (0, 0)

p1 = np.array([1, 0, 0], dtype=np.double)
x1 = path_edge.numpy() @ (6 * p1)
p2 = np.array([0, 1, 0], dtype=np.double)
x2 = path_edge.numpy() @ (6 * p2)
p3 = np.array([0, 0, 1], dtype=np.double)
x3 = path_edge.numpy() @ (6 * p3)

neq_constraint1 = lambda decision: np.dot(link_cost(decision), x1 - decision[len(cap):])
neq_constraint2 = lambda decision: np.dot(link_cost(decision), x2 - decision[len(cap):])
neq_constraint3 = lambda decision: np.dot(link_cost(decision), x3 - decision[len(cap):])


constraints = [{'type': 'eq', 'fun': eq_constraint},
                {'type': 'ineq', 'fun': neq_constraint1},
                {'type': 'ineq', 'fun': neq_constraint2},
                {'type': 'ineq', 'fun': neq_constraint3}]

result = minimize(obj, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP', tol=1e-20)
enh_exact = result.x[:len(cap)]
p_exact = equilibrium_solving(enh_exact)
x = path_edge @ (p * 6)
t = tfree * (1 + 0.15 * (x / (cap + enh_exact)) ** 4)
tt = np.dot(x, t)
leadercost_exact = tt + gamma * np.dot(cash, enh_exact ** 2)

Result[0, 0] = leadercost_exact
Result[0, 1:6] = enh_exact
Result[0, 6:] = p_exact

"""so"""



enh = torch.zeros_like(cap)
initial_guess = torch.cat([enh, x_original]).numpy()
bounds = [(0, None) for _ in range(len(initial_guess))]

constraints = [{'type': 'eq', 'fun': eq_constraint}]
result = minimize(obj, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP', tol=1e-20)

enh_so = result.x[:len(cap)]
p_so = equilibrium_solving(enh_so)
x = path_edge @ (p_so * 6)
t = tfree * (1 + 0.15 * (x / (cap + enh_so)) ** 4)
c = path_edge.t() @ t
tt = np.dot(x, t)
leadercost_so = tt + gamma * np.dot(cash, enh_so ** 2)

gap_so = (leadercost_so - leadercost_exact) / leadercost_exact

Result[2, 0] = leadercost_so
Result[2, 1:6] = enh_so
Result[2, 6:] = p_so



"""ioa"""
enh = np.zeros(len(cap), dtype=np.double)
obj_old = torch.inf
enh_old = enh * 1.0 


p = equilibrium_solving(enh)
x = path_edge @ (p * 6) 
while True:
        
    def obj_sub(enh):
        enh[enh == 0] = 1e-10
        t = tfree * (1 + 0.15 * (x / (cap + enh)) ** 4)
        tt = np.dot(x, t)
        return tt + gamma * np.dot(cash, enh ** 2)

    initial_guess = enh * 1.0
    bounds = [(0, None) for _ in range(len(enh))]
    result = minimize(obj_sub, initial_guess, bounds=bounds)
    enh = result.x
    
    
    obj = obj_sub(enh)

    if max(abs(enh - enh_old)) < xi:
        break
    obj_old = obj
    enh_old = enh * 1.0
    
    
    p = equilibrium_solving(enh)
    x = path_edge @ (p * 6)

enh_ioa = enh * 1.0
p_ioa = equilibrium_solving(enh_ioa)
x = path_edge @ (p_ioa * 6)
t = tfree * (1 + 0.15 * (x / (cap + enh_ioa)) ** 4)
tt = np.dot(x, t)
leadercost_ioa = tt + gamma * np.dot(cash, enh_ioa ** 2)

gap_ioa = (leadercost_ioa - leadercost_exact) / leadercost_exact

Result[1, 0] = leadercost_ioa
Result[1, 1:6] = enh_ioa
Result[1, 6:] = p_ioa


Eps = 10 ** torch.arange(-6, -2.5, step=0.5, dtype=torch.double)
T_list = [1, 2, 3, 4, 5]




for xi in [1e-5]:

    """Dol-MD"""
    r = 0.1
    alpha = 0.005
    
    
    Gap_dol = list()
    for epsilon in Eps:
        enh = torch.zeros(len(cap), dtype=torch.double)
        enh.requires_grad_()
        p_0 = torch.ones(path_number, dtype=torch.double)
        p_0 /= path_demand.t() @ (path_demand @ p_0)
        
        iter_num = 0
        enh_old = enh * 1.0
        while iter_num <= 100000:
            
            p = p_0 * 1.0
            q = path_demand.t() @ demand
            while True:
                f = q * p
                x = path_edge @ f
                t = tfree * (1 + 0.15 * (x / (cap + enh)) ** 4)
                c = path_edge.t() @ t
                with torch.no_grad():
                    gap = compute_gap(c, f, demand)
                    if gap < epsilon:
                        break
                
                p *= torch.exp(-r * c)
                p /= path_demand.t() @ (path_demand @ p)
               
            tt = torch.dot(x, t)
            obj = tt + gamma * torch.dot(cash, enh ** 2)
            obj.backward()
            # grad = enh.grad * 1.0
        
            # grad[enh == 0] = 0
            # if iter_num > 0 and torch.norm(grad) ** 2 < xi:
            #     break
            
            
            with torch.no_grad():
                enh -= alpha * enh.grad
                enh.clamp_(0)
            enh.grad.zero_()
            
            if max(abs(enh - enh_old)) < xi:
                break
            enh_old = enh * 1.0
            
            iter_num += 1
        else:
            print('does not converge')
        
        enh = enh.detach().numpy()
        p = equilibrium_solving(enh)
        x = path_edge @ (p * 6)
        t = tfree * (1 + 0.15 * (x / (cap + enh)) ** 4)
        tt = np.dot(x, t)
        leadercost = tt + gamma * np.dot(cash, enh ** 2)
        
        if epsilon == Eps[-1]:
            Result[3, 0] = leadercost
            Result[3, 1:6] = enh
            Result[3, 6:] = p
        if epsilon == 1e-4:
            Result[4, 0] = leadercost
            Result[4, 1:6] = enh
            Result[4, 6:] = p
        
        
        gap = (leadercost - leadercost_exact) / leadercost_exact
        print(leadercost, gap)
        Gap_dol.append(gap)
    
    
    """SAB"""
    alpha = 0.005
    Gap_sab = list()
    for epsilon in Eps:
        enh = torch.zeros(len(cap), dtype=torch.double)
        enh.requires_grad_()
        
        
        p_0 = torch.ones(path_number, dtype=torch.double)
        p_0 /= path_demand.t() @ (path_demand @ p_0)
        
        iter_num = 0
        enh_old = enh * 1.0
        while iter_num <= 100000:
            with torch.no_grad():
                p = p_0 * 1.0
                q = path_demand.t() @ demand
                while True:
                    f = q * p
                    x = path_edge @ f
                    t = tfree * (1 + 0.15 * (x / (cap + enh)) ** 4)
                    c = path_edge.t() @ t
                    with torch.no_grad():
                        gap = compute_gap(c, f, demand)
                        if gap < epsilon:
                            break
                    
                    p *= torch.exp(-r * c)
                    p /= path_demand.t() @ (path_demand @ p)
           
        
    
                cap_new = cap + enh
                u_x = 0.15 * tfree * 4 * (x / cap_new) ** 3 / cap_new
                u_cap = -0.15 * tfree * 4 * (x / cap_new) ** 3 * x / cap_new ** 2
            
                c_f = torch.mm(path_edge.t(), torch.diag(u_x))
                c_f = torch.mm(c_f, path_edge)
            
                c_cap = torch.mm(path_edge.t(), torch.diag(u_cap))
            
            
                J_up = torch.cat([c_f, -path_demand.t()], 1)
                J_down = torch.cat([path_demand, torch.zeros(len(demand), len(demand))], 1)
                J = torch.cat([J_up, J_down], 0)
                invJ = torch.inverse(J)
            
                Right = torch.cat([-c_cap, torch.zeros(len(demand), len(cap))], 0)
            
                f_cap = torch.mm(invJ, Right)[:len(f), :]
    
                x_cap = torch.mm(path_edge, f_cap)
            
                x_exp = x_cap * 1.0
        
        
            cap_new = cap + enh
            x_vir = x_exp @ enh - (x_exp @ enh - x).detach()
            t_vir = tfree * (1 + 0.15 * (x_vir / cap_new) ** 4)
            TT = torch.dot(x_vir, t_vir) 
            objective = TT + gamma * torch.dot(cash, enh ** 2)
            objective.backward()
    
            
            
            
            with torch.no_grad():
                enh -= alpha * enh.grad
                enh.clamp_(0)
            enh.grad.zero_()
            
            if max(abs(enh - enh_old)) < xi:
                break
            enh_old = enh * 1.0
            
            iter_num += 1
        else:
            print('does not converge')
            
        enh = enh.detach().numpy()
        p = equilibrium_solving(enh)
        x = path_edge @ (p * 6)
        t = tfree * (1 + 0.15 * (x / (cap + enh)) ** 4)
        tt = np.dot(x, t)
        leadercost = tt + gamma * np.dot(cash, enh ** 2)
        
        gap = (leadercost - leadercost_exact) / leadercost_exact
        
        print(leadercost, gap)
        Gap_sab.append(gap)
    
    
    """Sil-MD"""
   
    Gap_sil = list()
    r = 0.25
    alpha = 0.005
    p_0 = torch.ones(path_number, dtype=torch.double)
    p_0 /= path_demand.t() @ (path_demand @ p_0)
    enh = torch.zeros(len(cap), dtype=torch.double)
    enh.requires_grad_()
    for jj, T in enumerate(T_list):
        print(T)
        enh = torch.zeros_like(cap)
        enh.requires_grad_()
        
        
        iter_num = 0
        enh_old = enh * 1.0
        while iter_num <= 1000000:
            
            p = p_0 * 1.0
            i = 0
            while i < T:
                f = q * p
                x = path_edge @ f
                t = tfree * (1 + 0.15 * (x / (cap + enh)) ** 4)
                c = path_edge.t() @ t
                p *= torch.exp(-r * c)
                p /= path_demand.t() @ (path_demand @ p)
                i += 1
             
            f = q * p
            x = path_edge @ f
            t = tfree * (1 + 0.15 * (x / (cap + enh)) ** 4)
            tt = torch.dot(x, t)
            obj = tt + gamma * torch.dot(cash, enh ** 2)
            obj.backward()
            grad = enh.grad * 1.0
            with torch.no_grad():
                
                f = q * p_0
                x = path_edge @ f
                t = tfree * (1 + 0.15 * (x / (cap + enh)) ** 4)
                c = path_edge.t() @ t
                equilibrium_gap = compute_gap(c, f, demand)
              
            with torch.no_grad():
                enh -= alpha * enh.grad
                enh.clamp_(0)
    
            enh.grad.zero_()
            
            if max(abs(enh - enh_old)) < xi and equilibrium_gap < 1e-7:
                break
            enh_old = enh * 1.0
            
            with torch.no_grad():
                
                
                p_0 *= torch.exp(-r * c)
                p_0 /= path_demand.t() @ (path_demand @ p_0)
                       
            iter_num += 1
            grad[enh == 0] = 0       
            
            
        enh = enh.detach().numpy()
        p = equilibrium_solving(enh)
        x = path_edge @ (p * 6)
        t = tfree * (1 + 0.15 * (x / (cap + enh)) ** 4)
        tt = np.dot(x, t)
        leadercost = tt + gamma * np.dot(cash, enh ** 2)
        
        gap = (leadercost - leadercost_exact) / leadercost_exact
        
        if T == 0:
            Result[5, 0] = leadercost
            Result[5, 1:6] = enh
            Result[5, 6:] = p
        if T == 1:
            Result[6, 0] = leadercost
            Result[6, 1:6] = enh
            Result[6, 6:] = p
        if T == 5:
            Result[7, 0] = leadercost
            Result[7, 1:6] = enh
            Result[7, 6:] = p
        
        print(leadercost)
        Gap_sil.append(gap)
    

fig = plt.figure(figsize=(5, 1.25), dpi=1000)

ax = plt.subplot(1, 2, 1)    

ax.set_yscale('log')
ax.set_xscale('log')

ax.spines['top'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)


# plt.title('(a) Dol-MD', fontsize=15, y=1.0)
plt.plot(Eps, [gap_ioa for _ in range(len(Eps))], linestyle='--', color='k', label='ioa', linewidth=2)
plt.plot(Eps, [gap_so for _ in range(len(Eps))], linestyle='dotted', color='k', label='so', linewidth=2)
# plt.legend(['SAB', 'Dol-MD', 'IOA', 'SO'], frameon=False)

plt.xlabel(r'$\epsilon$', fontsize=15)
plt.ylabel('opt. gap', fontsize=14)

plt.ylim([1e-6, 5])
plt.xticks([1e-6, 1e-5, 1e-4, 1e-3], fontsize=13)
plt.yticks(fontsize=13)



ax = plt.subplot(1, 2, 2)    
    

ax.set_yscale('log')

ax.spines['top'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)


# plt.title('(b) Sil-MD', fontsize=15, y=1.0)

plt.plot(T_list, [gap_ioa for _ in range(len(T_list))], linestyle='--', color='k', label='ioa', linewidth=2)
plt.plot(T_list, [gap_so for _ in range(len(T_list))], linestyle='dotted', color='k', label='so', linewidth=2)
# plt.legend(['Sil-MD', 'IOA', 'SO'], frameon=False)

plt.xlabel(r'$T$', fontsize=15)
# plt.ylabel('Optimality gap', fontsize=15)
plt.xlim([0.7, 5.3])
plt.ylim([1e-6, 5])
plt.xticks([1, 2, 3, 4, 5], fontsize=13)
plt.yticks([1e-6, 1e-4, 1e-2, 1e0], ['', '', '', ''], fontsize=14)
    
ax = plt.subplot(1, 2, 1)


plt.plot(Eps, Gap_sab, color='dodgerblue', linestyle='--', label='sab', linewidth=2)
plt.plot(Eps, Gap_dol, linestyle='--', color='firebrick', label='dol', linewidth=2)

plt.grid(linestyle='-', linewidth=1, alpha=0.35)

ax = plt.subplot(1, 2, 2)    
   
plt.plot(T_list, Gap_sil, color='forestgreen', linestyle='--', label='sil', linewidth=2)

    
 
line_ioa = mlines.Line2D([], [], color='k', linestyle='--', label='IOA')
line_so = mlines.Line2D([], [], color='k', linestyle='dotted', label='SO')

line_sab1 = mlines.Line2D([], [], color='dodgerblue', linestyle='--', label='SAB')
line_dol1 = mlines.Line2D([], [], color='firebrick', linestyle='--', label='DolMD ')
line_sil1 = mlines.Line2D([], [], color='forestgreen', linestyle='--', label='SilMD')

plt.grid(linestyle='-', linewidth=1, alpha=0.35)


fig.legend(handles=[line_ioa, line_so, line_sab1, line_dol1, line_sil1], loc='lower center', fontsize=13, bbox_to_anchor=(0.5, -0.8), ncol=3, frameon=False)


plt.subplots_adjust(wspace=0.1, hspace=-0.2)


plt.savefig('cndp_braess.pdf', bbox_inches='tight', dpi=1000)
# plt.close()