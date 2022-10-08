#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import networkx as nx
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

city = 'chicagosketch'

with open('Network/' + city + '.location', 'r') as f:
    location_data = f.readlines()
location_data = [line.split() for line in location_data]
X = [float(location[1]) / 1e+6 for location in location_data]
Y = [float(location[2]) / 1e+6 for location in location_data]
node_number = len(location_data)

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
demand_a = gamma * demand
demand_h = (1 - gamma) * demand
q_a = path_demand.t() @ demand_a
q_h = path_demand.t() @ demand_h

"""solving for UE"""
print("Solving for the UE when no CAVs are controlled")
p_a = torch.ones(path_number, dtype=torch.double).to(device)
p_a /= path_demand.t() @ (path_demand @ p_a)

p_h = torch.ones(path_number, dtype=torch.double).to(device)
p_h /= path_demand.t() @ (path_demand @ p_h)
    

p_a = torch.ones(path_number, dtype=torch.double).to(device)
p_h = torch.ones(path_number, dtype=torch.double).to(device)
p_a /= path_demand.t() @ (path_demand @ p_a)
p_h /= path_demand.t() @ (path_demand @ p_h)


i = 0
r = 0.25
T_max = 10000
while i < T_max:
    f_a = q_a * p_a
    x_a = path_edge @ f_a
    f_h = q_h * p_h
    x_h = path_edge @ f_h
    x = x_a + x_h
    r_a = torch.nan_to_num(x_a / x)
    cap_eff = cap * (1 + r_a ** 2)
    t = tfree * (1 + 0.15 * (x / cap_eff) ** 4)
    u_a = t
    u_h = t
    c_a = path_edge.t() @ u_a
    c_h = path_edge.t() @ u_h
    
    """compute_gap"""
    TT_a = torch.dot(x_a, t)
    TT_h = torch.dot(x_h, t)
    c_mat[d_number, p_number] = c_a
    GT_a = torch.dot(torch.min(c_mat, 1)[0], demand_a)
    c_mat[d_number, p_number] = c_h
    GT_h = torch.dot(torch.min(c_mat, 1)[0], demand_h)
    gap_a = (TT_a - GT_a) / TT_a
    gap_h = (TT_h - GT_h) / TT_h
    gap = max(gap_a, gap_h)
    if gap < 5 * 1e-3:
        break    
    
    p_h *= torch.exp(-r * c_h)
    p_h /= path_demand.t() @ (path_demand @ p_h)
    p_a *= torch.exp(-r * c_a)
    p_a /= path_demand.t() @ (path_demand @ p_a)
    i += 1


"""Stackelberg routing"""
print("Performing sensitivity analysis")
p_a0 = p_a.detach()  
p_h = torch.ones(path_number, dtype=torch.double).to(device)
p_h /= path_demand.t() @ (path_demand @ p_h)

p_a0.requires_grad_()
p_a = p_a0 * 1.0
f_a = q_a * p_a
x_a = path_edge @ f_a
    
i = 0
r = 0.25
T_max = 1000
while i < T_max:
    
    f_h = q_h * p_h
    x_h = path_edge @ f_h
    x = x_a + x_h
    r_a = torch.nan_to_num(x_a / x)
    cap_eff = cap * (1 + r_a ** 2)
    t = tfree * (1 + 0.15 * (x / cap_eff) ** 4)
    u_a = t
    u_h = t
    c_a = path_edge.t() @ u_a
    c_h = path_edge.t() @ u_h
    
    with torch.no_grad():
        TT_h = torch.dot(x_h, t)
        c_mat[d_number, p_number] = c_h
        GT_h = torch.dot(torch.min(c_mat, 1)[0], demand_h)
        gap_h = (TT_h - GT_h) / TT_h
    if gap_h < 5 * 1e-3:
        break
    p_h *= torch.exp(-r * c_h)
    p_h /= path_demand.t() @ (path_demand @ p_h)
    p_a *= torch.exp(-r * c_a)
    p_a /= path_demand.t() @ (path_demand @ p_a)
    i += 1
f_h = q_h * p_h
x_h = path_edge @ f_h
x = x_a + x_h
r_a = torch.nan_to_num(x_a / x)
cap_eff = cap * (1 + r_a ** 2)
t = tfree * (1 + 0.15 * (x / cap_eff) ** 4)
TT = torch.dot(x, t)
TT_star = TT.detach() * 1.0
TT.backward()
grad_pa = p_a0.grad

torch.save(grad_pa, city + '.grad_pa.pt')


print("Performing an one-step mirror descent update for each OD pair")
decrease = torch.zeros_like(demand)
with torch.no_grad():
    for i in range(len(pool)):
        if i % 5000 == 0:
            print("Completed " + str(i) + " / " + str(len(pool)))
        grad = torch.zeros_like(grad_pa)
        grad[pool[i]] = grad_pa[pool[i]]
        p_new = p_a0 * torch.exp(-1e-4 * grad)
        p_new[pool[i]] /= torch.sum(p_new[pool[i]])
        direction = p_new - p_a0

        decrease[i] = torch.dot(direction, -grad)     
torch.save(decrease, city + '.decrease.pt')

K = 100
top = torch.topk(decrease, K)[1]

print("Visualizing OD pairs with toppest potential to reduce congestion after controlled")
G = nx.DiGraph()
for i in range(node_number):
    G.add_node(i, pos=(X[i], Y[i]))
G.add_edges_from(zip(s_node, t_node))
pos_G = nx.get_node_attributes(G, 'pos')

plt.figure(figsize=(4, 5.5))
ax = plt.gca()
ax.spines['top'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
options = {
        "arrows": False,
        "node_size": 0.1,
        "node_color": 'grey',
        "edge_color": 'grey',
        "width": 0.25,
        }

nx.draw(G, pos_G, **options)

for kk in range(2):
    if kk == 0:
        topkk = top[:10]
    if kk == 1:
        topkk = top[10:20]
    start_node = [a_node[i] for i in topkk]
    end_node = [b_node[i] for i in topkk]
    arrow_node = set(start_node + end_node)
    H = nx.DiGraph()
    for k in arrow_node:
        H.add_node(k, pos=(X[k], Y[k]))
    H.add_edges_from(zip(start_node, end_node))
    pos_H = nx.get_node_attributes(H, 'pos')
    if kk == 0:
        color = 'red'
        width = 0.75
    if kk == 1:
        color = 'blue'
        width = 0.5
    options = {
            "arrows": True,
            "node_size": 0.5,
            "node_shape": '.',
            "node_color": color,
            "edge_color": color,
            "width": width,
            "style": '-',
            "arrowsize": 5,
            }
    nx.draw(H, pos_H, **options)

plt.xlim([0.355, 0.845])
plt.ylim([1.585, 2.225])

plt.savefig('Result-H-1.pdf', bbox_inches='tight', dpi=200)
plt.show()
