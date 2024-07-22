# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:25:46 2024

@author: Jiayang Li
"""
import torch
import pickle
from graph import Network
from read import read_notoll


for NETWORK in ['sf']:
        
    node_count, link_count, flow_count, \
    Innode, Outnode, Link, Odtree, Nodeloc, \
    Capacity, Length, Fftime, Demand = read_notoll(NETWORK)
    net = Network(NETWORK,Innode, Outnode, Link, Odtree)

    
    if NETWORK == 'hearn':
        epsilon = pow(10, -10)
        
    elif NETWORK == 'sf':   
        demand = torch.tensor(list(Demand.values()), dtype=torch.double) / 1000
        Demand = dict(zip(Demand.keys(), [float(d) for d in demand]))
        tfree = torch.tensor(list(Fftime.values()), dtype=torch.double) / 0.1 * 0.06
        Fftime = dict(zip(Link, [float(t) for t in tfree]))
        cap_original = torch.tensor(list(Capacity.values()), dtype=torch.double) / 1000
        Capacity = dict(zip(Link, [float(c) for c in cap_original]))
        epsilon = pow(10, -6)
     
    elif NETWORK == "bar":
        demand = torch.tensor(list(Demand.values()), dtype=torch.double) * 2 / 1000
        Demand = dict(zip(Demand.keys(), [float(d) for d in demand]))
        tfree = torch.tensor(list(Fftime.values()), dtype=torch.double)
        cap_original = torch.tensor(list(Capacity.values()), dtype=torch.double) / 1000
        Capacity = dict(zip(Link, [float(c) for c in cap_original]))  
        epsilon = pow(10, -5)
        
        
    elif NETWORK == "cs":
        demand = torch.tensor(list(Demand.values()), dtype=torch.double) * 2 / 1000
        Demand = dict(zip(Demand.keys(), [float(d) for d in demand]))
        tfree = torch.tensor(list(Fftime.values()), dtype=torch.double)
        cap_original = torch.tensor(list(Capacity.values()), dtype=torch.double) / 1000
        Capacity = dict(zip(Link, [float(c) for c in cap_original]))
        epsilon = pow(10, -4)
        
    net.solve_ue(Demand, Fftime, Capacity, 1000, epsilon, warm_start=False)
    
    net.path_enumeration()
    net.generate_sparse_matrix()
    
    with open('result_sctp/net_' + NETWORK + '.pkl', 'wb') as f:
        pickle.dump(net, f)
        print(NETWORK + ': finished')
                        
    net.solve_ue(Demand, Fftime, Capacity, 1000, epsilon, warm_start=True)
    x_ue = torch.tensor(list(net.flow.values()), dtype=torch.double)
    torch.save(x_ue, 'result_sctp/ue_' + NETWORK + '.pt')
    net.solve_so(Demand, Fftime, Capacity, 1000, epsilon, warm_start=False)
    x_so = torch.tensor(list(net.flow.values()), dtype=torch.double)
    torch.save(x_so, 'result_sctp/so_' + NETWORK + '.pt')
    