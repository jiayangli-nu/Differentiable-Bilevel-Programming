# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:25:46 2024

@author: Jiayang Li
"""

import torch
from read import read_notoll


class Instance():
    
    def __init__(self, NETWORK):
        
        if NETWORK == 'hearn1' or NETWORK == 'hearn2':
        
            node_count, link_count, flow_count, \
            Innode, Outnode, Link, Odtree, Nodeloc, \
            Capacity, Length, Fftime, Demand = read_notoll('hearn')
        else:
            node_count, link_count, flow_count, \
            Innode, Outnode, Link, Odtree, Nodeloc, \
            Capacity, Length, Fftime, Demand = read_notoll(NETWORK)
            
        
        if NETWORK == 'hearn1':
            demand = torch.tensor(list(Demand.values()), dtype=torch.double)
            tfree = torch.tensor(list(Fftime.values()), dtype=torch.double)
            cap_original = torch.tensor(list(Capacity.values()), dtype=torch.double)
            toll_link = [1, 10, 11]
            # toll_link = [10, 11]
            epsilon = 1e-6
            xi = 1e-4
            eps_eva = 1e-8
            
        if NETWORK == 'hearn2':
            demand = torch.tensor(list(Demand.values()), dtype=torch.double)
            tfree = torch.tensor(list(Fftime.values()), dtype=torch.double)
            cap_original = torch.tensor(list(Capacity.values()), dtype=torch.double)
            toll_link = [10, 11]
            epsilon = 1e-6
            xi = 1e-4
            eps_eva = 1e-8
        
        if NETWORK == 'sf':
            
            demand = torch.tensor(list(Demand.values()), dtype=torch.double) / 1000
            Demand = dict(zip(Demand.keys(), [float(d) for d in demand]))
            tfree = torch.tensor(list(Fftime.values()), dtype=torch.double) / 0.1 * 0.06
            Fftime = dict(zip(Link, [float(t) for t in tfree]))
            cap_original = torch.tensor(list(Capacity.values()), dtype=torch.double) / 1000
            Capacity = dict(zip(Link, [float(c) for c in cap_original]))
            toll_link = [11, 14, 21, 32, 35, 38, 41, 45, 46, 48, 51, 52, 57, 64, 66, 68, 70, 73]
            epsilon = pow(10, -6)
            xi = 1e-4
            
            eps_eva = 1e-6
            
        elif NETWORK == "bar":
            demand = torch.tensor(list(Demand.values()), dtype=torch.double) * 2 / 1000
            Demand = dict(zip(Demand.keys(), [float(d) for d in demand]))
            tfree = torch.tensor(list(Fftime.values()), dtype=torch.double)
            cap_original = torch.tensor(list(Capacity.values()), dtype=torch.double) / 1000
            Capacity = dict(zip(Link, [float(c) for c in cap_original]))  
            toll_link = [1514, 1464, 2265, 2517, 2274, 2062,  677,  704,  644,  674,  649, 2176,
                         1752, 1635, 2484, 1652, 1531, 1782, 2007, 1892]
            
            epsilon = pow(10, -4)
            xi = 1e-4
            
            eps_eva = 1e-6
            
        elif NETWORK == "cs":
            demand = torch.tensor(list(Demand.values()), dtype=torch.double) * 2 / 1000
            Demand = dict(zip(Demand.keys(), [float(d) for d in demand]))
            tfree = torch.tensor(list(Fftime.values()), dtype=torch.double)
            cap_original = torch.tensor(list(Capacity.values()), dtype=torch.double) / 1000
            Capacity = dict(zip(Link, [float(c) for c in cap_original]))
            toll_link = [658,  566,  800, 1082,  810,  646,  654,  849, 1053,  807,  869,  857,
                         771,  789,  805,  847,  775,  650,  966,  801,  865,  796,  852,  621,
                         630,  853,  755,  861,  642,  638,  634,  782,  813,  856,  783,  561,
                         655,  662,  779,  860]
            
            epsilon = pow(10, -4)
            xi = 1e-4
            
            eps_eva = 1e-6
            
        Instance.Capacity = Capacity
        Instance.Length = Length
        Instance.Fftime = Fftime
        Instance.Demand = Demand
        
        Instance.demand = demand
        Instance.cap = cap_original
        Instance.tfree = tfree
        Instance.toll_link = toll_link
        Instance.epsilon = epsilon
        Instance.xi = xi
        Instance.eps_eva = eps_eva
        Instance.Link = Link
        
        