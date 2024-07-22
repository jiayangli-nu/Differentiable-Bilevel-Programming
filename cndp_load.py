# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:25:46 2024

@author: Jiayang Li
"""

import torch
from read import read_notoll


class Instance():
    
    def __init__(self, NETWORK):
        
        node_count, link_count, flow_count, \
        Innode, Outnode, Link, Odtree, Nodeloc, \
        Capacity, Length, Fftime, Demand = read_notoll(NETWORK)

        
        if NETWORK == 'sf':
            
            demand = torch.tensor(list(Demand.values()), dtype=torch.double) / 100 * 0.11
            Demand = dict(zip(Demand.keys(), [float(d) for d in demand]))
            tfree = torch.tensor(list(Fftime.values()), dtype=torch.double) / 0.1 * 0.06
            Fftime = dict(zip(Link, [float(t) for t in tfree]))
            cap_original = torch.tensor(list(Capacity.values()), dtype=torch.double) / 1000
            Capacity = dict(zip(Link, [float(c) for c in cap_original]))
            enh_link = [15, 18, 16, 19, 24, 25, 28, 47, 38, 73]
            cash = torch.tensor([26, 26, 40, 40, 25, 25, 48, 48, 34, 34], dtype=torch.double)
            eta = 0.001
            ratio = 2
            epsilon = pow(10, -5)
            xi = 1e-3
            
            eps_eva = 1e-6
            
        elif NETWORK == "bar":
            demand = torch.tensor(list(Demand.values()), dtype=torch.double) * 2 / 1000
            Demand = dict(zip(Demand.keys(), [float(d) for d in demand]))
            tfree = torch.tensor(list(Fftime.values()), dtype=torch.double)
            cap_original = torch.tensor(list(Capacity.values()), dtype=torch.double) / 1000
            Capacity = dict(zip(Link, [float(c) for c in cap_original]))  
            enh_link = [1416, 2143, 1120, 2484, 2298,  415, 1418, 2269,  410,  616, 2006, 2118,
                        1897, 1456,  540, 1514, 2165, 2281, 2033,  537, 2208, 1106, 1459, 1917,
                        1661, 2517, 2433,  630, 2508, 1548, 2287, 1457, 2276, 2151, 1460, 2274,
                        463, 2009, 1316, 1635, 2109, 1448, 1331,  704,  677, 2482, 1888, 1652,
                        1531,  471]
            
            length = torch.tensor(list(Length.values()), dtype=torch.double)
            cash = length[enh_link]
            eta = 1
            ratio = 2
            epsilon = pow(10, -5)
            xi = 1e-3
            
            eps_eva = 1e-6
            
        elif NETWORK == "cs":
            demand = torch.tensor(list(Demand.values()), dtype=torch.double) * 2 / 1000
            Demand = dict(zip(Demand.keys(), [float(d) for d in demand]))
            tfree = torch.tensor(list(Fftime.values()), dtype=torch.double)
            cap_original = torch.tensor(list(Capacity.values()), dtype=torch.double) / 1000
            Capacity = dict(zip(Link, [float(c) for c in cap_original]))
            enh_link = [964, 2949, 1214, 2126,  944,  594, 2170,  433,  519,  954, 2138, 2778,
                        462,  946,  528, 1798,  961, 1314, 2380,  925, 2925,  760,  530, 1086,
                        1734, 2927, 2173,  447,  575,  750, 1688,  424, 2193,  522,  804,  806,
                        448, 2192,  587,  445,  443,  736, 1227, 1402, 1031, 2119,  955, 1311,
                        2109, 2180]
            
            length = torch.tensor(list(Length.values()), dtype=torch.double)
            cash = length[enh_link]
            eta = 1
            ratio = 2
            epsilon = pow(10, -4)
            xi = 1e-3
            
            eps_eva = 1e-6
            
        Instance.Capacity = Capacity
        Instance.Length = Length
        Instance.Fftime = Fftime
        Instance.Demand = Demand
        
        Instance.demand = demand
        Instance.cap_original = cap_original
        Instance.tfree = tfree
        Instance.enh_link = enh_link
        Instance.cash = cash
        Instance.eta = eta
        Instance.ratio = ratio
        Instance.epsilon = epsilon
        Instance.xi = xi
        Instance.eps_eva = eps_eva
        Instance.Link = Link
        
        