# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:25:46 2024

@author: Jiayang Li
"""
import torch
from GP import GP, sppconvert
from GP_SO import GP_SO


class Network():
    
    def __init__(self, name, Innode, Outnode, Link, Odtree):
        self.name = name
        self.Innode = Innode
        self.Outnode = Outnode
        self.Link = Link
        self.Odtree = Odtree
        
        self.x = None
        self.f = None
        
        self.pathflow = None
        self.flow = None
        self.path_number = None
        self.pathset = None
        self.path_id = dict()
        self.pathset2 = dict()
        
        self.path2od = list()
        
        self.od2num = dict()
        
        self.path_edge_lk = []
        self.path_edge_kindex = []
        self.path_demand_odindex = []
        self.path_demand_kindex = []
        self.numberofpath = 0
        self.numberoflink = 0
        self.numberofod = 0
        self.path_edge_index = []
        self.path_demand_index = []
        self.path_edge = None
        self.path_demand = None
        
        
    def LC(self, o, weight, Outnode, Link):
        impedance = {}
        predecessor = {}
        for i in Outnode.keys():
            if i != o:
                impedance[i] = float('inf')
            else:
                impedance[i] = 0
            predecessor[i] = -1
        Q = [o]
        while len(Q) != 0:
            i = Q[0]
            del Q[0]
            for j in Outnode[i]:
                l = (i,j)
                if impedance[j] > impedance[i] + weight[l]:
                    impedance[j] = impedance[i] + weight[l]
                    predecessor[j] = i
                    if j not in Q:
                        Q.append(j)
        return (impedance, predecessor)
    def LS(self, o, d, weight, Outnode, Link):
        impedance = {}
        predecessor = {}
        for ii in Outnode.keys():
            if ii != o:
                impedance[ii] = float('inf')
            else:
                impedance[ii] = 0
            predecessor[ii] = -1
        Q = [o]
        i = float('inf')
        while i != d:
            minw = Q[0]
            minim = impedance[minw]
            minindex = 0
            for index,w in enumerate(Q):
                if impedance[w] < minim:
                    minindex = index
                    minw = w
                    minim = impedance[w]
            i = minw
            del Q[minindex]
            for j in Outnode[i]:
                l = (i,j)
                if impedance[j] > impedance[i] + weight[l]:
                    impedance[j] = impedance[i] + weight[l]
                    predecessor[j] = i
                    if j not in Q:
                        Q.append(j)        
        return (impedance, predecessor)
        
    def LC_2factors(self, o, weight1, weight2, a, Outnode):
        impedance1 = {}
        impedance2 = {}
        predecessor = {}
        for i in Outnode.keys():
            if i != o:
                impedance1[i] = 1000000000
                impedance2[i] = 1000000000
            else:
                impedance1[i] = 0
                impedance2[i] = 0
            predecessor[i] = -1
        Q = [o]
        while len(Q) != 0:
            i = Q[0]
            del Q[0]
            for j in Outnode[i]:
                l = (i,j)
                i_impedance = impedance1[i] + a * impedance2[i]
                j_impedance = impedance1[j] + a * impedance2[j]
                l_impedance = weight1[l] + a * weight2[l]
                if j_impedance  > (i_impedance + l_impedance):
                    impedance1[j] = impedance1[i] + weight1[l]
                    impedance2[j] = impedance2[i] + weight2[l]
                    predecessor[j] = i
                    if j not in Q:
                        Q.append(j)
        return (impedance1, impedance2, predecessor)

    def LS_2factors(self, o, d,  weight1, weight2, a, Outnode):
        impedance1 = {}
        impedance2 = {}
        predecessor = {}
        for ii in Outnode.keys():
            if ii != o:
                impedance1[ii] = 1000000000
                impedance2[ii] = 1000000000
            else:
                impedance1[ii] = 0
                impedance2[ii] = 0
            predecessor[ii] = -1
        Q = [o]
        i = float('inf')
        while i != d:
            minw = Q[0]
            minim1 = impedance1[minw]
            minim2 = impedance2[minw]
            minindex = 0
            for index,w in enumerate(Q):
                if (impedance1[w] + a * impedance2[w]) < (minim1 + a * minim2):
                    minindex = index
                    minw = w
                    minim1 = impedance1[w]
                    minim2 = impedance2[w]
            i = minw
            del Q[minindex]
            for j in Outnode[i]:
                l = (i,j)
                i_impedance = impedance1[i] + a * impedance2[i]
                j_impedance = impedance1[j] + a * impedance2[j]
                l_impedance = weight1[l] + a * weight2[l]
                if j_impedance  > (i_impedance + l_impedance):
                    impedance1[j] = impedance1[i] + weight1[l]
                    impedance2[j] = impedance2[i] + weight2[l]
                    predecessor[j] = i
                    if j not in Q:
                        Q.append(j)
        return (impedance1, impedance2, predecessor)
    
    def solve_ue(self, demand, fftime, capacity, K0, e0, warm_start=False, toll=None, delete=True):
        
        Pathflow, Flow, _, _, Numberofpath, pathset = GP(self, demand, fftime, capacity, K0, e0, warm_start=warm_start, toll=toll, delete=delete)
        self.pathflow = Pathflow 
        self.flow = Flow
        self.path_number = Numberofpath
        self.pathset = pathset
        self.demand = demand
        
        ii = 0
        for od in demand.keys():
            self.od2num[od] = ii
            ii += 1
                
    def solve_so(self, demand, fftime, capacity, K0, e0, warm_start=True):
        Pathflow, Flow, _, _, Numberofpath, pathset = GP_SO(self, demand, fftime, capacity, K0, e0, warm_start=warm_start)
        self.pathflow = Pathflow 
        self.flow = Flow
        self.path_number = Numberofpath
        self.pathset = pathset
        
        
    def path_enumeration(self):
    
        kindex = 0
        odindex = 0
        self.numberoflink = len(self.Link)
        self.numberofod = len(self.pathflow)
        link_index = dict(zip(self.Link, range(len(self.Link))))
        self.numberofpath = 0
        
        self.path_edge_lk = list()
        self.path_edge_kindex = list()
        self.path_demand_odindex = list()
        self.path_demand_kindex = list()
        
        self.path2od = list()
        for o in self.Odtree.keys():
            for d in self.Odtree[o]:
                self.path_id[(o, d)] = dict()
                self.pathset2[(o, d)] = list()
                for k in self.pathflow[(o, d)].keys():
                    self.numberofpath += 1
                    # f[kindex] = self.pathflow[(o, d)][k]
                    for i in range(0, len(k) - 1):
                        lk = link_index[(k[i], k[i + 1])]
                        self.path_edge_lk.append(lk)
                        self.path_edge_kindex.append(kindex)
                        # path_edge[lk][kindex] = 1
                    # path_demand[odindex][kindex] = 1
                    self.path_demand_odindex.append(odindex)
                    self.path_demand_kindex.append(kindex)
                    self.pathset2[(o, d)].append(kindex)
                    self.path_id[(o, d)][k] = kindex
                    
                    self.path2od.append(self.od2num[(o, d)])
                    kindex += 1
                odindex += 1
                            
                
    def generate_flow(self):
        self.x = torch.tensor([self.flow[ii] for ii in self.Link], dtype=torch.double)
        f = torch.zeros(self.path_number, dtype=torch.double)
        for o in self.Odtree.keys():
            for d in self.Odtree[o]:
                for k in self.pathflow[(o, d)].keys():
                    kindex = self.path_id[(o, d)][k]
                    f[kindex] = self.pathflow[(o, d)][k]
        self.f = f
        
    def load_flow(self, f):
        for o in self.Odtree.keys():
            for d in self.Odtree[o]:
                for k in self.pathflow[(o, d)].keys():
                    kindex = self.path_id[(o, d)][k]
                    self.pathflow[(o, d)][k] = f[kindex]
                    
        flow = {l: 0 for l in self.Link}
        for o in self.Odtree.keys():
            for d in self.Odtree[o]:
                for path in self.pathflow[(o, d)].keys():
                    for i,j in zip(path[:-1],path[1:]):
                        l = (i,j)
                        flow[l] += self.pathflow[(o, d)][path]
        self.flow = flow
        
    def delete_path(self):
        for o in self.Odtree.keys():
            for d in self.Odtree[o]:
                deletepathset = []
                for path in self.pathflow[(o, d)].keys():
                    if abs(self.pathflow[(o, d)][path]) < pow(10, -9):
                        deletepathset.append(path)
                for dp in deletepathset:
                    
                    self.pathflow[(o, d)].pop(dp)
                    self.pathset[(o, d)].remove('-'.join(str(num) for num in dp))
    
    
    def generate_sparse_matrix(self):
                
        self.path_edge_index = torch.LongTensor([self.path_edge_lk, self.path_edge_kindex])
        self.path_demand_index = torch.LongTensor([self.path_demand_odindex, self.path_demand_kindex])
        path_edge_values = torch.DoubleTensor([1.0] * len(self.path_edge_lk))
        path_demand_values = torch.DoubleTensor([1.0] * len(self.path_demand_odindex))
        self.path_edge = torch.sparse.DoubleTensor(self.path_edge_index, path_edge_values, torch.Size([self.numberoflink, self.numberofpath]))
        self.path_demand = torch.sparse.DoubleTensor(self.path_demand_index, path_demand_values, torch.Size([self.numberofod, self.numberofpath]))
    
    def path_update(self, link_cost):

        link_index = dict(zip(self.Link, range(len(self.Link))))
        spps = {}
        odindex = 0
        for o in self.Odtree.keys():
            impedance, predecessor = self.LC(o, link_cost, self.Outnode, self.Link)
            for d in self.Odtree[o]:
                spp = sppconvert(predecessor, o, d)
                spps[(o, d)] = spp
                spp_string = '-'.join(str(num) for num in spp)
                if spp_string not in self.pathset[(o,d)]:
                    
                    self.pathset[(o, d)].add(spp_string)
                    self.pathflow[(o, d)][spp] = 1e-6
                    self.numberofpath += 1
                    k_index = self.numberofpath - 1
                    self.path2od.append(self.od2num[(o, d)])
                    for i in range(0, len(spp) - 1):
                        lk = link_index[(spp[i], spp[i + 1])]
                        self.path_edge_lk.append(lk)
                        self.path_edge_kindex.append(k_index)
                    self.path_demand_odindex.append(odindex)
                    self.path_demand_kindex.append(k_index)
                    
                    self.path_id[(o, d)][spp] = k_index
                    
                    
                odindex += 1
        self.generate_sparse_matrix()
        self.path_number = self.numberofpath


    def all_or_nothing(self, link_cost):

        link_index = dict(zip(self.Link, range(len(self.Link))))
        x = torch.zeros(len(self.Link), dtype=torch.double)
        for o in self.Odtree.keys():
            impedance, predecessor = self.LC(o, link_cost, self.Outnode, self.Link)
            for d in self.Odtree[o]:
                spp = sppconvert(predecessor, o, d)
                for i in range(0, len(spp) - 1):
                    lk = link_index[(spp[i], spp[i + 1])]
                    x[lk] += self.demand[(o, d)]
        return x
                        
       