"""
Created on Mon Apr  1 14:19:04 2019

@author: Qianni Wang and Jiayang Li
"""
from copy import deepcopy


def BPR(fftime, capacity, flow, alpha, beta):
    return (fftime*(1 + alpha * pow((flow/capacity),beta)))

def BPR_1_derivative(x_a, c_a, t0):
    return((4 * t0 * 0.15/(pow(c_a, 4)))*(pow(x_a, 3)))

def sppconvert(predecessor, o, d):
    if d == o:
        return tuple([o])
    p = d
    convert = [p]
    while True:
        p = predecessor[p]
        convert.insert(0, p)
        if p == o:
            break
    return tuple(convert)    

def update_linkflowtime(spp, flow, time, fftime, capacity, add_flow, toll=None):
    for i,j in zip(spp[:-1],spp[1:]):
        l = (i,j)
        flow[l] += add_flow
        time[l] = BPR(fftime[l], capacity[l], flow[l], 0.15, 4)
        if not toll == None:
            time[l] += toll[l]
    return (flow, time)

def calc_pathcost(path, linkimpedance):
    return sum(linkimpedance[(i, j)] for i, j in zip(path[:-1], path[1:]))

def calc_hkl(path, spp, time, isinspp, flow, capacity, fftime):
    hkl = 0
    path_link = {(i1,j1) for i1,j1 in zip(path[:-1], path[1:])}
    spp_link = {(i2,j2) for i2,j2 in zip(spp[:-1],spp[1:])}
    add_set = path_link.union(spp_link)
    add_set = add_set.difference(set(spp_link).intersection(set(path_link)))
    for l in list(add_set):
        hkl += BPR_1_derivative(flow[l], capacity[l], fftime[l])
    return hkl


def GP(net, demand, fftime, capacity, K0, e0, warm_start=False, toll=None, delete=True):
    

    
    #initialize
    maxInIter = 5
    RGlist = []
    Objlist = []
    k = 0
    RG = float('inf')
    if warm_start is False:
        time = deepcopy(fftime)
        if not toll == None: 
            for key in net.Link:
                time[key] += toll[key]
        flow = {key: 0 for key in net.Link}
        pathflow = {key: {} for key in demand.keys()}
        pathset = {key: set() for key in demand.keys()}
        for o in net.Odtree.keys():
            impedance, predecessor = net.LC(o, time, net.Outnode, net.Link)
            for d in net.Odtree[o]:
                spp = sppconvert(predecessor, o, d)
                spp_string = '-'.join(str(num) for num in spp)
                pathset[(o,d)].add(spp_string)
                pathflow[(o,d)][spp] = demand[(o,d)]
                if not toll == None: 
                    flow, time = update_linkflowtime(spp, flow, time, fftime, capacity, demand[(o,d)], toll=toll)
                else:
                    flow, time = update_linkflowtime(spp, flow, time, fftime, capacity, demand[(o,d)])
    else:
        pathflow = deepcopy(net.pathflow)
        pathset = deepcopy(net.pathset)
        flow = deepcopy(net.flow)
        time = {key: 0 for key in net.Link}
        for key in net.Link:
            time[key] = BPR(fftime[key], capacity[key], flow[key], 0.15, 4)
        if not toll == None: 
            for key in net.Link:
                time[key] += toll[key]
        
        # fenzi = 0
        # fenmu = 0
        # for o in net.Odtree.keys():
        #     impedance, predecessor = net.LC(o, time, net.Outnode, net.Link)
        #     for d in net.Odtree[o]:
        #         fenzi += impedance[d] * demand[(o,d)]
        # for link in net.Link:
        #     fenmu += flow[link] * time[link]
        # RG = 1 - (fenzi / fenmu)
        # if RG <= e0:
        #     return (pathflow, flow, RGlist, Objlist, net.path_number, pathset)
        

    #mainloop
    
    # if not toll == None: 
    #     print('yes')
    while True:
        # print(RG)
        if RG < 1e-6:
            maxInIter = 5
        spps = {}
        fenzi = 0
        fenmu = 0
        numberofpath = 0
        
        for o in net.Odtree.keys():
            impedance, predecessor = net.LC(o, time, net.Outnode, net.Link)
            for d in net.Odtree[o]:
                spp = sppconvert(predecessor, o, d)
                spps[(o, d)] = spp
                spp_string = '-'.join(str(num) for num in spp)
                if spp_string not in pathset[(o,d)]:
                    pathset[(o, d)].add(spp_string)
                    pathflow[(o, d)][spp] = 0
                fenzi += impedance[d] * demand[(o,d)]
                numberofpath += len(pathset[(o, d)])
        for link in net.Link:
            fenmu += flow[link] * time[link]
        RG = 1 - (fenzi / fenmu)
        
        obj = calc_obj(flow, fftime, capacity, toll=toll)
        Objlist.append(obj)
        RGlist.append(RG)
        # print('Step %s: RG = '%k, RG)
        # if (k % 1 == 0):
        #     print('Step %s: RG = '%k, RG)
        if (k >= K0) or (RG <= e0):
            if k >= K0:
                print('does not converge')
            break
        
        # start_time = tm.time()
        for il in range(maxInIter):
            # execution_time = 0
            # print(il)
            total_shift_od_num = 0
            for o in net.Odtree.keys():
                for d in net.Odtree[o]:
                    # start_time = tm.time()
                    spp = spps[(o, d)]
                    maxCost = 0.0
                    minCost = 100000.0
                    for path in pathflow[(o, d)].keys():
                        cst = calc_pathcost(path, time)
                        if (cst > maxCost):
                            maxCost = cst
                        if (cst < minCost):
                            minCost = cst
                    shiftFlow = maxCost - minCost
                    # end_time = tm.time()
                    # execution_time += end_time - start_time
                    if (shiftFlow > RG/2.0):
                        total_shift_od_num += 1
                        isinspp = {(ii, jj): 1 for ii, jj in zip(spp[:-1], spp[1:])}
                        
                        for path in pathflow[(o, d)].keys():
                            if path != spp:
                                gkl = calc_pathcost(path, time) - calc_pathcost(spp, time)
                                
                                hkl = calc_hkl(path, spp, time, isinspp, flow, capacity, fftime)
                                delta = max(pathflow[(o, d)][spp] + (
                                            pathflow[(o, d)][path] - max(0, pathflow[(o, d)][path] - gkl / hkl)), 0) - \
                                        pathflow[(o, d)][spp]
                                pathflow[(o, d)][path] -= delta
                                pathflow[(o, d)][spp] += delta
                                shiftFlow += abs(delta)
                                if not toll == None: 
                                    flow, time = update_linkflowtime(path, flow, time, fftime, capacity, (-1) * delta, toll=toll)
                                    flow, time = update_linkflowtime(spp, flow, time, fftime, capacity, delta, toll=toll)
                                else:
                                    flow, time = update_linkflowtime(path, flow, time, fftime, capacity, (-1) * delta)
                                    flow, time = update_linkflowtime(spp, flow, time, fftime, capacity, delta)
            if (total_shift_od_num < 3):
                break
        # end_time = tm.time()
        # execution_time = end_time - start_time
        # print("running time: ", execution_time, "s")
        
        if delete:
            for o in net.Odtree.keys():
                for d in net.Odtree[o]:
                    deletepathset = []
                    for path in pathflow[(o, d)].keys():
                        if abs(pathflow[(o, d)][path]) < pow(10, -9):
                            deletepathset.append(path)
    
                    for dp in deletepathset:
                        
                        pathflow[(o, d)].pop(dp)
                        pathset[(o, d)].remove('-'.join(str(num) for num in dp))
        k += 1    
        
    if delete:
        for o in net.Odtree.keys():
                for d in net.Odtree[o]:
                    deletepathset = []
                    for path in pathflow[(o, d)].keys():
                        if abs(pathflow[(o, d)][path]) < pow(10, -9):
                            deletepathset.append(path)
    
                    for dp in deletepathset:
                        
                        pathflow[(o, d)].pop(dp)
                        pathset[(o, d)].remove('-'.join(str(num) for num in dp))
         
    numberofpath = 0
    for o in net.Odtree.keys():
        for d in net.Odtree[o]:
            numberofpath += len(pathset[(o, d)])

        
    return (pathflow, flow, RGlist, Objlist, numberofpath, pathset)
  

def calc_obj(flow, fftime, capacity,toll=None):
    obj = 0
    for l in flow.keys():
        obj += fftime[l] * flow[l] + 0.15 * fftime[l] * pow(flow[l], 5)/(5 * pow(capacity[l], 4))
        if not toll == None:
            obj += flow[l] * toll[l]
    return obj
