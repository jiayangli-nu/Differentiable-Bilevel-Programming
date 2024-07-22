# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:25:46 2024

@author: 84091
"""
import torch
import pickle
import time as tm
from sctp_load import Instance
from numba import jit


@jit(nopython=True)
def update_b(c, b, path2od):
    for k, w in enumerate(path2od):
        bmin = b[w]
        if c[k] < bmin:
            b[w] = c[k]
    return b


for NETWORK in ['cs']:
    
    if NETWORK == 'sf':
        
        Delta_list = 0.001
        
    if NETWORK == 'bar' or NETWORK =='cs':
        Delta_list = 0.01
        
        
    # methods = ['emcdt']
    # for delta in Delta_list:
    #     methods.append('mct' + str(delta))
    #     methods.append('ct' + str(delta))
    
    methods = ['mct', 'emcdt', 'ct']
        
    initializations = list()

    for i, method in enumerate(methods):
        
        path_to_pickle = 'result_sctp/' + method + '_' + NETWORK + '.pkl'
        with open(path_to_pickle, 'rb') as f:
            Result = pickle.load(f)
            
        print(Result['solution'])
        print(method, Result['obj'])
            
        initializations.append([Result['solution'], method])
        
    all_zero = torch.zeros_like(Result['solution'])
    constant_tolls = [0]
        
    for c in constant_tolls:
        initializations.append([all_zero + c, str(c)])
    
    path_to_pickle = 'result_sctp/initialization_' + NETWORK + '.pkl'
    with open(path_to_pickle, 'wb') as f:
        pickle.dump(initializations, f)
    
    
