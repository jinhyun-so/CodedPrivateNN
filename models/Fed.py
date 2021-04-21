#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1,len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvg_with_BACC_Dec(w, alpha_array, dec_z_array, is_debug=False):
    w_avg = copy.deepcopy(w[0])
    dec_len = len(w)

    K = len(alpha_array)
    w_dec_array = [] # for debugging
    for i in range(K):
        w_tmp = copy.deepcopy(w[0])
        w_dec_array.append(w_tmp)

    for k in w_avg.keys():
        tmp1 = w_avg[k].cpu().detach().numpy()
        cur_shape = tmp1.shape

        d = np.prod(cur_shape).astype('int')

        #print(k, d)
        #print(cur_shape)

        f_tilde = np.empty((dec_len,d,1))

        for i in range(dec_len):
            tmp = w[i][k].cpu().detach().numpy()
            f_tilde[i,:,:] = np.reshape(tmp,(d,1))

        f_dec = BACC_Dec(f_tilde, alpha_array, dec_z_array)

        for i in range(K):
            f_tmp = f_dec[i,:]
            

            if d == 1:
                tmp = f_tmp[0][0]
                #f_tmp_tensor = torch.Tensor(tmp).cuda()
                #print(f_tmp_tensor)
                #w_dec_array[i][k] += -w_dec_array[i][k] + f_tmp_tensor[0][0]
                w_dec_array[i][k] += -w_dec_array[i][k] + tmp.astype('long')
            else:
                f_tmp = np.reshape(f_tmp,cur_shape)
                f_tmp_tensor = torch.Tensor(f_tmp).cuda()
                w_dec_array[i][k] += -w_dec_array[i][k] + f_tmp_tensor

        if d == 1:
            f_sum = np.sum(f_dec, axis=0)
            f_mean = f_sum[0][0]/dec_len

            w_avg[k] += -w_avg[k] + f_mean.astype('long')
        else:
            f_sum = np.sum(f_dec, axis=0)
            f_mean = np.reshape(f_sum/dec_len,cur_shape)
    
            f_mean_tensor = torch.Tensor(f_mean).cuda()
        
            #print(w_avg[k][0])
            #print(f_mean[0])

            w_avg[k] += -w_avg[k] + f_mean_tensor
    if is_debug == True:
        return w_avg, w_dec_array
    else:
        return w_avg

def FedAvg_with_LCC_Dec(w, alpha_array, dec_z_array):
    w_avg = copy.deepcopy(w[0])
    dec_len = len(w)
    for k in w_avg.keys():
        tmp1 = w_avg[k].cpu().detach().numpy()
        cur_shape = tmp1.shape

        f_tilde = np.empty((dec_len,np.prod(cur_shape),1))

        for i in range(dec_len):
            tmp = w[i][k].cpu().detach().numpy()
            f_tilde[i,:,:] = np.reshape(tmp,(np.prod(cur_shape),1))

        f_dec = LCC_Dec(f_tilde, alpha_array, dec_z_array)

        f_sum = np.sum(f_dec, axis=0)
        f_mean = np.reshape(f_sum/dec_len,cur_shape)
    
        f_mean_tensor = torch.Tensor(f_mean).cuda()
        
        #print(w_avg[k][0])
        #print(f_mean[0])

        w_avg[k] += -w_avg[k] + f_mean_tensor
    return w_avg

def FedQAvg(w,q_val):
    w_avg = copy.deepcopy(w[0])
    #print(w_avg)
    for k in w_avg.keys():
        #print(k)
        for i in range(1,len(w)):
            #print(w[i][k])
            #print(torch.min(w[i][k]),torch.max(w[i][k]))
            #print((w[i][k]))
            temp = torch.mul(w[i][k], q_val).round()
            temp = torch.div(temp, q_val)
            
            #temp_np = temp.cpu().detach().numpy() 
            #print(k, np.shape(temp_np))
            #print((temp))
            w_avg[k] += temp
        w_avg[k] = torch.div(w_avg[k], len(w))
    
    loc = 0
    w_out = np.zeros((1,62346),dtype=float)
    return w_avg, w_out

def FedQAvg2(w,q_val,args):
    w_avg = copy.deepcopy(w[0])
    #print(w_avg)
    for k in w_avg.keys():
        #print(k)
        for i in range(1,len(w)):
            #print(w[i][k])
            #print(torch.min(w[i][k]),torch.max(w[i][k]))
            #print((w[i][k]))
            temp = w[i][k]
            temp_np = temp.cpu().detach().numpy()
            temp_np = np.round(temp_np*q_val,0)
            temp_np = temp_np / q_val
            
            #temp_np = temp.cpu().detach().numpy() 
            #print(k, np.shape(temp_np))
            #print((temp))
            w_avg[k] +=  torch.from_numpy(temp_np).float().to(args.device)
        w_avg[k] = torch.div(w_avg[k], len(w))
    
    loc = 0
    w_out = np.zeros((1,62346),dtype=float)
    '''
    for k in w_avg.keys():
        
        temp = w[i][k]
        temp_np = temp.cpu().detach().numpy()
            
        temp_len = np.prod(np.shape(temp_np))
        temp_np = np.reshape(temp_np,(temp_len))
        w_out[0][loc:loc+temp_len] = temp_np.astype(float)
            
        #print(i,k,w_out[i][loc],temp_np[0])
        loc = loc + temp_len
    '''
    return w_avg, w_out

def FedBrea(w_locals,m,A,q_bit,p,args):
    
    w_locals_np = {} # in finite-field
    for i in range(len(w_locals)):
        temp_np = {} #np.array([])
        w_local_np_flatten = np.array([],dtype=float)
    #     print(i)
        for k in w_locals[i].keys():
            tmp = w_locals[i][k].cpu().detach().numpy()
            tmp_flatten = tmp.flatten()
            w_local_np_flatten = np.concatenate((w_local_np_flatten,tmp_flatten), axis=0)
    #         tmp_q = my_q(tmp,8,p)
            temp_np.update({k: tmp})
    #         print(k,np.shape(tmp))
    #         temp_np = np.concatenate((temp_np,  ),axis=0)
        #print(np.shape(w_local_np_flatten))
        w_locals_np.update({i: my_q(w_local_np_flatten, q_bit,p)})

    idx_select = multi_Krum(w_locals_np, m, A, q_bit, p)

    print('selected clients index;',idx_select)
    w_avg = copy.deepcopy(w_locals[idx_select[0]])
    #print('fc1.bias:',w_avg['fc1.bias'])

    for k in w_avg.keys():
        for i in range(1,m):
            q_val = (2**q_bit)
            temp = torch.mul(w_locals[idx_select[i]][k], q_val).round()
            temp = torch.div(temp, q_val)
            w_avg[k] += temp

            #temp = w_locals[i][k]
            #temp_np = temp.cpu().detach().numpy()
            #temp_np = np.round(temp_np*(2**q_bit),0)
            #temp_np = temp_np / (2**q_bit)
            
            #temp_np = temp.cpu().detach().numpy() 
            #print(k, np.shape(temp_np))
            #print((temp))
            #w_avg[k] +=  torch.from_numpy(temp_np).float().to(args.device)
        w_avg[k] = torch.div(w_avg[k], m)

    return w_avg, idx_select


def multi_Krum(w_locals_np, m, A, q_bit, p):
    
    # m = the number of selected models
    # A = the number of adversaries
    N = len(w_locals_np)
    dist = np.zeros((N,N))
#     dist_float = np.zeros((N,N)) # for debugging
    
    for i in range(N):
        for j in range(N):
            if i==j:
                dist[i,j] = 0
            else:
                dist_tmp = np.mod(w_locals_np[i] - w_locals_np[j],p)
                dist_tmp = np.reshape(dist_tmp,(len(dist_tmp),1))
                
                dot_interval = 512
                num_iter = np.ceil(len(dist_tmp)/dot_interval).astype(int)
                #print(len(dist_tmp),num_iter)
                for k in range(num_iter):
                    pos_stt = k*dot_interval
                    pos_end = np.min((len(dist_tmp),(k+1)*dot_interval))
                    tmp = dist_tmp[pos_stt:pos_end,0]
                    dist[i,j] = np.mod(dist[i,j] + np.matmul(tmp.transpose(),tmp),p)
                #dist_tmp_float = my_q_inv(dist_tmp, q_bit,p)
                #dist[i,j] = np.mod(np.matmul(dist_tmp.transpose(),dist_tmp),p)
                #dist_float[i,j] = np.matmul(dist_tmp_float.transpose(),dist_tmp_float)
    #print(dist)
    #print(my_q_inv(dist, q_bit*2,p))
    #print(dist_float)
    
    # select lowest client whose socre is the lowest
    idx_select = []
    for k in range(m):
        score = np.zeros((N))
        
        for i in range(N):
            tmp_arr = dist[i,:]
            tmp_arr_sorted = np.sort(tmp_arr)
            num_sum = N-A-2-k
            score[i] = np.sum(tmp_arr_sorted[1:1+num_sum])
    #         print(tmp_arr)
    #         print(tmp_arr_sorted)
    #         print(score[i])
    #         print()

    #     print(score)
        idx_min_score = np.argmin(score)
        dist[idx_min_score,:] = (p-1)/2
        dist[:,idx_min_score] = (p-1)/2
        
        #print(dist)
        
        idx_select.append(idx_min_score)
        #print(idx_select)
    return idx_select


def Quantization(w):
    w_avg = copy.deepcopy(w[0])
    w_out = np.zeros((len(w),118346),dtype=float)
    for i in range(len(w)):
        loc = 0
        for k in w_avg.keys():
        
            temp = w[i][k]
            temp_np = temp.cpu().detach().numpy()
            
            temp_len = np.prod(np.shape(temp_np))
            temp_np = np.reshape(temp_np,(temp_len))
            w_out[i][loc:loc+temp_len] = temp_np.astype(float)
            
            #print(i,k,w_out[i][loc],temp_np[0])
            loc = loc + temp_len
            #print(loc)

    return np.round(w_out * 1024)

def Quantization_Finite(w,q_bit,p):
    w_avg = copy.deepcopy(w[0])
    w_out = np.zeros((len(w),62346),dtype=float)
    for i in range(len(w)):
        loc = 0
        for k in w_avg.keys():
        
            temp = w[i][k]
            temp_np = temp.cpu().detach().numpy()
            
            temp_len = np.prod(np.shape(temp_np))
            temp_np = np.reshape(temp_np,(temp_len))
            w_out[i][loc:loc+temp_len] = temp_np.astype(float)
            
            #print(i,k,w_out[i][loc],temp_np[0])
            loc = loc + temp_len
            print(loc)

    return my_q(w_out,q_bit,p) #w_out * 1024

def my_q(X,q_bit,p):
    X_int = np.round(X*(2**q_bit))
    is_negative = (abs(np.sign(X_int)) - np.sign(X_int))/2
    out = X_int + p * is_negative
    return out.astype('int64')

def my_q_inv(X_q,q_bit,p):
    flag = X_q - (p-1)/2
    is_negative = (abs(np.sign(flag)) + np.sign(flag))/2
    X_q = X_q - p * is_negative
    return X_q.astype(float)/(2**q_bit)

def my_score(w,m):
    # m = the number of output
    dist_array = np.zeros((len(w),len(w)))
    #print(np.shape(dist_array))

    for i in range(len(w)):
        vec_i = w[i][:]
        for j in range(len(w)):
            vec_j = w[j][:]
            temp_diff = vec_i - vec_j
            dist_array[i][j] = np.sum(np.multiply(temp_diff,temp_diff)) #temp_diff.transpose() * temp_diff
            #print(temp_dist)
    print(dist_array)

def my_score_Finite(w,m,q_bit,p):
    # m = the number of output
    dist_array = np.zeros((len(w),len(w)))
    #print(np.shape(dist_array))

    for i in range(len(w)):
        vec_i = w[i][:].astype('int64')
        for j in range(len(w)):
            vec_j = w[j][:].astype('int64')
            temp_diff = np.mod(vec_i - vec_j,p)
            #print(temp_diff)
            dist_array[i][j] = np.mod(np.sum(np.multiply(temp_diff,temp_diff)),p) #temp_diff.transpose() * temp_diff
            #print(temp_dist)
    #print(dist_array)
    out = my_q_inv(dist_array,0,p)
    print(out)

def BACC_Dec(_f_tilde, _alpha_array, _z_array):
    '''
    inputs:
    
    _f_tilde : numpy [_N * (shape of f) ]
    _alpha_array : numpy [_K] array
    _z_array     : numpy [_N] array
    
    Parameters:
    _N : number of (non-straggling) worker nodes
    _K : number of submatrices
    
    Outputs:
    _f : numpy [_K * (shape of f)]    
    '''
    
    _K = len(_alpha_array)
    _N = len(_z_array)
    
    _N_, _m, _d = np.shape(_f_tilde)
    
    assert _N == _N_, "first dim of _f_tilde should be same as the length of _z_array!!\n"
    
    _f = np.zeros((_K,_m,_d))
    
    _W = np.ones((_K,_N))
    _W[:,1::2] = -1
    
    _U = np.reshape(_alpha_array,(_K,1)) - np.reshape(_z_array,(1,_N))
    _U = 1/_U
    _U = _U * _W
    
    for i in range(_K):
        denom =  np.sum(_U[i,:])
        for j in range(_N):
            _f[i,:,:] = _f[i,:,:] + _U[i,j]/denom * _f_tilde[j,:,:]
            
    return _f

def PI(vals):  # upper-case PI -- product of inputs
    accum = 1
    for v in vals:
        accum = accum*v
    return accum

def gen_Lagrange_coeffs(alpha_s,beta_s,is_K1=0):
    if is_K1==1:
        num_alpha = 1
    else:
        num_alpha = len(alpha_s)
    U = np.zeros((num_alpha, len(beta_s)))
#         U = [[0 for col in range(len(beta_s))] for row in range(len(alpha_s))]
    #print(alpha_s)
    #print(beta_s)
    for i in range(num_alpha):
        for j in range(len(beta_s)):
            cur_beta = beta_s[j];

            den = PI([cur_beta - o   for o in beta_s if cur_beta != o])
            num = PI([alpha_s[i] - o for o in beta_s if cur_beta != o])
            U[i][j] = num/den
            # for debugging
            # print(i,j,cur_beta,alpha_s[i])
            # print(test)
            # print(den,num) 
    return U

def LCC_Dec(_f_tilde, _alpha_array, _z_array):
    '''
    inputs:
    
    _f_tilde : numpy [_N * (shape of f) ]
    _alpha_array : numpy [_K] array
    _z_array     : numpy [_N] array
    
    Parameters:
    _N : number of (non-straggling) worker nodes
    _K : number of submatrices
    
    Outputs:
    _f : numpy [_K * (shape of f)]    
    '''
    
    _K = len(_alpha_array)
    _N = len(_z_array)
    
    _N_, _m, _d = np.shape(_f_tilde)
    
    assert _N == _N_, "first dim of _f_tilde should be same as the length of _z_array!!\n"
    
    _f = np.zeros((_K,_m,_d))
    
    _U = gen_Lagrange_coeffs(_alpha_array,_z_array,is_K1=0)
    
    for i in range(_K):
        for j in range(_N):
            _f[i,:,:] = _f[i,:,:] + _U[i,j] * _f_tilde[j,:,:]
            
    return _f