import numpy as np
import random
import math
import copy
import torch
from torch import nn

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

def LCC_Enc(_input_array, _alpha_array, _z_array):
    '''
    Inputs:
    
    _input_array : numpy [m * d] array
    _alpha_array : numpy [_K] array
    _z_array     : numpy [_N] array
    
    Parmeters:
    _m : sample size
    _d : feature size
    _N : number of worker nodes
    _K : number of submatrices 
    
    Output:
    _X_tilde : numpy [_N * (m/_K) * d] array
    
    '''
    _K = len(_alpha_array)
    _N = len(_z_array)
        
    _m, _d = np.shape(_input_array)
    
    _m_i = np.floor(int(_m) / int(_K)).astype(int)
    #print ('@BACC_Enc: N,K, m_i=',_N,_K,_m_i,'\n')
    
    assert _m_i >= 1, "data size(=m) should be larger than or equal to K \n"
    
    _X_tilde = np.zeros((_N,_m_i,_d))
    
    _U = gen_Lagrange_coeffs(_z_array,_alpha_array,is_K1=0)
    
    for i in range(_N):
        for j in range(_K):
            _X_tilde[i,:,:] = _X_tilde[i,:,:] + _U[i,j] * _input_array[_m_i*j:_m_i*(j+1),:]
    
    return _X_tilde

def BACC_Enc(_input_array, _alpha_array, _z_array):
    '''
    Inputs:
    
    _input_array : numpy [m * d] array
    _alpha_array : numpy [_K] array
    _z_array     : numpy [_N] array
    
    Parmeters:
    _m : sample size
    _d : feature size
    _N : number of worker nodes
    _K : number of submatrices 
    
    Output:
    _X_tilde : numpy [_N * (m/_K) * d] array
    
    '''
    _K = len(_alpha_array)
    _N = len(_z_array)
        
    _m, _d = np.shape(_input_array)
    
    _m_i = np.floor(int(_m) / int(_K)).astype(int)
    #print ('@BACC_Enc: N,K, m_i=',_N,_K,_m_i,'\n')
    
    assert _m_i >= 1, "data size(=m) should be larger than or equal to K \n"
    
    _X_tilde = np.zeros((_N,_m_i,_d))
    
    _W = np.ones((_N,_K))
    _W[:,1::2] = -1
    
    _U = np.reshape(_z_array,(_N,1)) - np.reshape(_alpha_array,(1,_K))
    # print("before:",_U[0:3,0:3],"\n")
    _U = 1/_U
    # print("after:",_U[0:3,0:3],"\n")
    _U = _U * _W
    # print("after multi by _W: ",_U[0:3,0:3],"\n")
    
    for i in range(_N):
        denom = np.sum(_U[i,:])
        for j in range(_K):
            _X_tilde[i,:,:] = _X_tilde[i,:,:] + _U[i,j]/denom * _input_array[_m_i*j:_m_i*(j+1),:]
    
    return _X_tilde

def BACC_Enc_withNoise(_input_array, _N, _K, _T, _sigma, _Noise_Alloc = None, _Noise = None, is_predefined_noise=False):
    '''
    Inputs:
    
    _input_array : numpy [m * d] array
    _N : number of worker nodes
    _K : number of submatrices 
    _T : privacy parameter = number of random matrices
    _sigma : variance

    _Noise_Alloc : location of random matrices
        if None: randomly assign (_T) locations out of ( _K + _T )
        else: assign (_T) locations accordingly. 
              e.g. when _K=3, T_2, it could be [0, 1, 0, 0, 1].

    Parmeters:
    _alpha_array : numpy [_K] array
    _z_array     : numpy [_N] array
    _m : sample size
    _d : feature size
    
    Output:
    _X_tilde : numpy [_N * (m/_K) * d] array
    '''
    j_array = np.array(range(_K+_T))
    _alpha_array = np.cos((2*j_array+1)*math.pi/(2*(_K+_T))) #np.cos((2*j_array+1)*math.pi/(2*K))

    i_array = np.array(range(_N))
    _z_array = np.cos(i_array*2*math.pi/_N/2) # np.cos(i_array*2*math.pi/N/2)

    if _Noise_Alloc == None:
        noise_idxs = np.random.choice(range(_K+_T), _T, replace=False)
        noise_idxs = np.sort(noise_idxs)
    else:
        noise_idxs = _Noise_Alloc    

    _m, _d = np.shape(_input_array)
    
    _m_i = np.floor(int(_m) / int(_K)).astype(int)
    print ('@BACC_Enc: N,K,T, m_i=',_N,_K,_T,_m_i,'\n')
    
    assert _m_i >= 1, "data size(=m) should be larger than or equal to K \n"

    _X_extended = np.empty((_m_i*(_K+_T),_d))
    
    # print(_Noise)

    if is_predefined_noise == False:
        _Noise = np.random.normal(0,_sigma,size=(_m_i * _T,_d))
    else:
        assert np.shape(_Noise)[1] == _d, "dimension of noise should be the same as the dimension of data \n"
        assert np.shape(_Noise)[0] >= _m_i * _T, "number of noise should be equal to (or larger than) T"
    
    i_loc = 0
    j_loc = 0 # noise index
    for i in range(_K+_T):
        if i in noise_idxs:
            _X_extended[i*_m_i:(i+1)*_m_i,:] = _Noise[j_loc*_m_i:(j_loc+1)*_m_i,:]
            j_loc += 1
        else:
            _X_extended[i*_m_i:(i+1)*_m_i,:] = _input_array[i_loc*_m_i:(i_loc+1)*_m_i,:]
            i_loc += 1
    
    # return _X_extended for debugging
    # noise_idxs should be known to the server for decoding
    return BACC_Enc(_X_extended, _alpha_array, _z_array), _X_extended, noise_idxs

def BACC_Enc_Model_withNoise(_net, _N, _K, _T, _sigma, _Noise_Alloc = None):
    '''
    Inputs:
    
    _net : pytorch model
    _N : number of worker nodes
    _K : number of submatrices 
    _T : privacy parameter = number of random matrices
    _sigma : variance

    _Noise_Alloc : location of random matrices
        if None: randomly assign (_T) locations out of ( _K + _T )
        else: assign (_T) locations accordingly. 
              e.g. when _K=3, T_2, it could be [0, 1, 0, 0, 1].

    Parmeters:
    _alpha_array : numpy [_K] array
    _z_array     : numpy [_N] array
    
    Output:
    _net_array : array of net, whose length is _N
    '''

    j_array = np.array(range(_K+_T))
    _alpha_array = np.cos((2*j_array+1)*math.pi/(2*(_K+_T))) #np.cos((2*j_array+1)*math.pi/(2*K))

    i_array = np.array(range(_N))
    _z_array = np.cos(i_array*2*math.pi/_N/2) # np.cos(i_array*2*math.pi/N/2)

    if _Noise_Alloc == None:
        noise_idxs = np.random.choice(range(_K+_T), _T, replace=False)
        noise_idxs = np.sort(noise_idxs)
    else:
        noise_idxs = _Noise_Alloc


    net_array = []
    w_array = []

    for n in range(_N):
        net_tmp = copy.deepcopy(_net)
        net_array.append(net_tmp)
        w_array.append(net_tmp.state_dict())

    net_tmp = copy.deepcopy(_net)
    w_tmp = net_tmp.state_dict()

    for k in w_tmp.keys():
        tmp1 = w_tmp[k].cpu().detach().numpy()
        cur_shape = tmp1.shape
        _d = np.prod(cur_shape)

        _W_extended = np.empty((1*(_K+_T),_d))

        for i in range(_K+_T):
            if i in noise_idxs:
                _W_extended[i:(i+1),:] = np.random.normal(0,_sigma,size=(1,_d))
            else:
                _W_extended[i:(i+1),:] = np.reshape(tmp1,(1,_d))
        coded_W = BACC_Enc(_W_extended, _alpha_array, _z_array)

        #print(k)
        #print(np.shape(coded_W))
        #print()

        for n in range(_N):
            tmp = np.reshape(coded_W[n,0,:],cur_shape)
            w_array[n][k] += - w_array[n][k] + torch.Tensor(tmp).cuda()

            #print(n,k,w_array[n][k])

    for n in range(_N):
        net_array[n].load_state_dict(w_array[n])

    return net_array


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
    
    print('chekc the length!',_N, _N_)
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


def BACC_Enc_withNoise_v2(_input_array, _N, _K, _T, _sigma, _Noise_Alloc = None, _Noise = None, is_predefined_noise=False):
    '''
    Inputs:
    
    _input_array : numpy [m * d] array
    _N : number of worker nodes
    _K : number of submatrices 
    _T : privacy parameter = number of random matrices
    _sigma : variance

    _Noise_Alloc : location of random matrices
        if None: randomly assign (_T) locations out of ( _K + _T )
        else: assign (_T) locations accordingly. 
              e.g. when _K=3, T_2, it could be [0, 1, 0, 0, 1].

    Parmeters:
    _alpha_array : numpy [_K] array
    _z_array     : numpy [_N] array
    _m : sample size
    _d : feature size
    
    Output:
    _X_tilde : numpy [_N * (m/_K) * d] array
    '''

    j_array_org = np.array(range(_N))
    stt_pos = int(np.floor((_N-_K-_T)/2))

    j_array = j_array_org[stt_pos:stt_pos+_K+_T]

    _alpha_array = np.cos((2*j_array+1)*math.pi/(2*_N)) #np.cos((2*j_array+1)*math.pi/(2*K))

    i_array = np.array(range(_N))
    _z_array = np.cos(i_array*2*math.pi/_N/2) # np.cos(i_array*2*math.pi/N/2)

    if _Noise_Alloc == None:
        noise_idxs = np.random.choice(range(_K+_T), _T, replace=False)
        noise_idxs = np.sort(noise_idxs)
    else:
        noise_idxs = _Noise_Alloc    

    _m, _d = np.shape(_input_array)
    
    _m_i = np.floor(int(_m) / int(_K)).astype(int)
    print ('@BACC_Enc: N,K,T, m_i=',_N,_K,_T,_m_i,'\n')
    
    assert _m_i >= 1, "data size(=m) should be larger than or equal to K \n"

    _X_extended = np.empty((_m_i*(_K+_T),_d))
    
    # print(_Noise)

    if is_predefined_noise == False:
        _Noise = np.random.normal(0,_sigma,size=(_m_i * _T,_d))
    else:
        assert np.shape(_Noise)[1] == _d, "dimension of noise should be the same as the dimension of data \n"
        assert np.shape(_Noise)[0] >= _m_i * _T, "number of noise should be equal to (or larger than) T"
    
    i_loc = 0
    j_loc = 0 # noise index
    for i in range(_K+_T):
        if i in noise_idxs:
            _X_extended[i*_m_i:(i+1)*_m_i,:] = _Noise[j_loc*_m_i:(j_loc+1)*_m_i,:]
            j_loc += 1
        else:
            _X_extended[i*_m_i:(i+1)*_m_i,:] = _input_array[i_loc*_m_i:(i_loc+1)*_m_i,:]
            i_loc += 1
    
    # return _X_extended for debugging
    # noise_idxs should be known to the server for decoding
    return BACC_Enc(_X_extended, _alpha_array, _z_array), _X_extended, noise_idxs

def BACC_Enc_Data_v3(_input_array, _N, _K, _T, _sigma, _alpha_array, _z_array, _Noise_Alloc = None, _Noise = None, is_predefined_noise=False, _is_LCC = False):
    '''
    Inputs:
    
    _input_array : numpy [m * d] array
    _N : number of worker nodes
    _K : number of submatrices 
    _T : privacy parameter = number of random matrices
    _sigma : variance

    _alpha_array : numpy [_K] array
    _z_array     : numpy [_N] array

    _Noise_Alloc : location of random matrices
        if None: randomly assign (_T) locations out of ( _K + _T )
        else: assign (_T) locations accordingly. 
              e.g. when _K=3, T_2, it could be [0, 1, 0, 0, 1].

    Parmeters:    
    _m : sample size
    _d : feature size
    
    Output:
    _X_tilde : numpy [_N * (m/_K) * d] array
    '''


    if _Noise_Alloc == None:
        noise_idxs = np.random.choice(range(_K+_T), _T, replace=False)
        noise_idxs = np.sort(noise_idxs)
    else:
        noise_idxs = _Noise_Alloc
    
    if _T == 0:
        noise_idxs = []

    _m, _d = np.shape(_input_array)
    
    _m_i = np.floor(int(_m) / int(_K)).astype(int)
    print ('@BACC_Enc: N,K,T, m_i=',_N,_K,_T,_m_i,'\n')
    
    assert _m_i >= 1, "data size(=m) should be larger than or equal to K \n"
    assert len(_alpha_array) == _K + _T, "length of _alpha_array should be equal to K+T"
    assert len(_z_array) == _N, "length of _z_array should be equal to N"

    _X_extended = np.empty((_m_i*(_K+_T),_d))
    
    # print(_Noise)

    if is_predefined_noise == False:
        _Noise = np.random.normal(0,_sigma,size=(_m_i * _T,_d))
    else:
        assert np.shape(_Noise)[1] == _d, "dimension of noise should be the same as the dimension of data \n"
        assert np.shape(_Noise)[0] >= _m_i * _T, "number of noise should be equal to (or larger than) T"
    
    i_loc = 0
    j_loc = 0 # noise index
    for i in range(_K+_T):
        if i in noise_idxs:
            _X_extended[i*_m_i:(i+1)*_m_i,:] = _Noise[j_loc*_m_i:(j_loc+1)*_m_i,:]
            j_loc += 1
        else:
            _X_extended[i*_m_i:(i+1)*_m_i,:] = _input_array[i_loc*_m_i:(i_loc+1)*_m_i,:]
            i_loc += 1
    
    # return _X_extended for debugging
    # noise_idxs should be known to the server for decoding
    if _is_LCC == False:
        return BACC_Enc(_X_extended, _alpha_array, _z_array), _X_extended, noise_idxs
    else: 
        return LCC_Enc(_X_extended, _alpha_array, _z_array), _X_extended, noise_idxs

def BACC_Enc_Model_withNoise_v2(_net, _N, _K, _T, _sigma, _Noise_Alloc = None):
    '''
    Inputs:
    
    _net : pytorch model
    _N : number of worker nodes
    _K : number of submatrices 
    _T : privacy parameter = number of random matrices
    _sigma : variance

    _Noise_Alloc : location of random matrices
        if None: randomly assign (_T) locations out of ( _K + _T )
        else: assign (_T) locations accordingly. 
              e.g. when _K=3, T_2, it could be [0, 1, 0, 0, 1].

    Parmeters:
    _alpha_array : numpy [_K] array
    _z_array     : numpy [_N] array
    
    Output:
    _net_array : array of net, whose length is _N
    '''

    j_array_org = np.array(range(_N))
    stt_pos = int(np.floor((_N-_K-_T)/2))

    j_array = j_array_org[stt_pos:stt_pos+_K+_T]

    _alpha_array = np.cos((2*j_array+1)*math.pi/(2*_N)) #np.cos((2*j_array+1)*math.pi/(2*K))

    i_array = np.array(range(_N))
    _z_array = np.cos(i_array*2*math.pi/_N/2) # np.cos(i_array*2*math.pi/N/2)

    if _Noise_Alloc == None:
        noise_idxs = np.random.choice(range(_K+_T), _T, replace=False)
        noise_idxs = np.sort(noise_idxs)
    else:
        noise_idxs = _Noise_Alloc


    net_array = []
    w_array = []

    for n in range(_N):
        net_tmp = copy.deepcopy(_net)
        net_array.append(net_tmp)
        w_array.append(net_tmp.state_dict())

    net_tmp = copy.deepcopy(_net)
    w_tmp = net_tmp.state_dict()

    for k in w_tmp.keys():
        tmp1 = w_tmp[k].cpu().detach().numpy()
        cur_shape = tmp1.shape
        _d = np.prod(cur_shape)

        _W_extended = np.empty((1*(_K+_T),_d))

        for i in range(_K+_T):
            if i in noise_idxs:
                _W_extended[i:(i+1),:] = np.random.normal(0,_sigma,size=(1,_d))
            else:
                _W_extended[i:(i+1),:] = np.reshape(tmp1,(1,_d))
        coded_W = BACC_Enc(_W_extended, _alpha_array, _z_array)

        #print(k)
        #print(np.shape(coded_W))
        #print()

        for n in range(_N):
            tmp = np.reshape(coded_W[n,0,:],cur_shape)
            w_array[n][k] += - w_array[n][k] + torch.Tensor(tmp).cuda()

            #print(n,k,w_array[n][k])

    for n in range(_N):
        net_array[n].load_state_dict(w_array[n])

    return net_array

def BACC_Enc_Model_withNoise_v3(_net, _N, _K, _T, _sigma, _alpha_array, _z_array, _Noise_Alloc = None, _is_LCC = False):
    '''
    Inputs:
    
    _net : pytorch model
    _N : number of worker nodes
    _K : number of submatrices 
    _T : privacy parameter = number of random matrices
    _sigma : variance

    _alpha_array : numpy [_K+_T] array
    _z_array     : numpy [_N] array

    _Noise_Alloc : location of random matrices
        if None: randomly assign (_T) locations out of ( _K + _T )
        else: assign (_T) locations accordingly. 
              e.g. when _K=3, T_2, it could be [0, 1, 0, 0, 1].

    Parmeters:
    
    
    Output:
    _net_array : array of net, whose length is _N
    '''

    if _Noise_Alloc == None:
        noise_idxs = np.random.choice(range(_K+_T), _T, replace=False)
        noise_idxs = np.sort(noise_idxs)
    else:
        noise_idxs = _Noise_Alloc

    if _T == 0:
        noise_idxs = []

    net_array = []
    w_array = []

    for n in range(_N):
        net_tmp = copy.deepcopy(_net)
        net_array.append(net_tmp)
        w_array.append(net_tmp.state_dict())

    net_tmp = copy.deepcopy(_net)
    w_tmp = net_tmp.state_dict()

    for k in w_tmp.keys():
        tmp1 = w_tmp[k].cpu().detach().numpy()
        cur_shape = tmp1.shape
        _d = np.prod(cur_shape)

        _W_extended = np.empty((1*(_K+_T),_d))

        for i in range(_K+_T):
            if i in noise_idxs:
                _W_extended[i:(i+1),:] = np.random.normal(0,_sigma,size=(1,_d))
            else:
                _W_extended[i:(i+1),:] = np.reshape(tmp1,(1,_d))

        if _is_LCC == False:
            coded_W = BACC_Enc(_W_extended, _alpha_array, _z_array)
        else:
            coded_W = LCC_Enc(_W_extended, _alpha_array, _z_array)

        #print(k)
        #print(np.shape(coded_W))
        #print()

        for n in range(_N):
            tmp = np.reshape(coded_W[n,0,:],cur_shape)
            w_array[n][k] += - w_array[n][k] + torch.Tensor(tmp).cuda()

            #print(n,k,w_array[n][k])

    for n in range(_N):
        net_array[n].load_state_dict(w_array[n])

    return net_array

def BACC_Enc_Model_withNoise_v4(_net, _N, _K, _T, _sigma, _alpha_array, _z_array, _Noise_Alloc = None, _is_LCC = False):
    '''
    Inputs:
    
    _net : pytorch model
    _N : number of worker nodes
    _K : number of submatrices 
    _T : privacy parameter = number of random matrices
    _sigma : variance

    _alpha_array : numpy [_K+_T] array
    _z_array     : numpy [_N] array

    _Noise_Alloc : location of random matrices
        if None: randomly assign (_T) locations out of ( _K + _T )
        else: assign (_T) locations accordingly. 
              e.g. when _K=3, T_2, it could be [0, 1, 0, 0, 1].

    Parmeters:
    
    
    Output:
    _net_array : array of net, whose length is _N
    '''

    if _Noise_Alloc == None:
        noise_idxs = np.random.choice(range(_K+_T), _T, replace=False)
        noise_idxs = np.sort(noise_idxs)
    else:
        noise_idxs = _Noise_Alloc

    if _T == 0:
        noise_idxs = []

    net_array = []
    w_array = []

    for n in range(_N):
        net_tmp = copy.deepcopy(_net)
        net_array.append(net_tmp)
        w_array.append(net_tmp.state_dict())

    net_tmp = copy.deepcopy(_net)
    w_tmp = net_tmp.state_dict()

    for k in w_tmp.keys():
        tmp1 = w_tmp[k].cpu().detach().numpy()
        cur_shape = tmp1.shape
        _d = np.prod(cur_shape)

        _W_extended = np.empty((1*(_K+_T),_d))

        _w_cur = np.reshape(tmp1,(1,_d))
        _cur_power = np.sum(_w_cur * _w_cur) / _d

        print(k,_cur_power)

        for i in range(_K+_T):
            if i in noise_idxs:
                _W_extended[i:(i+1),:] = np.random.normal(0,_cur_power*_sigma,size=(1,_d))
            else:
                _W_extended[i:(i+1),:] = _w_cur

        if _is_LCC == False:
            coded_W = BACC_Enc(_W_extended, _alpha_array, _z_array)
        else:
            coded_W = LCC_Enc(_W_extended, _alpha_array, _z_array)

        #print(k)
        #print(np.shape(coded_W))
        #print()

        for n in range(_N):
            tmp = np.reshape(coded_W[n,0,:],cur_shape)
            w_array[n][k] += - w_array[n][k] + torch.Tensor(tmp).cuda()

            #print(n,k,w_array[n][k])

    for n in range(_N):
        net_array[n].load_state_dict(w_array[n])

    return net_array

def MutualInformationSecurity(_alpha_array_Signal, _alpha_array_Noise, _beta_array, _P, _sigma, _is_LCC = False):
    '''
    Calculate the mutual information security

    - input:
    alpha_array_Signal : size = K
    alpha_array_Noise  : size = T
    beta_array         : size = t // set of beta's assigned for colluding users 
    
    - output:
    mutual information security (scalor value)
    '''

    _t = len(_beta_array)
    _K = len(_alpha_array_Signal)
    _T = len(_alpha_array_Noise)
    
    _W = np.ones((_t,_K + _T))
    _W[:,1::2] = -1
    
    _alpha_array = np.concatenate((_alpha_array_Signal,_alpha_array_Noise))
#     print(_alpha_array)
    
    if _is_LCC == False:
        _U = np.reshape(_beta_array,(_t,1)) - np.reshape(_alpha_array,(1,_K+_T))
        _U = 1/_U
        _U = _U * _W
    
        for i in range(_t):
            denom = np.sum(_U[i,:])
            _U[i,:] = _U[i,:] / denom
    else:
        _U = gen_Lagrange_coeffs(_beta_array,_alpha_array,is_K1=0)
    
    _L       = _U[:,0:_K]
    _L_tilde = _U[:,_K:_K+_T]

    
    _SIG = np.matmul(_L,np.transpose(_L))
    _SIG_tilde = np.matmul(_L_tilde, np.transpose(_L_tilde))
    _SIG_tilde_Inv = np.linalg.inv(_SIG_tilde)
    
    _D = _P/_sigma * np.matmul(_SIG_tilde_Inv, _SIG)
        
    return np.log2(np.linalg.det(np.identity(_t) + _D))

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