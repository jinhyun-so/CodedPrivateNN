import numpy as np
import random
import math

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
    print ('@BACC_Enc: N,K, m_i=',_N,_K,_m_i,'\n')
    
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