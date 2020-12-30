import numpy as np
import random

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