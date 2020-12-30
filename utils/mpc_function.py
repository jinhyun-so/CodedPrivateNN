import numpy as np
import random
from array import array
import math
import time

np.random.seed(42) # set the seed of the random number generator for consistency

# p = 15485863 # field size

def modular_inv(a,p):
    x, y, m = 1, 0, p
    while a > 1:
        q = a//m;
        t = m ;

        m = np.mod(a,m)
        a = t
        t = y

        y, x = x - np.int64(q)*np.int64(y), t

        if x<0:
            x = np.mod(x,p)
    return np.mod(x,p)

def divmod(_num, _den, _p):
    # compute num / den modulo prime p
    _num = np.mod(_num,_p)
    _den = np.mod(_den,_p)
    _inv = modular_inv(_den,_p)
    # print(_num,_den,_inv)
    return np.mod(np.int64(_num) * np.int64(_inv), _p)

def PI(vals,p):  # upper-case PI -- product of inputs
    accum = 1
    for v in vals:
        accum = np.mod(accum*v,p)
    return accum

def gen_Lagrange_coeffs(alpha_s,beta_s,p,is_K1=0):
    if is_K1==1:
        num_alpha = 1
    else:
        num_alpha = len(alpha_s)
    U = np.zeros((num_alpha, len(beta_s)),dtype='int64')
#         U = [[0 for col in range(len(beta_s))] for row in range(len(alpha_s))]
    #print(alpha_s)
    #print(beta_s)
    for i in range(num_alpha):
        for j in range(len(beta_s)):
            cur_beta = beta_s[j];

            den = PI([cur_beta - o   for o in beta_s if cur_beta != o], p)
            num = PI([alpha_s[i] - o for o in beta_s if cur_beta != o], p)
            U[i][j] = divmod(num,den,p)
            # for debugging
            # print(i,j,cur_beta,alpha_s[i])
            # print(test)
            # print(den,num) 
    return U.astype('int64')

def BGW_encoding(X,N,T,p):
    m = len(X)
    d = len(X[0])

    alpha_s = range(1,N+1)
    alpha_s = np.int64(np.mod(alpha_s,p))
    X_BGW = np.zeros((N,m,d),dtype='int64')
    R = np.random.randint(p,size=(T+1,m,d))
    R[0,:,:] = np.mod(X, p)

    for i in range(N):
        for t in range(T+1):
            X_BGW[i,:,:] = np.mod( X_BGW[i,:,:] + R[t,:,:]*(alpha_s[i]**t), p)
    return X_BGW

def gen_BGW_lambda_s(alpha_s, p):
    lambda_s = np.zeros((1, len(alpha_s)),dtype='int64')

    for i in range(len(alpha_s)):
            
        cur_alpha = alpha_s[i];

        den = PI([cur_alpha - o   for o in alpha_s if cur_alpha != o], p)
        num = PI([0 - o for o in alpha_s if cur_alpha != o], p)
        lambda_s[0][i] = divmod(num,den,p)
    return lambda_s.astype('int64')

def BGW_decoding(f_eval,worker_idx,p): # decode the output from T+1 evaluation points
    # f_eval     : [RT X d ]
    # worker_idx : [ 1 X RT]
    # output     : [ 1 X d ]

    #t0 = time.time()
    max = np.max(worker_idx) + 2
    alpha_s = range(1,max)
    alpha_s = np.int64(np.mod(alpha_s,p))
    alpha_s_eval = [alpha_s[i] for i in worker_idx]
    #t1 = time.time()
    # print(alpha_s_eval)
    lambda_s = gen_BGW_lambda_s(alpha_s_eval, p).astype('int64')
    #t2 = time.time()
    # print(lambda_s.shape)
    f_recon = np.mod(np.dot(lambda_s,f_eval), p)
    #t3 = time.time()
    #print 'time info for BGW_dec', t1-t0, t2-t1, t3-t2
    return f_recon


def LCC_encoding(X,N,K,T,p):
    m = len(X)
    d = len(X[0])
    # print(m,d,m//K)
    X_sub = np.zeros((K+T,m//K,d),dtype='int64')
    for i in range(K):
        X_sub[i] = X[i*m//K:(i+1)*m//K:]
    for i in range(K,K+T):
        X_sub[i] = np.random.randint(p, size=(m//K,d))

    n_beta = K+T
    stt_b, stt_a = -int(np.floor(n_beta/2)), -int(np.floor(N/2))
    beta_s, alpha_s = range(stt_b,stt_b+n_beta), range(stt_a,stt_a+N)
    alpha_s = np.array(np.mod(alpha_s,p)).astype('int64')
    beta_s = np.array(np.mod(beta_s,p)).astype('int64')

    U = gen_Lagrange_coeffs(alpha_s,beta_s,p)
    # print U

    X_LCC = np.zeros((N,m//K,d),dtype='int64')
    for i in range(N):
        for j in range(K+T):
            X_LCC[i,:,:] = np.mod(X_LCC[i,:,:] + np.mod(U[i][j]*X_sub[j,:,:],p),p)
    return X_LCC

def LCC_encoding_w_Random(X,R_,N,K,T,p):
    m = len(X)
    d = len(X[0])
    # print(m,d,m//K)
    X_sub = np.zeros((K+T,m//K,d),dtype='int64')
    for i in range(K):
        X_sub[i] = X[i*m//K:(i+1)*m//K:]
    for i in range(K,K+T):
        X_sub[i] = R_[i-K,:,:].astype('int64')

    n_beta = K+T
    stt_b, stt_a = -int(np.floor(n_beta/2)), -int(np.floor(N/2))
    beta_s, alpha_s = range(stt_b,stt_b+n_beta), range(stt_a,stt_a+N)

    alpha_s = np.array(np.mod(alpha_s,p)).astype('int64')
    beta_s = np.array(np.mod(beta_s,p)).astype('int64')

    #alpha_s = np.int64(np.mod(alpha_s,p))
    #beta_s = np.int64(np.mod(beta_s,p))

    U = gen_Lagrange_coeffs(alpha_s,beta_s,p)
    # print U

    X_LCC = np.zeros((N,m//K,d),dtype='int64')
    for i in range(N):
        for j in range(K+T):
            X_LCC[i,:,:] = np.mod(X_LCC[i,:,:] + np.mod(U[i][j]*X_sub[j,:,:],p),p)
    return X_LCC

def LCC_encoding_w_Random_partial(X,R_,N,K,T,p,worker_idx):
    m = len(X)
    d = len(X[0])
    # print(m,d,m//K)
    X_sub = np.zeros((K+T,m//K,d),dtype='int64')
    for i in range(K):
        X_sub[i] = X[i*m//K:(i+1)*m//K:]
    for i in range(K,K+T):
        X_sub[i] = R_[i-K,:,:].astype('int64')

    n_beta = K+T
    stt_b, stt_a = -int(np.floor(n_beta/2)), -int(np.floor(N/2))
    beta_s, alpha_s = range(stt_b,stt_b+n_beta), range(stt_a,stt_a+N)
    alpha_s = np.array(np.mod(alpha_s,p)).astype('int64')
    beta_s = np.array(np.mod(beta_s,p)).astype('int64')
    alpha_s_eval = [alpha_s[i] for i in worker_idx]

    U = gen_Lagrange_coeffs(alpha_s_eval,beta_s,p)
    # print U

    N_out = U.shape[0]
    X_LCC = np.zeros((N_out,m//K,d),dtype='int64')
    for i in range(N_out):
        for j in range(K+T):
            X_LCC[i,:,:] = np.mod(X_LCC[i,:,:] + np.mod(U[i][j]*X_sub[j,:,:],p),p)
    return X_LCC


def LCC_decoding(f_eval,f_deg,N,K,T,worker_idx,p):
    RT_LCC = f_deg*(K+T-1) + 1

    n_beta = K #+T
    stt_b, stt_a = -int(np.floor(n_beta/2)), -int(np.floor(N/2))
    beta_s, alpha_s = range(stt_b,stt_b+n_beta), range(stt_a,stt_a+N)
    alpha_s = np.array(np.mod(alpha_s,p)).astype('int64')
    beta_s = np.array(np.mod(beta_s,p)).astype('int64')
    alpha_s_eval = [alpha_s[i] for i in worker_idx]
    
    U_dec = gen_Lagrange_coeffs(beta_s,alpha_s_eval,p)

    # print U_dec 

    f_recon = np.mod((U_dec).dot(f_eval),p)

    return f_recon.astype('int64')

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
         
    
def matrix_multiply(X1, X2, p, T): # secure matrix multiplication (single reshare for degree reduction)
        
    t0_matmul = time.time()

    if dot_ON:
        mult = np.mod(X1.dot(X2), p) # multiply the current secret shares
    else:
        mult = np.mod(np.matmul(X1,X2), p) # multiply the current secret shares

    t_matmul = time.time() - t0_matmul

    # degree reduction
    t0 = time.time()

    row, col  = np.shape(mult)
        
    Q = np.empty((2*T+1, row, col), dtype='int64') # store secret shares
        
    for j in range(1, 2*T+2): # workers 1 to 2T+1 sends shares
     
        if rank == j:
            
            Qss = BGW_encoding(mult,N,T,p) # XXX create secret shares for all workers
            Q[j-1, :, :] = Qss[j-1, :, :] # own secret share
                
            for j in range(1, rank) +  range(rank+1,N+1): # secret share q
                data = np.reshape(Qss[j-1,:,:],row*col)
                comm.Send(data, dest=j) # sent data to worker j
#                     comm.send(Qss[j-1], dest=j)
        else:
            data = np.empty(row*col, dtype='int64')
            comm.Recv(data, source=j)
            Q[j-1,:,:] = np.reshape(data,(row,col)) # coefficients for the polynomial 
        
    Lcoeffs = gen_BGW_lambda_s(range(1, 2*T+2), p) # XXX need a vector of lagrange coeffs for 1 to N (in fact only 1 to 2T+1 is enough) but it doesn't matter much, also can take this out of the function and compute it once and store to speed up later
        
        
    out = np.empty((row, col), dtype='int64')
                   
    for j in range(2*T+1):        
        out += Lcoeffs[0][j] * Q[j,:,:] # index 0 to 2T+1, XXX MAKE THIS FINITE FIELD OPERATION WITH MOD
        
    t_deg_reduction = time.time() - t0
    return np.mod(out, p), t_matmul, t_deg_reduction

def MultPassive(A_SS_T, B_SS_T, R_SS_T, R_SS_2T, N,T,p):
    # A_SS_T, B_SS_T : [N x (size of A)], [N x (size of B)] : should have 3 dimensions
    
    # print("size of AB =", np.shape(A_SS_T)[1], np.shape(B_SS_T)[2])
    
    AB_SS_2T = np.empty((N,np.shape(A_SS_T)[1], np.shape(B_SS_T)[2]))
    for i in range(N):
        AB_SS_2T[i,:,:] = np.mod(np.matmul(A_SS_T[i,:,:], B_SS_T[i,:,:]), p)
    delta_SS_2T = np.mod(AB_SS_2T - R_SS_2T,p)
    
    delta_SS_2T = np.reshape(delta_SS_2T,(N,np.prod(delta_SS_2T.shape[1:]))) 
    
    worker_idx = random.sample(range(N), 2*T+1 ) # XXX
    delta = BGW_decoding(delta_SS_2T[worker_idx,:],worker_idx,p)
    
    # print(delta)
    delta = np.reshape(delta,(np.shape(A_SS_T)[1], np.shape(B_SS_T)[2]))

    AB_SS_T = np.mod(delta + R_SS_T, p)
    
    return AB_SS_T.astype('int64')

def TruncPr(a_BGW, k, m, p, N, T):
    # assert 2**(k+1) <= p, "Check the condition: (k,m,p)"
    # a_BGW : [N X 1 X d] where N: # of workers, d: length vector a 

    b_BGW = np.mod(a_BGW + 2**(k-1), p)

    r1 = np.random.randint(2**m,size=a_BGW.shape[1:])
    r2 = np.random.randint(2**(k-m),size=a_BGW.shape[1:])
    #print(r1)
    #print(r2)

    r1_BGW = BGW_encoding(r1,N,T,p)
    r2_BGW = BGW_encoding(r2,N,T,p)

    r_BGW = np.mod((2**m)*r2_BGW + r1_BGW , p)

    #print(a_BGW.shape)
    #print(r_BGW.shape)

    c_BGW = np.mod( b_BGW + r_BGW ,p)


    RT_BGW = T+1 # XXX
    worker_idx = random.sample(range(N),RT_BGW) # XXX

    c = BGW_decoding(c_BGW[worker_idx,0,:],worker_idx,p)
    #print(c)

    c_prime = np.mod(c, 2**m)

    a_prime_BGW = np.mod(c_prime - r1_BGW, p)

    d_BGW = np.mod(a_BGW- a_prime_BGW, p)
    
    d_BGW = divmod(d_BGW, 2**m, p)

    return d_BGW.astype('int64')