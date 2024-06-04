import numpy as np
from function_evaluation import func_and_deriv

def MSE(f, df, x, val, lamb=8, k=0.1):
    a = df + lamb*f*(k + np.tan(lamb*x))
    b = val

    loss = (a-b)**2
    return loss

def diff_loss(f, df, x):
    loss = np.sum(MSE(f, df, x, val=0)) / len(x)
    return loss

def boundary_loss(f0, u0=1, nabla=2):
    loss = nabla*(f0-u0)**2
    return loss

def loss_function(x,num_qubits,l,theta,shots_simulator):
    f, df = func_and_deriv(x,num_qubits,l,theta,shots_simulator)
    loss = diff_loss(f, df, x) #+ boundary_loss(f[0])
    return loss