import numpy as np

def MSE(dx, f, x, val, lamb=8, k=0.1):
    a = dx + lamb*f*(k + np.tan(lamb*x))
    b = val

    loss = (a-b)**2
    return loss

def diff_loss(dx, f, x):
    loss = np.sum(MSE(dx, f, x, val=0)) / len(x)
    return loss

def boundary_loss(f0, u0=1, nabla=2):
    loss = nabla*(f0-u0)**2
    return loss

def loss_function(dx, f, x):
    loss = diff_loss(dx, f, x) + boundary_loss(f[0])
    return loss