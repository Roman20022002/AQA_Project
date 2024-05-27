import numpy as np

def theta_gate(qc, theta, i):
    
    assert len(theta)==3, "theta incorrect size"
    
    qc.rz(theta[0], i)
    qc.rx(theta[1], i)
    qc.rz(theta[2], i)
    
    return 0

def entangling_gate(qc, n):

    for i in range(0, n-1, 2):
            qc.cx(i, i+1)
    for i in range(1, n-1, 2):
            qc.cx(i, i+1)

    return 0

def HEA(qc, theta, n, l):

    Theta = np.reshape(theta, (n,l,3))
    
    for i in range(l):
        for j in range(n):

            theta_gate(qc, Theta[j,i,:], j) 
        entangling_gate(qc, n)
    qc.measure_all()
    return qc