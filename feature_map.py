import numpy as np

def phi(x,qubit):
    return 2*qubit*np.arccos(x)

def phi_plus(x,qubit): 
    return 2*qubit*np.arccos(x) + np.pi/2

def phi_min(x,qubit):
    return 2*qubit*np.arccos(x) - np.pi/2

#Create feature map
def U(qc, x,num_qubits, label):
    assert label <= 2*num_qubits, "Label cannot be bigger than 2*num_qubits!"

    #Return U if label = 0
    if label == 0:
        for i in range(num_qubits):
            qc.ry(phi(x,i),i)

    #Return U+ (for C+) if label in [1,num_qubits]
    elif label >= 1 and label <= num_qubits:
        for i in range(num_qubits):
            if i == label-1:
                qc.ry(phi_plus(x,i),i)
            else:
                qc.ry(phi(x,i),i)

    #Return U- (for C-) if label in [num_qubits+1,2*num_qubits]
    else:
        for i in range(num_qubits):
            if i == label-num_qubits-1:
                qc.ry(phi_min(x,i),i)
            else:
                qc.ry(phi(x,i),i)
    return qc