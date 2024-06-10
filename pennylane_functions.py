import numpy as np
import pennylane as qml

#Create rotation angle functions
def phi(x,qubit):
    return 2*qubit*np.arccos(x)

def phi_plus(x,qubit): 
    return 2*qubit*np.arccos(x) + np.pi/2

def phi_min(x,qubit):
    return 2*qubit*np.arccos(x) - np.pi/2

#Create feature map
def U(x,num_qubits, label):
    assert label <= 2*num_qubits, "Label cannot be bigger than 2*num_qubits!"

    #Return U if label = 0
    if label == 0:
        for i in range(num_qubits):
            qml.RY(phi(x,i),i)

    #Return U+ (for C+) if label in [1,num_qubits]
    elif label >= 1 and label <= num_qubits:
        for i in range(num_qubits):
            if i == label-1:
                qml.RY(phi_plus(x,i),i)
            else:
                qml.RY(phi(x,i),i)

    #Return U- (for C-) if label in [num_qubits+1,2*num_qubits]
    else:
        for i in range(num_qubits):
            if i == label-num_qubits-1:
                qml.RY(phi_min(x,i),i)
            else:
                qml.RY(phi(x,i),i)
    
    return 0

#Define the Hamiltonian
def H(n_qubits):
    ops = qml.operation.Tensor(*[qml.PauliZ(i) for i in range(n_qubits)])
    return ops

#Define circuit for the variational quantum classifier
def theta_gate(theta, i):
    
    assert len(theta)==3, "theta incorrect size"
    
    qml.RZ(theta[0], i)
    qml.RX(theta[1], i)
    qml.RZ(theta[2], i)
    return 0

def entangling_gate(n):
    for i in range(0, n-1, 2):
            qml.CNOT([i, i+1])
    for i in range(1, n-1, 2):
            qml.CNOT([i, i+1])
    return 0

def HEA(theta, n, l):
    Theta = np.reshape(theta, (n,l,3))
    
    for i in range(l):
        for j in range(n):
            theta_gate(Theta[j,i,:], j) 
        entangling_gate(n)
    return 0

def build_circuit(x,num_qubits, label, theta, l):
    U(x, num_qubits, label)
    qml.Barrier()
    HEA(theta, num_qubits, l)
    hamiltonian = H(num_qubits)
    return qml.expval(hamiltonian)