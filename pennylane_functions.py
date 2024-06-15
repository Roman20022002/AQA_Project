from pennylane import numpy as np
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
            qml.RY(phi(x,i),i) #, id = round(phi(x,i),2))

    #Return U+ (for C+) if label in [1,num_qubits]
    elif label >= 1 and label <= num_qubits:
        for i in range(num_qubits):
            if i == label-1:
                qml.RY(phi_plus(x,i),i) #, id = round(phi_plus(x,i),2))
            else:
                qml.RY(phi(x,i),i) #, id = round(phi(x,i),2))

    #Return U- (for C-) if label in [num_qubits+1,2*num_qubits]
    else:
        for i in range(num_qubits):
            if i == label-num_qubits-1:
                qml.RY(phi_min(x,i),i) #, id = round(phi_min(x,i),2))
            else:
                qml.RY(phi(x,i),i) #, id = round(phi(x,i),2))
    
    return 0

#Define the Hamiltonian
def H(n_qubits):
    ops = qml.operation.Tensor(*[qml.PauliZ(i) for i in range(n_qubits)])
    return ops

#Define circuit for the variational quantum classifier
def theta_gate(theta, i):
    
    assert len(theta)==3, "theta incorrect size"
    
    qml.RZ(theta[0], i) # , id = round(theta[0],2))
    qml.RX(theta[1], i) #, id = round(theta[1],2))
    qml.RZ(theta[2], i) #, id = round(theta[2],2))
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
        qml.Barrier()
    return 0

def build_circuit(x,num_qubits, label, theta, l):
    U(x, num_qubits, label)
    qml.Barrier()
    HEA(theta, num_qubits, l)
    hamiltonian = H(num_qubits)
    return qml.expval(hamiltonian)

def f_func(x,num_qubits,theta, l):
    label = 0
    dev = qml.device('default.qubit', wires=list(range(num_qubits)))
    circuit = qml.QNode(build_circuit, dev)
    result = circuit(x, num_qubits, label, theta, l)
    # qml.drawer.use_style('black_white')
    # fig, ax = qml.draw_mpl(circuit)(x, num_qubits, label, theta, l)
    # plt.show()
    return result

def dphi(x):
    return -1/np.sqrt(1-x**2)

def df_func(x,num_qubits,theta,l):
    C_plus = 0 
    C_minus = 0
    for i in range(1,2*num_qubits+1):
        if i <= num_qubits:
            dev = qml.device('default.qubit', wires=list(range(num_qubits)))
            circuit = qml.QNode(build_circuit, dev)
            result = circuit(x, num_qubits, i, theta, l)
            # qml.drawer.use_style('black_white')
            # fig, ax = qml.draw_mpl(circuit)(x, num_qubits, i, theta, l)
            # plt.show()
            C_plus += result
        else:
            dev = qml.device('default.qubit', wires=list(range(num_qubits)))
            circuit = qml.QNode(build_circuit, dev)
            result = circuit(x, num_qubits, i, theta, l)
            # qml.drawer.use_style('black_white')
            # fig, ax = qml.draw_mpl(circuit)(x, num_qubits, i, theta, l)
            # plt.show()
            C_minus += result

    return 1/4*dphi(x)*(C_plus-C_minus)

def func_and_deriv(x, num_qubits, theta, l):
    f = []
    df = []

    for i in x:
        f.append(f_func(i,num_qubits,theta,l) + (1-f_func(0, num_qubits, theta, l))) 
        df.append(df_func(i,num_qubits,theta,l))

    return np.array(f), np.array(df)

def MSE(f, df, x, val, lamb=8, k=0.1):
    a = df + lamb*f*(k + np.tan(lamb*x))
    b = val

    loss = (a-b)**2
    return loss

def u(x, x0, u0, kappa, lam=8):
    u_tilde = np.exp(-kappa*lam*x0)*np.cos(lam*x0)
    const = u0 - u_tilde
    return np.exp(-kappa*lam*x)*np.cos(lam*x) + const

def lin_reg(i, epochs):
    return 1 - i/epochs

def reg_loss(x, num_qubits, theta, l, n_iter, epochs, x0=0, u0=1, kappa=0.1, lam=8, step = 4):
    sigm = lin_reg(n_iter, epochs)
    x_reg = x[::step]

    f_reg, df_reg = func_and_deriv(x_reg, num_qubits, theta, l)
    u_reg = u(x_reg, x0, u0, kappa, lam)

    loss_array = sigm*(f_reg - u_reg)**2

    return np.sum(loss_array)

def diff_loss(f, df, x):
    loss = np.sum(MSE(f, df, x, val=0)) / len(x)
    return loss

def boundary_loss(f0, u0=1, nabla=2):
    loss = nabla*(f0-u0)**2
    return loss

def loss_function(x,num_qubits, theta, l, n_iter, epochs):
    f, df = func_and_deriv(x,num_qubits, theta, l)
    loss = diff_loss(f, df, x) #+ reg_loss(x, num_qubits, theta, l, n_iter, epochs) #+ boundary_loss(f[0]) 
    return loss