from feature_map import *
from VQC import *
import numpy as np
import qiskit as qk
from qiskit_aer import AerSimulator

def create_circuit(x,n,l,Theta,label):
    qc = qk.QuantumCircuit(n)
    qc = U(qc, x, n, label)
    qc.barrier()
    qc = HEA(qc, Theta, n, l)
    return qc

def expect_value(qc, shots):
    #run circuit
    simulator = AerSimulator()
    result = simulator.run(qc,shots=shots).result()


    #Get counts and normalize them
    counts = result.get_counts()
    total_counts = sum(counts.values())
    counts_normalized = {state: counts[state]/total_counts for state in counts}

    #Get keys of dictionary
    keys = list(counts_normalized.keys())

    #Calculate expectation value of Z^n 
    expectation_value = 0
    for i in range(len(keys)):
        tmp = keys[i]

        #Extract number of zeros and ones
        num_zeros = tmp.count('0')
        num_ones = tmp.count('1')

        #Determine eigenvalue of operator
        eigenvalue = 1**num_zeros * (-1)**num_ones
        
        #Add to expectation value
        expectation_value += eigenvalue * counts_normalized.get(tmp)
    return expectation_value

def f_func(x,n,l,Theta,shots):
    label = 0
    circuit = create_circuit(x,n,l,Theta,label)
    expectation = expect_value(circuit,shots)
    return expectation

def dphi(x):
    return -1/np.sqrt(1-x**2)

def df_func(x,n,l,Theta,shots):
    C_plus = 0 
    C_minus = 0
    for i in range(1,2*n+1):
        if i <= n:
            circuit = create_circuit(x,n,l,Theta,i)
            expectation = expect_value(circuit,shots)
            C_plus += expectation
        else:
            circuit = create_circuit(x,n,l,Theta,i)
            expectation = expect_value(circuit,shots)
            C_minus += expectation

    return 1/4*dphi(x)*(C_plus-C_minus)

def func_and_deriv(x,n,l,Theta,shots):
    f = []
    df = []

    for i in x:
        f.append(f_func(i,n,l,Theta,shots))
        df.append(df_func(i,n,l,Theta,shots))
    
    return np.array(f), np.array(df)