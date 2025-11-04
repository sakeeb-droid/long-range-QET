from qiskit import QuantumCircuit, generate_preset_pass_manager, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import SparsePauliOp

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, EstimatorV2 as Estimator

from qiskit_aer import AerSimulator
import numpy as np

class MinimalQETCircuit:
    def __init__(self, h, k, v_measure=False):
        self.h = h
        self.k = k
        self.v_measure = v_measure
        self.alpha = -np.arcsin((1/np.sqrt(2))*(np.sqrt(1+h/np.sqrt(h**2+k**2))))
        self.phi = 0.5*np.arcsin((h*k)/np.sqrt((h**2+2*k**2)**2+(h*k)**2))
        self.qreg = QuantumRegister(2, 'q')
        self.creg = ClassicalRegister(2, 'c')
        self.qc = QuantumCircuit(self.qreg, self.creg)

    def build_circuit(self):
        self.qc.ry(2*self.alpha, self.qreg[0])
        self.qc.cx(self.qreg[0], self.qreg[1])
        self.qc.h(self.qreg[0])
        self.qc.measure(self.qreg[0], self.creg[0])

        with self.qc.if_test((self.creg[0], 0)):
            self.qc.ry(2*self.phi, self.qreg[1])
        with self.qc.if_test((self.creg[0], 1)):
            self.qc.ry(-2*self.phi, self.qreg[1])
        
        if self.v_measure:
            self.qc.h(self.qreg[1])
        
        self.qc.measure(self.qreg[1], self.creg[1])

    def get_circuit(self):
        return self.qc
    
    def draw_circuit(self):
        return self.qc.draw('mpl', initial_state=True)
    
    def get_counts(self, backend, shots=1024, optimization_level=1):
        self.backend = backend
        self.shots = shots
        self.optimization_level = optimization_level

        sampler = Sampler(mode=self.backend)
        pm = generate_preset_pass_manager(self.optimization_level, self.backend)
        job = sampler.run(pm.run([self.qc]), shots=self.shots)
        counts = job.result()

        return counts[0].data.c.get_counts()
    
    def calculate_EA(self):
        ene_A=(self.h**2)/(np.sqrt(self.h**2+self.k**2))
        error_A = []

        for orig_bitstring, count in self.get_counts(backend=self.backend, shots=self.shots).items():
            bitstring = orig_bitstring[::-1]
            ene_A += self.h*(-1)**int(bitstring[0])*count/self.shots

        for i in range(count):
            error_A.append(self.h*(-1)**int(bitstring[0]))

        std_A = np.std(error_A)/np.sqrt(self.shots)
        E_A = self.h**2/np.sqrt(self.h**2+self.k**2)
        print(f"Alice's exact local energy: {E_A}")
        print(f"Alice's measured local energy, E_A: {ene_A} ± {std_A}")

        return E_A, ene_A, std_A
    
    def calculate_EV(self):
        if self.v_measure == False:
            raise ValueError("To calculate the interacting energy, set v_measure=True when initializing the MinimalQETCircuit class.")
        else:
            ene_V=(2*self.k**2)/(np.sqrt(self.h**2+self.k**2))
            error_V = []

            for orig_bitstring, count in self.get_counts(backend=self.backend, shots=self.shots).items():
                bitstring = orig_bitstring[::-1]
                ene_V += 2*self.k*(-1)**int(bitstring[0])*(-1)**int(bitstring[1])*count/self.shots

            for i in range(count):
                error_V.append(2*self.k*(-1)**(int(bitstring[1])))

            std_V = np.std(error_V)/np.sqrt(self.shots)
            print(f"The interacting energy, E_V: {ene_V} ± {std_V}")

            return ene_V, std_V
        
    def calculate_EB(self):
        if self.v_measure == False:
            ene_B=(self.h**2)/(np.sqrt(self.h**2+self.k**2))
            error_B = []
            for orig_bitstring, count in self.get_counts(backend=self.backend, shots=self.shots).items():
                bitstring = orig_bitstring[::-1]
                ene_B += self.h*(-1)**int(bitstring[1])*count/self.shots

            for i in range(count):
                error_B.append(self.h*(-1)**int(bitstring[1]))

            std_B = np.std(error_B)/np.sqrt(self.shots)
            print(f"Bob's measured local energy, E_B: {ene_B} ± {std_B}")

            return ene_B, std_B
        else:
            raise ValueError("To calculate Bob's local energy, set v_measure=False when initializing the MinimalQETCircuit class.")