"""
FDM SOLVER FOR TIME-INDEPENDENT 1D SCHRODINGER EQUATION
Mario B. Amaro
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time

beg=time.time()

# 1D Schrodinger

x_min, x_max = -2, 2 # fm (if in Natural Units), m (if in SI)
N = 1000 # Number of mesh points

x=np.linspace(x_min,x_max,N)
dx=x[1]-x[0]

eigs=int(N/10) # No. of Eigenvalues to solve for
eigs_int=7 # No. of Eigenvalues to display

#hbar, m = 1.05457182e-34, 9.1093837e-31 # SI Units (Electron)
#hbar, m = 1.05457182e-34, 1.6726219e-27 # SI Units (Proton)
hbar, m = 1, 1 # Natural Units

# EDIT POTENTIAL AS NEEDED

def V(x): 
    w=10
    return 0.5*m*w**2*x**2 # Harmonic Oscillator

print("Constructing Hamiltonian Matrix...")

H=np.zeros((N-2)**2).reshape(N-2,N-2)
for i in range(N-2):
    for j in range(N-2):
        if i==j:
            H[i,j]=V(x[i+1])+(hbar**2/(m*dx**2))
        elif np.abs(i-j)==1:
            H[i,j]=-1*(hbar**2/(2*m*dx**2))
        else:
            H[i,j]=0

print("Hamiltonian Matrix Constructed.")

val,vec=sp.sparse.linalg.eigsh(H,k=eigs,which='SM')

print("Eigenproblem Solved.")

val,vec=np.array(val),np.array(vec)
z = np.argsort(val)
z = z[0:eigs_int]
energies=(val[z]/val[z][0])
print("Real Eigenvalues (Normalized):",np.real(energies))
print("Real Eigenvalues (Non-normalized):",np.real(val[z]))

plt.figure(figsize=(12,10))
for i in range(eigs_int):
    y = []
    y = np.append(y,vec[:,z[i]])
    y = np.append(y,0)
    y = np.insert(y,0,0)
    rho=np.real(y*np.conj(y))
    plt.plot(x,rho,lw=3, label=f"Eig. {i+1} (E={round(np.real(energies[i]),2)})")
    plt.xlabel('x', size=20)
    plt.ylabel('|$\psi$(x)|$^2$',size=20)
plt.legend()
plt.title('Normalized Probability Density',size=20)
plt.grid()
plt.show()

print("Runtime:",time.time()-beg)
