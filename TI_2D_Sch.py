"""
SOLVER FOR TIME-INDEPENDENT 2D SCHRODINGER EQUATION
Mario B. Amaro
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time as time

beg=time.time()

x_min, x_max = -25e-15, 25e-15
y_min, y_max = -25e-15, 25e-15
N, M = 100, 100

x = np.linspace(x_min, x_max, N)
y = np.linspace(y_min, y_max, M)

dx = x[1]-x[0]
dy = y[1]-y[0]

eigs=int(N/2) # No of Eigenvalues to solve for
eigs_int=6 # No of Eigenvalues of interest

#hbar, m = 1.05457182e-34, 9.1093837e-31 # SI Units (Electron)
hbar, m = 1.05457182e-34, 1.6726219e-27 # SI Units (Proton)
#hbar, m = 1, 1 # Natural Units

def V(x,y):
    w=1.6022e-12/hbar
    return 0.5*m*w**2*(x**2+y**2)

H_size = (N-2) * (M-2)
H = np.zeros((H_size, H_size))

for j in range(1, N-1):
    for k in range(1, M-1):
        idx = (j-1)*(M-2) + (k-1) 
        H[idx, idx] = V(x[j], y[k]) + hbar**2/(m*dx**2) + hbar**2/(m*dy**2)
        if j > 1:
            H[idx, idx-(M-2)] = -hbar**2/(2*m*dx**2)
        if j < N-2: 
            H[idx, idx+(M-2)] = -hbar**2/(2*m*dx**2)
        if k > 1: 
            H[idx, idx-1] = -hbar**2/(2*m*dy**2)
        if k < M-2:
            H[idx, idx+1] = -hbar**2/(2*m*dy**2)  
            
print("Hamiltonian Matrix Constructed.")

val,vec=sp.sparse.linalg.eigsh(H,k=eigs,which='SM')

print("Eigenproblem Solved.")

val,vec=np.array(val),np.array(vec)
z = np.argsort(val)
z = z[0:eigs_int]
energies=(val[z]/val[z][0])
print("Real Eigenvalues (Energy):",np.real(energies))
print("Real Eigenvalues (Non-normalized):",np.real(val[z]))

for i in range(eigs_int):
    toplot=vec[:,i]
    toplot=toplot.reshape(N-2,M-2)
    toplot = np.pad(toplot, pad_width=1, mode='constant', constant_values=0)
    plt.imshow(np.real(toplot*np.conj(toplot)),cmap='seismic', interpolation='none') # interpolation='spline36'
    plt.title(f'Probability Density Eig {i+1} (E={round(np.real(val[i])/1.6022e-13,1)} MeV)')
    plt.ylabel('y (m)')
    plt.xlabel('x (m)')
    plt.show()

print("Runtime:",time.time()-beg)