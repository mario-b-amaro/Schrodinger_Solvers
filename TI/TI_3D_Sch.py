"""
SOLVER FOR TIME-INDEPENDENT 3D SCHRODINGER EQUATION
Mario B. Amaro
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time as time

beg=time.time()

x_min, x_max = -10e-15, 10e-15
y_min, y_max = -10e-15, 10e-15
z_min, z_max = -10e-15, 10e-15
N, M, L = 20, 20, 20

x = np.linspace(x_min, x_max, N)
y = np.linspace(y_min, y_max, M)
z = np.linspace(z_min, z_max, L)

dx = x[1]-x[0]
dy = y[1]-y[0]
dz = z[1]-z[0]

eigs=N # No Eigenvalues to solve for
eigs_int=10 # No of Eigenvalues of interest

hbar, m = 1.05457182e-34, 9.1093837e-31
hbar, m = 1.05457182e-34, 1.6726219e-27 # SI Units (Proton)
#hbar, m = 1, 1 # Natural Units

def V(x,y,z):
    w=1.6022e-12/hbar
    return 0.5*m*w**2*(x**2+y**2+z**2)

H_size = (N-2) * (M-2) * (L-2)
H = np.zeros((H_size, H_size))

def index_map(j, k, l, M, L):
    return ((j-1)*(M-2) + (k-1))*(L-2) + (l-1)

for j in range(1, N-1):
    for k in range(1, M-1):
        for l in range(1, L-1):
            idx = index_map(j, k, l, M, L)
            H[idx, idx] = V(x[j], y[k], z[l]) + hbar**2/(m*dx**2) + hbar**2/(m*dy**2) + hbar**2/(m*dz**2)
            
            if j > 1: 
                H[idx, index_map(j-1, k, l, M, L)] = -hbar**2/(2*m*dx**2)
            if j < N-2:
                H[idx, index_map(j+1, k, l, M, L)] = -hbar**2/(2*m*dx**2)
            if k > 1:
                H[idx, index_map(j, k-1, l, M, L)] = -hbar**2/(2*m*dy**2)
            if k < M-2:
                H[idx, index_map(j, k+1, l, M, L)] = -hbar**2/(2*m*dy**2)
            if l > 1:
                H[idx, index_map(j, k, l-1, M, L)] = -hbar**2/(2*m*dz**2)
            if l < L-2:
                H[idx, index_map(j, k, l+1, M, L)] = -hbar**2/(2*m*dz**2)

print("Hamiltonian Matrix Constructed.")

val,vec=sp.sparse.linalg.eigs(H,k=eigs,which='SM')

print("Eigenproblem Solved.")

val,vec=np.array(val),np.array(vec)
z = np.argsort(val)
z = z[0:eigs_int]
energies=(val[z]/val[z][0])
print("Real Eigenvalues (Energy) [Normalized to $E_0$]:",np.real(energies))
energies_nonnorm=val[z]
print("Real Eigenvalues (Energy) [Not Normalized]:",np.real(energies_nonnorm))

for i in range(eigs_int):
    toplot = vec[:, i]
    toplot = toplot.reshape(N-2,M-2,L-2)
    toplot = np.pad(toplot, pad_width=1, mode='constant', constant_values=0)
    #toplot = np.real(toplot)
    mip = np.max(np.real(toplot*np.conj(toplot)), axis=2) # Plotting the slice of highest energy
        
    plt.imshow(mip, cmap='seismic', interpolation='spline36')
    plt.title(f'Probability Density Eig {i+1} (E={round(np.real(energies[i]), 3)})')
    plt.show()

print("Runtime:",time.time()-beg)
