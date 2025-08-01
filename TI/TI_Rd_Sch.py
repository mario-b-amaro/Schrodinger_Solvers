"""
SOLVER FOR TIME-INDEPENDENT RADIAL SCHRODINGER EQUATION
Mario B. Amaro
"""
 
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import time

beg=time.time()

# 1D Schrodinger Radial

r_min, r_max = 0.1, 15000 # fm
N = 2000 #Number of Meshpoints

r=np.linspace(r_min,r_max,N)
dr=r[1]-r[0]

eigs=int(N/10) # No. of Eigenvalues to solve for
eigs_int=4 # No. of Eigenvalues of interest

#hbar, m = 1.05457182e-34, 1.6726219e-27 # SI Units
hbar, m = 1, 0.511 # Natural Units

l=0
Z=1 # proton number, 1 for Hydrogen

def V(r): # All natural units
    e=0.30282212088
    hbar=1
    epsilon_0=1
    c=1

    V_c=-e**2/(4*np.pi*epsilon_0)*(Z/r)
    
    return V_c # Note output is in MeV

print("Constructing Hamiltonian Matrix...")

H=np.zeros((N-2)**2).reshape(N-2,N-2)
for i in range(N-2):
    for j in range(N-2):
        if i==j:
            H[i,j]=V(r[i+1])+(hbar**2/(m*dr**2)+(hbar**2*l*(l+1))/(2*m*r[i+1]**2))
        elif np.abs(i-j)==1:
            H[i,j]=-(hbar**2/(2*m*dr**2))
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
    u = []
    u = np.append(u,vec[:,z[i]])
    u = np.append(u,0)
    u = np.insert(u,0,0)
    rho=np.real(u*np.conj(u))
    plt.plot(r,rho,lw=3, label=f"Eig. {i+1} (E={round(np.real(val[i])*10**6,2)} eV)")
    plt.xlabel('x', size=20)
    plt.ylabel('|$\psi$(x)|$^2$',size=20)
plt.legend()
plt.title('Normalized Probability Density',size=20)
plt.grid()
plt.show()

print("Runtime:",time.time()-beg)
