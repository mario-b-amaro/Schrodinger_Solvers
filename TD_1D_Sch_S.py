"""
SOLVER FOR TIME-DEPENDENT 1D SCHRODINGER EQUATION
Mario B. Amaro
marioam@kth.se
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time as time

beg=time.time()

x_min, x_max = -25e-15, 25e-15 # Domain in x
t_min, t_max = 0, 2e-20 # Domain in time

N,T = 100,500 # number of steps in x and t, respectively
x=np.linspace(x_min,x_max,N)
t=np.linspace(t_min,t_max,T)
dx=x[1]-x[0] # x step
dt=t[1]-t[0] # t step

eigs=int(N/10) # Number of eigenvalues solving for in the Time-Independent problem (eigs>eigs_int)
eigs_int=3 # Eigenstate of Interest

#hbar, m = 1.05457182e-34, 9.1093837e-31 # SI Units (Electron)
hbar, m = 1.05457182e-34, 1.6726219e-27 # SI Units (Proton)
#hbar, m = 1, 1 # Natural Units

def V(x,t): # Potential, change as needed
    w=1.6022e-12/hbar
    return 0.5*m*w**2*x**2

# Solving Time-Independent Problem to get the eigenvector of interest

print("Solving Time-Independent Problem...")

H=np.zeros((N-2)**2,dtype = "complex_").reshape(N-2,N-2) # Hamiltonian
for i in range(N-2):
    for j in range(N-2):
        if i==j:
            H[i,j]=V(x[i+1],0)+(hbar**2/(m*dx**2))
        elif np.abs(i-j)==1:
            H[i,j]=-(hbar**2/(2*m*dx**2))
        else:
            H[i,j]=0
            
val,vec=sp.sparse.linalg.eigs(H,k=eigs,which='SM') # Calculate eigenvalues from smallest magnitude to highest
val,vec=np.array(val),np.array(vec) # Print variable val for the calculated eigenvalues
psi_eig=vec[:,eigs_int-1] # Use the eigenvalue of interest as defined above

print("Time-Independent Problem Solved")

print("Solving Time-Independent Problem...")

psi=psi_eig/np.trapz(np.abs(psi_eig))*x[1:-1]/max(x) # THIS IS THE PROPAGATED STATE
psi_time=[psi]

for q in range(T-1): # Time propagation
    A=np.zeros((N-2)**2,dtype = "complex_").reshape(N-2,N-2)
    for i in range(N-2):
        for j in range(N-2):
            if i==j:
                A[i,j]=V(x[i+1],t[q])+(hbar**2/(m*dx**2))-val[eigs_int-1]-1j*(hbar/dt)
            elif np.abs(i-j)==1:
                A[i,j]=-(hbar**2/(2*m*dx**2))
            else:
                A[i,j]=0
    
    b=-1j*(hbar/dt)*psi
    psi=sp.sparse.linalg.spsolve(A,b)

    psi_time.append(psi/np.trapz(np.abs(psi)))
    
psi_time=np.pad(psi_time, [(0, ), (1, )], mode='constant') # Add the boundary psi(x_min)=psi(x_max)=0

rho=[psi_time[i]*np.conj(psi_time[i]) for i in range(len(psi_time))] # Probability density

# Plots

plt.imshow(np.transpose(np.real(rho)),aspect='auto',extent=[min(t),max(t),min(x),max(x)])
plt.title("Probability Density Evolution")
plt.xlabel('t (s)')
plt.ylabel('x (m)')
plt.show()

plt.plot(t,(np.real(psi_time)**2+np.imag(psi_time)**2)[:,int((N-2)/2)])
plt.title("Time-evolution of ρ(x=0,t)")
plt.xlabel('t(s)')
plt.show()

plt.plot(t,np.real(psi_time)[:,int((N-2)/2)],label='$Re(ψ)$')
plt.plot(t,np.imag(psi_time)[:,int((N-2)/2)],label='$Im(ψ)$')
plt.title("Time-evolution of ψ(x=0,t)")
plt.legend()
plt.xlabel('t(s)')
plt.show()

print("Runtime:",time.time()-beg)