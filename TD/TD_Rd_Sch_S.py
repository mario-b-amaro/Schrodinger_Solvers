"""
SOLVER FOR TIME-DEPENDENT 1D SCHRÃ–DINGER EQUATION
MÃ¡rio B. Amaro
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time as time

beg=time.time()

a_0=5.29177210903e-11

r_min, r_max = 0, 50*a_0
t_min, t_max = 0, 5e-14

N,T = 100,500
r=np.linspace(r_min,r_max,N)
t=np.linspace(t_min,t_max,T)
dr=r[1]-r[0]
dt=t[1]-t[0]

l=1
eigs_int=2 # No. of Eigenvalues of interest
eigs=int(N/10) # No. of Eigenvalues to solve for

hbar, m = 1.05457182e-34, 9.1093837e-31 # SI Units (Electron)
#hbar, m = 1.05457182e-34, 1.6726219e-27 # SI Units (Proton)
#hbar, m = 1, 1 # Natural Units

def V(r,t):
    return -2.304e-28/r

# Calculation of Time-Independent Eigenvalue of Interest

print("Solving Time-Independent Problem...")

H=np.zeros((N-2)**2,dtype = "complex_").reshape(N-2,N-2)
for i in range(N-2):
    for j in range(N-2):
        if i==j:
            H[i,j]=V(r[i+1],0)+(hbar**2/(m*dr**2)+(hbar**2*l*(l+1))/(2*m*r[i+1]**2))
        elif np.abs(i-j)==1:
            H[i,j]=-(hbar**2/(2*m*dr**2))
        else:
            H[i,j]=0
            
val,vec=sp.sparse.linalg.eigs(H,k=eigs,which='SM')
val,vec=np.array(val),np.array(vec)
psi_eig=vec[:,eigs_int-1]

plt.plot(psi_eig)
plt.plot(psi_eig*np.conj(psi_eig))
plt.show()

print("Time-Independent Problem Solved")

print("Solving Time-Independent Problem...")

psi=psi_eig/np.trapz(np.abs(psi_eig))*np.cos(100*r[1:-1]/max(r)) # STATE TO BE SOLVED, CHANGE AT WILL
psi_time=[psi]

for w in range(T-1):
    A=np.zeros((N-2)**2,dtype = "complex_").reshape(N-2,N-2)
    for i in range(N-2):
        for j in range(N-2):
            if i==j:
                A[i,j]=V(r[i+1],t[w+1])+(hbar**2/(m*dr**2)+(hbar**2*l*(l+1))/(2*m*r[i+1]**2))-val[eigs_int-1]-1j*(hbar/dt)
            elif np.abs(i-j)==1:
                A[i,j]=-(hbar**2/(2*m*dr**2))
            else:
                A[i,j]=0
    
    b=-1j*(hbar/dt)*psi
    psi=sp.sparse.linalg.spsolve(A,b)

    psi_time.append(psi/np.trapz(np.abs(psi)))
    
psi_time=np.pad(psi_time, [(0, ), (1, )], mode='constant')
    
rho=[psi_time[i]*np.conj(psi_time[i]) for i in range(len(psi_time))]

# Plots

plt.imshow(np.transpose(np.real(rho)),aspect='auto',extent=[min(t),max(t),min(r),max(r)])
plt.title("Probability Density Evolution")
plt.xlabel('t (s)')
plt.ylabel('x (m)')
plt.show()
"""
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
"""

# Animation

%matplotlib qt

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(r, np.real(psi_time[0])) # Returns a tuple of line objects, thus the comma
line2, = ax.plot(r, np.imag(psi_time[0])) # Returns a tuple of line objects, thus the comma
ax.set_ylim([-1.1*np.max(np.abs(np.real(psi_time))),1.1*np.max(np.abs(np.real(psi_time)))])

for i in range(len(rho)):
    line1.set_ydata(np.real(psi_time[i]))
    line2.set_ydata(np.imag(psi_time[i]))
    fig.suptitle(f"t={round(t[i]/1e-15,2)} .1e-15 s")
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)


print("Runtime:",time.time()-beg)
