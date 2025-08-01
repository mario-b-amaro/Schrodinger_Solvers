"""
SOLVER FOR TIME-DEPENDENT 2D SCHRÃ–DINGER EQUATION
MÃ¡rio B. Amaro
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sp
import time as time

beg=time.time()

hbar, m = 1.05457182e-34, 1.6726219e-27

def V(x,y,t):
    w=1.6022e-12/hbar
    return 0.5*m*w**2*(x**2+(y*(t_max-t)/t_max)**2)

x_min, x_max = -10e-15, 10e-15
y_min, y_max = -10e-15, 10e-15
t_min, t_max = 0, 5e-14
N, M, T = 35, 35, 200

x = np.linspace(x_min, x_max, N)
y = np.linspace(y_min, y_max, M)
t = np.linspace(t_min,t_max,T)

dx = x[1]-x[0]
dy = y[1]-y[0]
dt = t[1]-t[0]

eig=3 # State of Interest [1-N]
eigs=int(N/2) # No of Eigenvalues to solve for

# Calculation of Time-Independent Eigenvalue of Interest

print("Solving Time-Independent Problem...")

H_size = (N-2) * (M-2)
H = np.zeros((H_size, H_size),dtype="complex_")

for j in range(1, N-1):
    for k in range(1, M-1):
        idx = (j-1)*(M-2) + (k-1) 
        H[idx, idx] = V(x[j], y[k],0) + hbar**2/(m*dx**2) + hbar**2/(m*dy**2)
        if j > 1:
            H[idx, idx-(M-2)] = -hbar**2/(2*m*dx**2)
        if j < N-2: 
            H[idx, idx+(M-2)] = -hbar**2/(2*m*dx**2)
        if k > 1: 
            H[idx, idx-1] = -hbar**2/(2*m*dy**2)
        if k < M-2:
            H[idx, idx+1] = -hbar**2/(2*m*dy**2)  

val,vec=sp.sparse.linalg.eigsh(H,k=eigs,which='SM')
val,vec=np.array(val),np.array(vec)
psi_eig=vec[:,eig-1]

print("Time-Independent Problem Solved")

toplot=psi_eig
toplot=toplot.reshape(N-2,M-2)
toplot = np.pad(toplot, pad_width=1, mode='constant', constant_values=0)
plt.imshow(np.real(toplot*np.conj(toplot)),cmap='seismic', interpolation='none')
plt.title(f'Probability Density Eig {eig} (E={round(np.real(val[eig-1])/1.6022e-13,1)} MeV)')
plt.show()

plt.imshow(np.real(toplot),cmap='seismic', interpolation='none')
plt.show()

plt.imshow(np.imag(toplot),cmap='seismic', interpolation='none')
plt.show()

print("Solving Time-Dependent Problem")

psi=psi_eig/np.trapz(np.abs(psi_eig)) # TO PROPAGATE, CHANGE AT WILL
psi_time=[np.pad(np.reshape(psi**2,(N-2,M-2)),[(1,),(1,)],mode="constant")]

for q in range(T-1):
    A_size = (N-2)*(M-2)
    A=np.zeros((A_size,A_size),dtype="complex_")
    for j in range(1, N-1):
        for k in range(1, M-1):
            idx = (j-1)*(M-2) + (k-1) 
            A[idx, idx] = V(x[j], y[k], t[q+1]) + hbar**2/(m*dx**2) + hbar**2/(m*dy**2)-val[eig-1]-1j*(hbar/dt)
            if j > 1:
                A[idx, idx-(M-2)] = -hbar**2/(2*m*dx**2)
            if j < N-2: 
                A[idx, idx+(M-2)] = -hbar**2/(2*m*dx**2)
            if k > 1: 
                A[idx, idx-1] = -hbar**2/(2*m*dy**2)
            if k < M-2:
                A[idx, idx+1] = -hbar**2/(2*m*dy**2)
    
    b=-1j*(hbar/dt)*psi
    psi=sp.sparse.linalg.spsolve(A,b)
    
    psi_time.append(np.pad(np.reshape(psi,(N-2,M-2)),[(1,),(1,)],mode="constant"))
            
rho=psi_time*np.conj(psi_time)

print("Runtime:",time.time()-beg)


def init():
    im.set_data(np.zeros((N,M)))

# Animate Rho

fig = plt.figure(figsize=(8,8))
im = plt.imshow(np.real(psi_time[0]*np.conj(psi_time[0])),cmap='seismic', interpolation='none',vmin=0,vmax=np.max(np.real(psi_time*np.conj(psi_time))))

def animate(i):
    im.set_data(np.real(psi_time[i]*np.conj(psi_time[i])).reshape(N,M))
    return im

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=T,
                               interval=100)

"""
# Animate Re(Psi)

fig = plt.figure(figsize=(8,8))
def animate(i):
    im.set_data(np.real(psi_time[i]).reshape(N,M))
    return im
im = plt.imshow(np.real(psi_time[0]),cmap='seismic', interpolation='none',vmin=-np.max(np.abs(np.real(psi_time))),vmax=np.max(np.abs(np.real(psi_time))))
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=T,
                               interval=10)

# Animate Im(Psi)

fig = plt.figure(figsize=(8,8))
def animate(i):
    im.set_data(np.imag(psi_time[i]).reshape(N,M))
    return im
im = plt.imshow(np.imag(psi_time[0]),cmap='seismic', interpolation='none',vmin=-np.max(np.abs(np.imag(psi_time))),vmax=np.max(np.abs(np.imag(psi_time))))
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=T,
                               interval=10)
"""