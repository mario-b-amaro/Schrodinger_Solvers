"""
SOLVER FOR TIME-DEPENDENT 3D SCHRÃ–DINGER EQUATION
MÃ¡rio B. Amaro
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sp
import time as time

beg=time.time()

hbar, m = 1.05457182e-34, 1.6726219e-27

def V(x,y,z,t):
    w=1.6022e-12/hbar
    return 0.5*m*w**2*(x**2+y**2+z**2)

x_min, x_max = -10e-15, 10e-15
y_min, y_max = -10e-15, 10e-15
z_min, z_max = -10e-15, 10e-15
N, M, L, T = 20, 20, 20, 20
t_min, t_max = 0, 1e-20

x = np.linspace(x_min, x_max, N)
y = np.linspace(y_min, y_max, M)
z = np.linspace(z_min, z_max, L)

t = np.linspace(t_min,t_max,T)

dx = x[1]-x[0]
dy = y[1]-y[0]
dz = z[1]-z[0]
dt = t[1]-t[0]

eig=1 # State of Interest [1-N]
eigs=int(N/2) # No of Eigenvalues to solve for

# Calculation of Time-Independent Eigenvalue of Interest

print("Solving Time-Independent Problem...")

H_size = (N-2) * (M-2) * (L-2)
H = np.zeros((H_size, H_size),dtype="complex_")

def index_map(j, k, l, M, L):
    return ((j-1)*(M-2) + (k-1))*(L-2) + (l-1)

for j in range(1, N-1):
    for k in range(1, M-1):
        for l in range(1, L-1):
            idx = index_map(j, k, l, M, L)
            H[idx, idx] = V(x[j], y[k], z[l],0) + hbar**2/(m*dx**2) + hbar**2/(m*dy**2) + hbar**2/(m*dz**2)
            
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

val,vec=sp.sparse.linalg.eigsh(H,k=eigs,which='SM')
val,vec=np.array(val),np.array(vec)
psi_eig=vec[:,eig-1]

print("Time-Independent Problem Solved")

toplot = psi_eig
toplot = toplot.reshape(N-2,M-2,L-2)
toplot = np.pad(toplot, pad_width=2, mode='constant', constant_values=0)
#mip = np.max(np.real(toplot*np.conj(toplot)), axis=1) # Plotting the slice of highest energy
#plt.imshow(mip, cmap='seismic', interpolation='spline36')
plt.imshow(np.real(toplot[int(L/2)]*np.conj(toplot[int(L/2)])), cmap='seismic', interpolation='spline36')
plt.title(f'Probability Density Eig {eig} (E={round(np.real(val[eig-1])/1.6022e-13,1)} MeV)')
plt.show()

toplot = psi_eig
toplot = toplot.reshape(N-2,M-2,L-2)
toplot = np.pad(toplot, pad_width=2, mode='constant', constant_values=0)
#mip = np.max(np.real(toplot), axis=1) # Plotting the slice of highest energy
#plt.imshow(mip, cmap='seismic', interpolation='spline36')
plt.imshow(np.real(toplot[int(L/2)]), cmap='seismic', interpolation='spline36')
plt.title(f'Re(ψ) Eig {eig} (E={round(np.real(val[eig-1])/1.6022e-13,1)} MeV)')
plt.show()

toplot = psi_eig
toplot = toplot.reshape(N-2,M-2,L-2)
toplot = np.pad(toplot, pad_width=2, mode='constant', constant_values=0)
#mip = np.max(np.imag(np.conj(toplot)), axis=1) # Plotting the slice of highest energy
#plt.imshow(mip, cmap='seismic', interpolation='spline36')
plt.imshow(np.imag(toplot[int(L/2)]), cmap='seismic', interpolation='spline36')
plt.title(f'Im(ψ) Eig {eig} (E={round(np.real(val[eig-1])/1.6022e-13,1)} MeV)')
plt.show()

print("Solving Time-Dependent Problem")

psi=psi_eig/np.trapz(np.abs(psi_eig)) # STATE TO BE PROPAGATED, CHANGE AT WILL
psi_time=[np.pad(np.reshape(psi,(N-2,M-2,L-2)),[(1,),(1,),(1,)],mode="constant")]

for q in range(T-1):
    A_size = (N-2)*(M-2)*(L-2)
    A=np.zeros((H_size,H_size),dtype="complex_")
    for j in range(1,N-1):
        for k in range(1,M-1):
            for l in range(1,L-1):
                idx=index_map(j,k,l,M,L)
                A[idx,idx]=V(x[j],y[k],z[l],t[q+1])+hbar**2/(m*dx**2)+hbar**2/(m*dy**2)+hbar**2/(m*dz**2)-1j*(hbar/dt)-val[eig-1]               
                if j>1: 
                    A[idx,index_map(j-1,k,l,M,L)]=-hbar**2/(2*m*dx**2)
                if j<N-2:
                    A[idx,index_map(j+1,k,l,M,L)]=-hbar**2/(2*m*dx**2)
                if k>1:
                    A[idx,index_map(j,k-1,l,M,L)]=-hbar**2/(2*m*dy**2)
                if k<M-2:
                    A[idx,index_map(j,k+1,l,M,L)]=-hbar**2/(2*m*dy**2)
                if l>1:
                    A[idx,index_map(j,k,l-1,M,L)]=-hbar**2/(2*m*dz**2)
                if l<L-2:
                    A[idx,index_map(j,k,l+1,M,L)]=-hbar**2/(2*m*dz**2)
    
    b=-1j*(hbar/dt)*psi
    psi=sp.sparse.linalg.spsolve(A,b)
    
    psi_time.append(np.pad(np.reshape(psi,(N-2,M-2,L-2)),[(1,),(1,),(1,)],mode="constant"))
            
rho=psi_time*np.conj(psi_time)

print("Runtime:",time.time()-beg)


def init():
    im.set_data(np.zeros((N,M)))

# Animate Rho

fig = plt.figure(figsize=(8,8))
im = plt.imshow(np.real(psi_time[0][int(L/2)]*np.conj(psi_time[0][int(L/2)])),cmap='seismic', interpolation='none',vmin=0,vmax=np.max(np.real(psi_time*np.conj(psi_time))))

def animate(i):
    im.set_data(np.real(psi_time[i][int(L/2)]*np.conj(psi_time[i][int(L/2)])).reshape(N,M))
    return im

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=T, interval=10)

# Animate Re(Psi)

fig = plt.figure(figsize=(8,8))
def animate(i):
    im.set_data(np.real(psi_time[i][int(L/2)]).reshape(N,M))
    return im
im = plt.imshow(np.real(psi_time[0][int(L/2)]),cmap='seismic', interpolation='none',vmin=-np.max(np.abs(np.real(psi_time))),vmax=np.max(np.abs(np.real(psi_time))))
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=T,interval=10)

# Animate Im(Psi)

fig = plt.figure(figsize=(8,8))
def animate(i):
    im.set_data(np.imag(psi_time[i][int(L/2)]).reshape(N,M))
    return im
im = plt.imshow(np.imag(psi_time[0][int(L/2)]),cmap='seismic', interpolation='none',vmin=-np.max(np.abs(np.imag(psi_time))),vmax=np.max(np.abs(np.imag(psi_time))))
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=T,interval=10)
