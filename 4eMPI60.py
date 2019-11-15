#4e phase transitions L =60
#how to run: mpiexec -n 6 python 4eMPI60.py
from time import perf_counter
t1_start = perf_counter()

from mpi4py import MPI
comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

k = 1 #m^2 kg s^-2 K^-1 (skalert)
T = np.linspace(2.0,2.4,12) #temperature
beta = 1/(k*T)
L = 60 #dimensions
J = 1
N = 10**6 #number of cycles
burn = 40000 #ignore all cycles up to this point

@jit(nopython=True)
def func(k,T,beta,L,J,N):
    #matrix2 = [] #same direction
    matrix = np.zeros((L,L)) #define empty matrix
    for i in range(L): #total row is L
        for j in range(L):
            matrix[i,j] = np.random.choice(np.array([-1,1]))
            #matrix[i,j] = 1 #need one example with all in one direction
    #starter energy
    energy = 0
    for i in range(L):
        for j in range(L):
            energy -= matrix[i][j]*(matrix[(i+1)%L][j] + matrix[i][(j+1)%L])

    e_sum = 0.0; m_sum = 0.0; m2_sum = 0.0; e2_sum = 0.0; abs_m_sum = 0.0

    for i in range(N):
        for j in range(L**2):
            xind = np.random.randint(L)
            yind = np.random.randint(L)
            #her må vi ta med alle naboene fordi vi ser på kun ett element
            chosen_one = matrix[xind][yind]
            nabo_høyre = matrix[xind][(yind + 1)%L]
            nabo_under = matrix[(xind + 1)%L][yind]
            nabo_venstre = matrix[xind][(yind - 1)%L]
            nabo_over = matrix[(xind - 1)%L][yind]

            dE = 2*J*matrix[xind][yind]*(nabo_høyre+nabo_venstre+nabo_over+nabo_under)
            if dE <= 0:
                matrix[xind][yind] *= -1
                energy += dE
            else:
                r = np.random.uniform(0,1)
                if r <= np.exp(-beta*dE):
                    matrix[xind][yind] *= -1
                    energy += dE
        if i > burn:
            e_sum += energy
            m_sum += np.sum(matrix)
            m2_sum += (np.sum(matrix))**2
            e2_sum += energy**2    #E**2
            abs_m_sum += abs(np.sum(matrix))


    return e_sum, m_sum, m2_sum, e2_sum, abs_m_sum

E_mean = []
abs_M_mean = []
C = []
X = []
M = N
N = N - burn

for i in T: #iterating over each element in T
    e_sum, m_sum, m2_sum, e2_sum, abs_m_sum = func(k,i,1/(k*i),L,J,M) #beta also differs
    C.append((e2_sum/N - (e_sum/N)**2)/(k*i**2)/L**2) #i here too instead of T
    X.append((m2_sum/N - ((abs_m_sum)/N)**2)/(k*i)/L**2) #i here too instead of T
    e_sum = e_sum/L**2
    m_sum = m_sum/L**2
    m2_sum = m2_sum/L**2
    e2_sum = e2_sum/L**2
    abs_m_sum = abs_m_sum/L**2
    E_mean.append(e_sum/N) #expectation value divided by number of cycles
    abs_M_mean.append(abs_m_sum/N)


print(' Rank: ', rank)
E_mean = np.array(E_mean); abs_M_mean = np.array(abs_M_mean); C = np.array(C); X = np.array(X)
E_total = np.zeros_like(E_mean)
M_total = np.zeros_like(abs_M_mean)
C_total = np.zeros_like(C)
X_total = np.zeros_like(X)

comm.Reduce([E_mean, MPI.DOUBLE], [E_total, MPI.DOUBLE], op=MPI.SUM, root=0)
comm.Reduce([abs_M_mean, MPI.DOUBLE], [M_total, MPI.DOUBLE], op=MPI.SUM, root=0)
comm.Reduce([C, MPI.DOUBLE], [C_total, MPI.DOUBLE], op=MPI.SUM, root=0)
comm.Reduce([X, MPI.DOUBLE], [X_total, MPI.DOUBLE], op=MPI.SUM, root=0)

print('numprocs = ', numprocs)

if rank ==0:
    E_mean_avg = E_total/numprocs
    abs_M_mean_avg = M_total/numprocs
    C_avg = C_total/numprocs
    X_avg = X_total/numprocs


t1_stop = perf_counter()
print("Elapsed time during the whole program in seconds:",
                                        t1_stop-t1_start)
if rank ==0:
    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(T, E_mean_avg)
    plt.title('Expectation value for <E> as function of T (L=%d)' %L)

    plt.ylabel('expectation value for <E>')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(T, abs_M_mean_avg)
    plt.title('Expectation value for <|M|> as function of T (L=%d)' %L)

    plt.ylabel('expectation value for <|M|>')
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(T, C_avg)
    plt.title('The specific heat C as function of T (L=%d)' %L)
    plt.xlabel('T')
    plt.ylabel('specific heat C')
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(T, X_avg)
    plt.title('The susceptibility X as function of T (L=%d)' %L)
    plt.xlabel('T')
    plt.ylabel('susceptibility X')
    plt.grid()

    plt.show()
