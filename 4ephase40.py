#4e phase transitions L =40
from time import perf_counter
t1_start = perf_counter()

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

k = 1 #m^2 kg s^-2 K^-1 (skalert)
T = np.linspace(2.0,2.4,10) #temperature
beta = 1/(k*T)
L = 40 #dimensions
J = 1
N = 10**4 #number of cycles
burn = 5000 #ignore all cycles up to this point

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


t1_stop = perf_counter()
print("Elapsed time during the whole program in seconds:",
                                        t1_stop-t1_start)

fig = plt.figure()

plt.subplot(2, 2, 1)
plt.plot(T, E_mean)
plt.title('Expectation value for <E> as function of T (L=%d)' %L)

plt.ylabel('expectation value for <E>')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(T, abs_M_mean)
plt.title('Expectation value for <|M|> as function of T (L=%d)' %L)

plt.ylabel('expectation value for <|M|>')
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(T, C)
plt.title('The specific heat C as function of T (L=%d)' %L)
plt.xlabel('T')
plt.ylabel('specific heat C')
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(T, X)
plt.title('The susceptibility X as function of T (L=%d)' %L)
plt.xlabel('T')
plt.ylabel('susceptibility X')
plt.grid()

plt.show()
