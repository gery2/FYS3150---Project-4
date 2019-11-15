#4d probability
from time import perf_counter
t1_start = perf_counter()

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

k = 1 #m^2 kg s^-2 K^-1 (skalert)
T = 2.4 #temperature
beta = 1/(k*T)
L = 20 #dimensions
J = 1
N = 10**5 #number of cycles. should use close to a million.
N_array = np.arange(1,N+1,1)

@jit(nopython=True)
def func(k,T,beta,L,J,N):
    #matrix2 = [] #same direction
    matrix = np.zeros((L,L)) #define empty matrix
    for i in range(L): #total row is L
        for j in range(L):
            matrix[i,j] = np.random.choice(np.array([-1,1]))

    #starter energy
    energy = 0
    for i in range(L):
        for j in range(L):
            energy += matrix[i][j]*(matrix[(i+1)%L][j] + matrix[i][(j+1)%L])

    e_sum = 0.0; m_sum = 0.0; m2_sum = 0.0; e2_sum = 0.0; abs_m_sum = 0.0; accepted_configurations = 0

    energy_array = np.zeros(N)
    energy_mean_array = np.zeros(N)
    magnet_array = np.zeros(N)
    acc_config_array = np.zeros(N)

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
                accepted_configurations += 1
            else:
                r = np.random.uniform(0,1)
                if r <= np.exp(-beta*dE):
                    matrix[xind][yind] *= -1
                    energy += dE
                    accepted_configurations += 1

        e_sum += energy
        energy_array[i] = energy #using this array to plot histogram to count energies. (P(E))
        m_sum += np.sum(matrix)
        m2_sum += (np.sum(matrix))**2
        e2_sum += energy**2    #E**2
        abs_m_sum += abs(np.sum(matrix))

        energy_mean_array[i] = e_sum/(i+1)
        magnet_array[i] = abs_m_sum/(i+1)
        acc_config_array[i] = accepted_configurations

    return e_sum, m_sum, m2_sum, e2_sum, abs_m_sum, energy_mean_array, magnet_array, acc_config_array, accepted_configurations, energy_array

e_sum, m_sum, m2_sum, e2_sum, abs_m_sum, energy_mean_array, magnet_array, acc_config_array, accepted_configurations, energy_array = func(k,T,beta,L,J,N)

E_mean = e_sum/N #expectation value divided by number of cycles
print('E_mean = ', E_mean)
abs_M_mean = abs_m_sum/N
print('M_mean = ', abs_M_mean)
C = (e2_sum/N - (e_sum/N)**2)/(k*T**2)
print('C = ', C)
X = (m2_sum/N - ((m_sum)/N)**2)/(k*T)
print('X = ', X)
sigma = e2_sum/N - (e_sum/N)**2
print('sigma = ', sigma)




fig = plt.figure()
ax = fig.add_subplot(111)
n, bins, rectangles = ax.hist(energy_array, 50, density=True)
fig.canvas.draw()
plt.title('Probability P(E) for L = 20')
plt.xlabel('energies')
plt.ylabel('probability')
plt.show()

'''
#histogram of P(E) but wrong y axis
plt.hist(energy_array[40000:]) #P(E)
plt.title('Probability P(E) for L = 20')
plt.xlabel('energies')
plt.ylabel('number of occurences')
plt.show()
'''




'''
#plot energy per cycle
plt.figure(figsize=(10,7))
plt.plot(N_array, energy_mean_array)
plt.ylabel('Mean energy')
plt.xlabel('Monte Carlo cycles')

plt.title('Mean energy equilibrium study')
plt.show()


#plot magnetisation per cycle
plt.figure(figsize=(10,7))
plt.plot(N_array, magnet_array)
plt.ylabel('Magnetisation')
plt.xlabel('Monte Carlo cycles')

plt.title('magnetisation equilibrium study')
plt.show()
'''

t1_stop = perf_counter()
print("Elapsed time during the whole program in seconds:",
                                        t1_stop-t1_start)
