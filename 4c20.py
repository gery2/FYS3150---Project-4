#4c)
import numpy as np
from numba import jit

k = 1 #m^2 kg s^-2 K^-1 (skalert)
T = 1 #temperature
beta = 1/(k*T)
L = 20 #dimensions
J = 1
N = 1000 #number of cycles

#matrix2 = [] #same direction
matrix = [] #define empty matrix
for i in range(L): #total row is L
    row=[]
    for j in range(L): #total column is L
        row.append(np.random.choice([-1,1])) #adding 0 value for each column for this row
    matrix.append(row) #add fully defined column into the row

#starter energy
energy = 0
for i in range(L):
    for j in range(L):
        energy += matrix[i][j]*(matrix[(i+1)%L][j] + matrix[i][(j+1)%L])

e_sum = 0.0; m_sum = 0.0; m2_sum = 0.0; e2_sum = 0.0; abs_m_sum = 0.0;

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
            matrix[xind][yind] *= -1; energy += dE
        else:
            r = np.random.uniform(0,1)
            if r <= np.exp(-beta*dE):
                matrix[xind][yind] *= -1
                energy += dE


    e_sum += energy
    m_sum += np.sum(matrix)
    m2_sum += (np.sum(matrix))**2
    e2_sum += energy**2    #E**2
    abs_m_sum += abs(np.sum(matrix))

E_mean = e_sum/N #expectation value divided by number of cycles
print('E_mean = ', E_mean)
abs_M_mean = abs_m_sum/N
print('M_mean = ', abs_M_mean)
C = (e2_sum/N - (e_sum/N)**2)/(k*T**2)
print('C = ', C)
X = (m2_sum/N - ((m_sum)/N)**2)/(k*T)
print('X = ', X)
