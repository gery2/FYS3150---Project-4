#4a)
import numpy as np

k = 1 #m^2 kg s^-2 K^-1 (skalert)
T = 1
beta = 1/(k*T)
J = 1

E = [-8, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 0, -8, 0]
M = [4, 2, 0, 2, 2, 0, -2, 0, 2, 0 ,-2, 0, 0, -2, -4, -2]

Z = 0; M_mean = 0; X_sum = 0; C_sum = 0; E_mean = 0

for i in range(16):
    Z += np.exp(-E[i]*beta)

for i in range(16):
    E_mean += (E[i]*np.exp(-E[i]*beta))/Z

for i in range(16):
    C_sum += (E[i]**2*np.exp(-E[i]*beta))/Z

C = (C_sum - E_mean**2)/(k*T**2)

for i in range(16):
    M_mean += abs(M[i]*np.exp(-E[i]*beta))/Z

for i in range(16):
    X_sum += (M[i]**2*np.exp(-E[i]*beta))/Z

X = (X_sum - M_mean**2)/(k*T)

print('Z = ', Z)
print('E_mean = ', E_mean)
print('C = ', C)
print('M_mean = ', M_mean)
print('X = ', X)
print(' ')

Z_new = 4*np.cosh(8*beta*J) + 12 #12+2e(-8JB)+2e(8JB)
print('Z-analytical = ', Z_new)
E_new = (-32*np.cosh(8))/Z_new #2 of each microstate 2*8
print('E-analytical = ', E_new)
C_new = ((256*np.cosh(8))/Z_new - E_new**2)/(k*T**2) #2 of each microstate 2*64
print('C-analytical = ', C_new)
M_new = (abs(4*np.exp(8)) + abs(-4*np.exp(8))+16)/Z_new
print('M-analytical = ', M_new)
X_new = ((32 + 32*(np.exp(8)))/Z_new - M_new**2)/(k*T) #no cosh
print('X-analytical = ', X_new)
















#
