#4f extracting the critical temperature
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

C_peaks = [2.7, 2.75, 2.82, 2.9] #for L=100 first, then descending.

#Tc from L = 100,80,60,40
L = np.array([2.2760000000000002, 2.2760000000000002, 2.2853333333333334, 2.2853333333333334])

a, Tc = np.polyfit(1/L, C_peaks, deg=1) #finding Tc at L = inf (at x=0)
print('a = ', a, 'Tc = ', Tc)


plt.plot(1/L, C_peaks)
plt.show()
