import numpy as np
import matplotlib.pyplot as plt 

plt.rcParams['figure.figsize'] = [16,12]
plt.rcParams.update({'font.size':18})

# Create a simple signal with two frequencies 
dt = 0.001 # time step
t = np.arange(0,1,dt)
f = np.sin(2*np.pi*200*t) + np.sin(2*np.pi*100*t) # Sum of 2 frequencies 
f_clean = f 
f = f + 2.5*np.random.randn(len(t))  # Add noise

#plt.plot(t,f,color='c')
#plt.plot(t,f_clean,color='k')
#plt.xlim(t[0],t[-1])
#plt.legend()
#plt.show()


## Compute the Fast fourier transform 

n = len(f)
fhat = np.fft.fft(f,n)
PSD = fhat * np.conj(fhat) / n 
freq = (1/(dt*n)) * np.arange(n)
L = np.arange(1,np.floor(n/2),dtype='int')



# Filter small noise 
indices = PSD > 100 # Find all freq with large power 
fhat = indices * fhat # Zero out the small Fourier coffs 
ffilt = np.fft.ifft(fhat) # Convert back the signal without noise



# Plot 


fig,axs = plt.subplots(2,1)

plt.sca(axs[0])
plt.plot(t,f,color='c',label='Noisy')
plt.plot(t,f_clean,color='k',label = 'Clean')
plt.xlim(t[0],t[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(freq[L],PSD[L],color = 'c' ,label = 'Noisy')
plt.xlim(freq[L[0]],freq[L[-1]])
#plt.xticks(freq[L])
plt.legend()

plt.show()
