import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Getting started
    amp_data = np.load('amp_data.npz')['amp_data']
    
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(amp_data, 'b-')
    plt.subplot(1, 2, 2)
    plt.hist(amp_data, bins=100)
    plt.show()