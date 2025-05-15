
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

# Example: your two MSE arrays
mse_1 = np.load('mse_values2000.npy')
mse_2 = np.load('mse_values800.npy')
mse_3 = np.load('mse_values1000.npy')

# Min-max scaling to 0–100 range
scaled_mse_1 = (mse_1 - mse_1.min()) / (mse_1.max() - mse_1.min()) * 100
scaled_mse_2 = (mse_2 - mse_2.min()) / (mse_2.max() - mse_2.min()) * 100
scaled_mse_3 = (mse_3 - mse_3.min()) / (mse_3.max() - mse_3.min()) * 100

# Plotting
plt.plot(scaled_mse_2, label='tmax=800')
plt.plot(scaled_mse_1, label='tmax=2000')
plt.plot(scaled_mse_3, label='tmax=1000')
plt.xlabel('Iteration')
plt.ylabel('Scaled MSE ')
plt.legend()
plt.savefig("mse_trendt", dpi=600, bbox_inches='tight')
plt.show()
