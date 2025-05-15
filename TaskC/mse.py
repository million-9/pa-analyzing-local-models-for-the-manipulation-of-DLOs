
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Example data (MSE values) – 1D array
mse_values =np.load('controlmse.npy')  # Example data

# 1) Find the peaks
peak_indices= [ 0,6, 17 ,33, 46, 59, 69, 80, 93,len(mse_values)-1]# height=0 => find all local maxima above 0
print("Peak indices:", peak_indices)
print("Peak values:", mse_values[peak_indices])

# 2) Define a helper function for min–max scaling
def min_max_scale(segment):
    seg_min = np.min(segment)
    seg_max = np.max(segment)
    # Avoid division by zero if segment is constant
    if seg_max == seg_min:
        return np.zeros_like(segment)
    return (segment - seg_min) / (seg_max - seg_min) *100

# 3) Split at peaks:
#    We'll create segments from each peak to the next peak.
#    If you also want to include the segment before the first peak or after the last peak,
#    you can handle that by adding indices at the start/end.
segments = []
peak_indices_sorted = np.sort(peak_indices)

for i in range(len(peak_indices_sorted) - 1):
    start = peak_indices_sorted[i]
    end = peak_indices_sorted[i + 1]
    # slice from start to end (inclusive) so we can see the shape around each peak
    segment = mse_values[start : end]
    # 4) Scale this segment
    scaled_segment = min_max_scale(segment)
    segments.append(scaled_segment)

# Plot each scaled segment on the same figure
plt.figure()
for i, seg in enumerate(segments, start=1):
    # We'll plot from index 0..(len(seg)-1) so each segment starts at x=0
    x_vals = np.arange(len(seg))
    plt.plot(x_vals, seg, label=f'Cfg {i}')

plt.xlabel('Iterations')
plt.ylabel('Scaled MSE')
plt.legend()
plt.savefig("mse_trend", dpi=600, bbox_inches='tight')
plt.show()
