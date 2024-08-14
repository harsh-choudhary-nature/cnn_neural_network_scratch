# Harsh Choudhary, 2103117
from scipy.signal import correlate2d
import numpy as np

def convolution3D(input:np.ndarray,filter:np.ndarray,mode='valid')->np.ndarray:
    ans = []
    for input_channel_index in range(input.shape[0]):

        ans.append(correlate2d(input[input_channel_index],filter[input_channel_index],mode=mode))

    ans = np.array(ans)
    ans = np.sum(ans,axis=0)
    return ans
