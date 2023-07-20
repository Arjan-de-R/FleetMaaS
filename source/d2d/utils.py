import numpy as np
from MaaSSim.src_MaaSSim.d2d_demand import *

def zero_to_nan(indicator):
    indicator = indicator.astype(float)
    indicator[indicator == 0] = np.nan
    return indicator