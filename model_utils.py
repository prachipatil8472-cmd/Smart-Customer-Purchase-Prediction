
import numpy as np

def log_transform(values):
    return np.log1p(np.clip(values, 0, None))

