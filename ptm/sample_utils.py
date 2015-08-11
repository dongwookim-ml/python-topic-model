import numpy as np

def sampling_from_dist(prob):
    """
    highly optimized sampling method
    from multinomial distribution prob
    """
    c_sum = prob.cumsum()
    thr = c_sum[-1] * np.random.rand()
    return (c_sum < thr).cumsum()[-1]


def sampling_from_dict(key_val_dict):
    val_sum = sum(key_val_dict.values())
    thr = val_sum * np.random.rand()

    keys = key_val_dict.keys()
    tmp = 0
    for key in keys:
        tmp += key_val_dict[key]
        if tmp > thr:
            return key
