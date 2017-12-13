import numpy as np
from math import ceil

def pad_array (data, by, kind='zero'):
    N = len (data)
    result = np.zeros (N + 2 * by)
    result[by:-by] = data
    if kind == 'zero':
        return result
    if kind == 'weekly':
        for i in range (0, by):
            index = (-by + i + 7) % N
            result[i] = data[index]
        for i in range (-by, 0):
            index = (N + by + i - 7) % N
            result[N + 2*by + i] = data[index]

        return result

def make_pyramid (data, minimum=np.Inf):
    result = []
    kernel = np.r_[.0625, .25, .375, .25, .0625]
    minimum = min (minimum, len(kernel))

    while len(data) >= minimum:
        # blur and downsample
        next_data = np.convolve (pad_array (data, 2, 'weekly'), kernel, mode='valid')[::2]
        # upsample
        upsampled = upsample(next_data, to_shape=data.shape)
        # blur
        recovered = np.convolve (pad_array (upsampled, 2, 'zero'), kernel, mode='valid') * 2
        result.append (data - recovered)
        data = next_data

    result.append (next_data)
    return result

def max_pyramid_depth (N, minimum=np.Inf):
    minimum = min(minimum, 5)

    depth = 0
    while N >= minimum:
        N = ceil (N/2)
        depth += 1

    return depth + 1

def reconstruct_pyramid (pyramid):
    kernel = np.r_[.0625, .25, .375, .25, .0625]

    for i in range(len(pyramid)-1,0,-1):
        # upsample
        upsampled = upsample(pyramid[i], to_shape=pyramid[i-1].shape)
        # blur
        recovered = np.convolve (pad_array (upsampled, 2, 'zero'), kernel, mode='valid') * 2
        pyramid[i-1] += recovered

    return pyramid[0]

def upsample (data, to_shape=None, into=None, by=None, use_ceil=True):
    if into is None:
        into = np.zeros (to_shape)
    if by is None:
        if use_ceil:
            by = ceil (into.shape[0] / data.shape[0])
        else:
            by = into.shape[0] / data.shape[0]

        assert by - int(by) == 0, "Can't upsample by non-integer factor %f." % by

    into[::by] = data
    return into
