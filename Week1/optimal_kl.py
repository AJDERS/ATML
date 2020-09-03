def optimal_kl(bias, bound, epsilon=0.0001):
    '''
    Written by Anders Jess Pedersen, translated from MATLAB implementation by
    Yevgeny Seldin. Calculates the maximal bias `result` of a Bernoulli variable
    such that its KL-divergence from a Bernoulli variable with bias `bias` is bounded by
    `bound`, i.e. `result = argmax_y Dkl(bias||result) < bound`.

    Parameters: 
        bias (float): Bias of the Bernoulli variable.
        bound (float): The upper bound on the KL-divergence.
        epsilon (float): The precision of the iterative process.
          
    Returns: 
        result (float): argmax_y Dkl(bias||result) < bound.
    '''
    if any([(bias < 0), (bias > 1), (bound < 0)]):
        raise Exception('Wrong arguments')

    if (bound == 0.0):
        return bias

    result = (1 + bias) / 2
    step = (1 + bias) / 4

    if (bias > 0.0):
        p0 = bias
    else:
        p0 = 1.0
    
    while (step > epsilon):
        
        tmp = (1 - bias) / (1 - result)
        below_bound = (
            (bias * math.log(p0 / result) + (1 - bias) * math.log(tmp)) < bound
        )

        if below_bound:
            result += step
        else:
            result -= step
        step = step / 2
    return result