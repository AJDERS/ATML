import math
import pdb
import numpy as np
import matplotlib.pyplot as plt

def optimal_kl(bias, bound, epsilon, lower):
    '''
    Written by Anders Jess Pedersen, translated from MATLAB implementation by
    Yevgeny Seldin. Calculates the maximal bias `result` via binary search of 
    a Bernoulli variable such that its KL-divergence from a Bernoulli variable
    with bias `bias` is bounded by `bound`, i.e.
    `result = argmax_y Dkl(bias||result) < bound`.

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
    step = (1 - bias) / 4

    if (bias > 0.0):
        p0 = bias
    else:
        p0 = 1.0
    
    while (step > epsilon):
        result = min(1.0 - epsilon, result)
        kl = (bias * math.log(p0 / result) + (1.0 - bias) * math.log((1.0 - bias) / (1.0 - result)))
        if not lower:
            if kl < bound:
                result += step
            else:
                result -= step
        else:
            if kl > bound:
                result -= step
            else:
                result += step
        step = step / 2
    return result


def hoeffding_bound(phats, delta, n, lower):
    if not lower:
        bounds = phats + math.sqrt(math.log(1.0 / delta) / (2 * n))
    else:
        bounds = phats - math.sqrt(math.log(1.0 / delta) / (2 * n)) # check if corrent...
    return zip(phats, bounds)

def kl_bound(phats, delta, n, lower, epsilon):
    bound_for_kl = math.log((n + 1) / delta) / n
    bounds = np.array([optimal_kl(phat, bound_for_kl, epsilon, lower) for phat in phats])
    return zip(phats, bounds)


def pinsker_bound(phats, delta, n):
    bounds = phats + math.sqrt(math.log((n + 1.0) / delta) / (2 * n))
    return zip(phats, bounds)

def refined_pinsker_bound(phats, delta, n):
    bounds = phats + np.sqrt(2.0 * phats * math.log((n + 1.0) / delta) / (2 * n)) \
        + (2 * math.log((n + 1) / delta)) / n
    return zip(phats, bounds)

def remove_out_of_bounds(zipped_data):
    return [(x, y) for (x, y) in zipped_data if y < 1.0 and y > 0.0]

def make_plot(n = 1000, delta = 0.01, interval=[0.0, 1.0], lower=False, epsilon=2**(-52)):
    step_size = interval[1] / 1000
    phats = np.array([interval[0]+step_size*x for x in range(1000)])
    hoef_phats, hoef_bounds = zip(*remove_out_of_bounds(hoeffding_bound(phats, delta, n, lower)))
    kl_phats, kl_bounds = zip(*remove_out_of_bounds(kl_bound(phats, delta, n, lower, epsilon)))
    pin_phats, pin_bound = zip(*remove_out_of_bounds(pinsker_bound(phats, delta, n)))
    ref_pin_phats, ref_pin_bound = zip(*remove_out_of_bounds(refined_pinsker_bound(phats, delta, n)))

    _, ax = plt.subplots()
    ax.plot(hoef_phats, hoef_bounds, label='Hoeffding')
    ax.plot(kl_phats, kl_bounds, label='KL-inequality.')
    if not lower:
        ax.plot(pin_phats, pin_bound, label='Pinsker')
        ax.plot(ref_pin_phats, ref_pin_bound, label='Refined Pinsker')
    plt.legend()
    plt.grid()
    ax.set_ylabel('Bounds')
    ax.set_xlabel('Empirical Average')
    ax.set_title('Bounds as function of empirical average.')
    if lower:
        s = 'lower'
    else:
        s = 'upper'
    plt.savefig('bound_{}_{}_{}.png'.format(interval[0], interval[1], s))
    plt.show()
    plt.close()