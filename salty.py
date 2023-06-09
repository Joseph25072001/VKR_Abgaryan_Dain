from itertools import combinations
from multiprocessing import Pool
from tqdm import tqdm
from hashlib import sha256
import scipy
import numpy as np
import pandas as pd

# Function to apply effect into the metric

def apply_effect(nominator, relative_shift, denominator=None):
    """
    Applies effect to an array
    Parameters
    ----------
    nominator: np.ndarray(2d)
        given metric

    relative_shift: float
        relative shift for a given metric

    metric_name: str
        name of the metric for which to apply the effect

    denominator: np.ndarray(2d) or None
        given denominator for the metric
    """
#1st way: injection of theoretical distributions of treatment effects into metric values
    mu, sigma = (relative_shift + 1) * 1, 0.2*(1+relative_shift)
    n = np.random.normal(mu, sigma, size=len(nominator))
#     e = np.random.exponential(1*(1+relative_shift), size=len(nominator))
#     g = np.random.gamma(1*(1+relative_shift), 1*(1+relative_shift), size=len(nominator))
#     u = np.random.uniform(0.5*(1+relative_shift), 1.5*(1+relative_shift), len(nominator))
#     l = np.random.lognormal(relative_shift,relative_shift,len(nominator))
    
    return n*np.where(nominator==0,1e-5,nominator)

#2nd way: constant increase
    #return nominator * (relative_shift + 1)
    
#3d way: injection of average values into zero metric values
#     nominator = np.array(sorted(nominator))
#     average = np.average(metric[nominator>0])
#     increment = relative_shift * nominator.sum()
#     users_to_replace = round(increment/average)
#     nominator[0:users_to_replace] = average
#     return nominator

#4th way: injection of median values into zero metric values
#     nominator = np.array(sorted(nominator))
#     median = np.average(metric[nominator>0])
#     increment = relative_shift * nominator.sum()
#     users_to_replace = round(increment/median)
#     nominator[0:users_to_replace] = median
#     return nominator


## Function for p-value calculations
def run_salt(salt, data, split_count, nominator, method, effects, denominator, split_by):
    df = data.copy()
    df['split'] = df[split_by].apply(lambda x: int(sha256((str(x) + str(salt)).encode()).hexdigest()[:6], 16) % split_count)

    combination = list(combinations([i for i in range(split_count)], 2))

    p_values_per_combo = []
    power_per_combo = []

    for pair in combination:
        values_a_nom, values_b_nom = df.loc[df['split'] == pair[0], nominator], df.loc[df['split'] == pair[1], nominator]


        if denominator is not None:
            values_a_den, values_b_den = df.loc[df['split'] == pair[0], denominator], df.loc[df['split'] == pair[1], denominator]
            p_value = method(values_a_nom, values_a_den, values_b_nom, values_b_den).pvalue
            p_values_per_combo.append(p_value)
            power_per_combo.append(method(apply_effect(values_a_nom, effects[pair[0]], denominator=values_a_den), values_a_den,
                                          apply_effect(values_b_nom, effects[pair[1]], denominator=values_b_den), values_b_den).pvalue)

        else:
            p_value = method(values_a_nom, values_b_nom).pvalue
            p_values_per_combo.append(p_value)
            power_per_combo.append(method(apply_effect(values_a_nom, effects[pair[0]]),
                                          apply_effect(values_b_nom, effects[pair[1]])).pvalue)

    return p_values_per_combo, power_per_combo