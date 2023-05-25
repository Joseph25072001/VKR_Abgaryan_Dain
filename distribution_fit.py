from distfit import distfit
import pandas as pd
import numpy as np

def fit_distribution(metric_name, delta):
    dist = distfit(distr='popular')
    dist.fit_transform(delta, verbose=1)
    best_fitted_distribution = dist.summary['name'][0]
    score = dist.summary['score'][0]
    loc = dist.summary['loc'][0]
    scale = dist.summary['scale'][0]
    params = dist.summary['params'][0]
    return (metric_name, best_fitted_distribution, score, loc, scale, params)