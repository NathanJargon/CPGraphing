import numpy as np
from scipy.integrate import quad
import numdifftools as nd

def compute_derivative(func, x_vals):
    derivative_func = nd.Derivative(func)
    return derivative_func(x_vals)

def compute_integral(func, x_min, x_vals):
    return [quad(func, x_min, val)[0] for val in x_vals]