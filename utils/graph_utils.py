import numpy as np
from scipy.integrate import quad
import numdifftools as nd

def compute_derivative(func, x_vals):
    """
    Computes the numerical derivative of the given function at specified x values.

    Parameters:
    - func: The function to differentiate (must be callable).
    - x_vals: A NumPy array of x values where the derivative is evaluated.

    Returns:
    - A NumPy array containing the derivative values at each x in x_vals.
    """
    derivative_func = nd.Derivative(func)  # Create a numerical derivative function
    return derivative_func(x_vals)  # Evaluate the derivative at x_vals

def compute_integral(func, x_min, x_vals):
    """
    Computes the numerical integral of the given function from x_min to each x in x_vals.

    Parameters:
    - func: The function to integrate (must be callable).
    - x_min: The lower limit of integration.
    - x_vals: A NumPy array of x values where the integral is evaluated.

    Returns:
    - A list containing the integral values from x_min to each x in x_vals.
    """
    return [quad(func, x_min, val)[0] for val in x_vals]  # Compute definite integral for each x