import numdifftools as nd
import numpy as np

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