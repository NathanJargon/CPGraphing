from scipy.integrate import quad

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