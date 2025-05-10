from sympy import sympify, symbols
from sympy.core.sympify import SympifyError

def parse_functions(function_str):
    """
    Parses a string of comma-separated functions into a list of SymPy expressions.

    Parameters:
    - function_str: A string containing functions separated by commas.

    Returns:
    - A list of SymPy expressions.

    Raises:
    - ValueError: If any function is invalid.
    """
    x = symbols('x')
    try:
        functions = [sympify(f.strip()) for f in function_str.split(",")]
        return functions
    except SympifyError as e:
        raise ValueError(f"Invalid function input: {e}")