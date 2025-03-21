import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sympy import sympify, lambdify, symbols

def parse_function(func_str):
    x = symbols('x')
    try:
        expr = sympify(func_str)
        func = lambdify(x, expr, modules=['numpy'])
        return func, expr
    except Exception as e:
        print(f"Error parsing function: {e}")
        return None, None

def compute_derivative(func, x_vals, dx=1e-6):
    return [(func(x + dx) - func(x - dx)) / (2 * dx) for x in x_vals]

def compute_integral(func, x_vals):
    integrals = []
    for x in x_vals:
        integral, _ = quad(func, x_vals[0], x)
        integrals.append(integral)
    return integrals

def plot_graphs(x_vals, y_vals, y_derivative=None, y_integral=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label="Original Function", color="blue")
    if y_derivative:
        plt.plot(x_vals, y_derivative, label="Derivative", color="red")
    if y_integral:
        plt.plot(x_vals, y_integral, label="Integral", color="green")
    plt.title("Function, Derivative, and Integral")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    print("Welcome to the Calculus-Powered Graphing App!")
    func_str = input("Enter a mathematical function (e.g., x**2 + 3*x + 5): ")
    func, expr = parse_function(func_str)
    
    if not func:
        return
    
    x_range = input("Enter the range of x values (e.g., -10,10): ").split(',')
    if len(x_range) != 2:
        print("Invalid range. Please enter two numbers separated by a comma.")
        return
    
    try:
        x_min, x_max = float(x_range[0]), float(x_range[1])
    except ValueError:
        print("Invalid range. Please enter valid numbers.")
        return
    
    x_vals = np.linspace(x_min, x_max, 500)
    y_vals = [func(x) for x in x_vals]
    
    print("Options:")
    print("1. Plot the derivative")
    print("2. Plot the integral")
    print("3. Plot both derivative and integral")
    choice = input("Enter your choice (1/2/3): ")
    
    y_derivative = None
    y_integral = None
    
    if choice == "1":
        y_derivative = compute_derivative(func, x_vals)
    elif choice == "2":
        y_integral = compute_integral(func, x_vals)
    elif choice == "3":
        y_derivative = compute_derivative(func, x_vals)
        y_integral = compute_integral(func, x_vals)
    else:
        print("Invalid choice. Exiting.")
        return
    
    plot_graphs(x_vals, y_vals, y_derivative, y_integral)

if __name__ == "__main__":
    main()