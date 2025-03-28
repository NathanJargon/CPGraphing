import random
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QPushButton, QLineEdit, QLabel, QHBoxLayout, QCheckBox, QMessageBox
)
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sympy import symbols, sympify, lambdify
from utils.graph_utils import compute_derivative, compute_integral
from ui.styles import DARK_MODE_STYLE
import matplotlib.pyplot as plt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CPGraphing")
        self.setGeometry(100, 100, 1270, 800)

        # Apply dark mode
        self.setStyleSheet(DARK_MODE_STYLE)

        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.central_widget.setLayout(self.layout)

        # Graph placeholder
        self.figure, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Set the graph background to match dark mode
        self.ax.set_facecolor("#2b2b2b")
        self.figure.patch.set_facecolor("#2b2b2b")
        self.ax.tick_params(colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color("white")

        # Input fields and button layout
        self.input_layout = QHBoxLayout()

        # Function input
        self.function_label = QLabel("Function:")
        self.function_label.setFont(QFont("Arial", 12))
        self.input_layout.addWidget(self.function_label)

        self.function_input = QLineEdit()
        self.function_input.setFont(QFont("Arial", 12))
        self.function_input.setPlaceholderText("e.g., x**2 + 3*x + 5")
        self.input_layout.addWidget(self.function_input)

        # Range inputs
        self.range_label = QLabel("Range:")
        self.range_label.setFont(QFont("Arial", 12))
        self.input_layout.addWidget(self.range_label)

        self.x_min_input = QLineEdit()
        self.x_min_input.setFont(QFont("Arial", 12))
        self.x_min_input.setPlaceholderText("x_min")
        self.x_min_input.setFixedWidth(80)
        self.input_layout.addWidget(self.x_min_input)

        self.x_max_input = QLineEdit()
        self.x_max_input.setFont(QFont("Arial", 12))
        self.x_max_input.setPlaceholderText("x_max")
        self.x_max_input.setFixedWidth(80)
        self.input_layout.addWidget(self.x_max_input)

        # Add input layout to the main layout
        self.layout.addLayout(self.input_layout)

        # Checkboxes for graph filtering
        self.checkbox_layout = QHBoxLayout()
        self.show_function = QCheckBox("Show Function")
        self.show_function.setChecked(True)
        self.checkbox_layout.addWidget(self.show_function)

        self.show_derivative = QCheckBox("Show Derivative")
        self.show_derivative.setChecked(True)
        self.checkbox_layout.addWidget(self.show_derivative)

        self.show_integral = QCheckBox("Show Integral")
        self.show_integral.setChecked(True)
        self.checkbox_layout.addWidget(self.show_integral)

        self.layout.addLayout(self.checkbox_layout)

        # Buttons
        self.button_layout = QHBoxLayout()

        self.plot_button = QPushButton("Plot")
        self.plot_button.setFont(QFont("Arial", 12))
        self.plot_button.clicked.connect(self.plot_graphs)
        self.button_layout.addWidget(self.plot_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setFont(QFont("Arial", 12))
        self.reset_button.clicked.connect(self.reset_graph)
        self.button_layout.addWidget(self.reset_button)

        self.random_button = QPushButton("Random Input")
        self.random_button.setFont(QFont("Arial", 12))
        self.random_button.clicked.connect(self.generate_random_input)
        self.button_layout.addWidget(self.random_button)

        self.save_button = QPushButton("Save Graph")
        self.save_button.setFont(QFont("Arial", 12))
        self.save_button.clicked.connect(self.save_graph)
        self.button_layout.addWidget(self.save_button)

        self.layout.addLayout(self.button_layout)

    def plot_graphs(self):
        function_str = self.function_input.text()
        x_min = self.x_min_input.text()
        x_max = self.x_max_input.text()
        x = symbols('x')

        try:
            x_min = float(x_min) if x_min else -10
            x_max = float(x_max) if x_max else 10
            if x_min >= x_max:
                raise ValueError("x_min must be less than x_max")

            func = sympify(function_str)
            func_lambdified = lambdify(x, func, 'numpy')

            x_vals = np.linspace(x_min, x_max, 500)
            y_vals = func_lambdified(x_vals)
            derivative_vals = compute_derivative(func_lambdified, x_vals)
            integral_vals = compute_integral(func_lambdified, x_min, x_vals)

            self.ax.clear()

            if self.show_function.isChecked():
                self.ax.plot(x_vals, y_vals, label="Original Function", color="blue", linewidth=2)
            if self.show_derivative.isChecked():
                self.ax.plot(x_vals, derivative_vals, label="First Derivative", color="red", linewidth=2)
            if self.show_integral.isChecked():
                self.ax.plot(x_vals, integral_vals, label="Integral", color="green", linewidth=2)

            self.ax.set_title("Function, Derivative, and Integral", fontsize=16, fontweight="bold", color="white")
            self.ax.set_xlabel("x", fontsize=12, color="white")
            self.ax.set_ylabel("y", fontsize=12, color="white")
            self.ax.legend(loc="upper left", fontsize=10)
            self.ax.grid(True, linestyle="--", alpha=0.7)
            self.ax.set_facecolor("#2b2b2b")
            self.figure.patch.set_facecolor("#2b2b2b")

            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def reset_graph(self):
        self.ax.clear()
        self.canvas.draw()
        self.function_input.clear()
        self.x_min_input.clear()
        self.x_max_input.clear()
        self.show_function.setChecked(True)
        self.show_derivative.setChecked(True)
        self.show_integral.setChecked(True)

    def generate_random_input(self):
        random_function = random.choice(["x**2", "x**3", "sin(x)", "cos(x)", "x**2 + 3*x + 5"])
        random_x_min = random.randint(-20, -1)
        random_x_max = random.randint(1, 20)

        self.function_input.setText(random_function)
        self.x_min_input.setText(str(random_x_min))
        self.x_max_input.setText(str(random_x_max))

    def save_graph(self):
        try:
            self.figure.savefig("graph.png")
            QMessageBox.information(self, "Success", "Graph saved as 'graph.png'")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save graph: {e}")