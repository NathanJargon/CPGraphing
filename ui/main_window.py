import random
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QTextEdit, QVBoxLayout, QWidget, QPushButton, QLineEdit, QLabel, QHBoxLayout, QCheckBox, QMessageBox, QFileDialog
)
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sympy import lambdify
from utils.differentiation import compute_derivative
from utils.integration import compute_integral
from utils.input_parser import parse_functions
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
        self.figure = Figure(figsize=(12, 6))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Initialize graph in dark mode
        self.set_dark_mode()

        # Input fields and button layout
        self.input_layout = QHBoxLayout()

        # Function input
        self.function_label = QLabel("Functions (comma-separated):")
        self.function_label.setFont(QFont("Arial", 12))
        self.input_layout.addWidget(self.function_label)

        self.function_input = QLineEdit()
        self.function_input.setFont(QFont("Arial", 12))
        self.function_input.setPlaceholderText("e.g., x**2 + 3*x + 5, sin(x), log(x)")
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

        self.toggle_mode_button = QPushButton("Toggle Dark/Light Mode")
        self.toggle_mode_button.setFont(QFont("Arial", 12))
        self.toggle_mode_button.clicked.connect(self.toggle_mode)
        self.button_layout.addWidget(self.toggle_mode_button)

        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.setFont(QFont("Arial", 12))
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.button_layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.setFont(QFont("Arial", 12))
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.button_layout.addWidget(self.zoom_out_button)

        self.layout.addLayout(self.button_layout)

        self.results_box = QTextEdit()
        self.results_box.setFont(QFont("Courier", 10))
        self.results_box.setReadOnly(True)
        self.layout.addWidget(self.results_box)

    def set_dark_mode(self):
        """Set the graph to dark mode."""
        self.ax.set_facecolor("#2b2b2b")
        self.figure.patch.set_facecolor("#2b2b2b")
        self.ax.tick_params(colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color("white")

    def set_light_mode(self):
        """Set the graph to light mode."""
        self.ax.set_facecolor("white")
        self.figure.patch.set_facecolor("white")
        self.ax.tick_params(colors="black")
        self.ax.xaxis.label.set_color("black")
        self.ax.yaxis.label.set_color("black")
        self.ax.title.set_color("black")

    def toggle_mode(self):
        """Toggle between dark and light modes."""
        if self.ax.get_facecolor() == (0.16862745098039217, 0.16862745098039217, 0.16862745098039217, 1.0):  # Dark mode
            self.set_light_mode()
        else:
            self.set_dark_mode()
        self.canvas.draw()

    def zoom_in(self):
        """Zoom in on the graph."""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_xlim([xlim[0] + (xlim[1] - xlim[0]) * 0.1, xlim[1] - (xlim[1] - xlim[0]) * 0.1])
        self.ax.set_ylim([ylim[0] + (ylim[1] - ylim[0]) * 0.1, ylim[1] - (ylim[1] - ylim[0]) * 0.1])
        self.canvas.draw()

    def zoom_out(self):
        """Zoom out on the graph."""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_xlim([xlim[0] - (xlim[1] - xlim[0]) * 0.1, xlim[1] + (xlim[1] - xlim[0]) * 0.1])
        self.ax.set_ylim([ylim[0] - (ylim[1] - ylim[0]) * 0.1, ylim[1] + (ylim[1] - ylim[0]) * 0.1])
        self.canvas.draw()


    def plot_graphs(self):
        function_str = self.function_input.text()
        x_min = self.x_min_input.text()
        x_max = self.x_max_input.text()

        try:
            x_min = float(x_min) if x_min else -10
            x_max = float(x_max) if x_max else 10
            if x_min >= x_max:
                raise ValueError("x_min must be less than x_max")

            functions = parse_functions(function_str)
            x_vals = np.linspace(x_min, x_max, 500)

            self.ax.clear()
            self.results_box.clear()
            self.results_box.append("x\t" + "\t".join([f"f{i+1}(x)" for i in range(len(functions))]))
            self.results_box.append("-" * 50)

            colors = ["blue", "red", "green", "orange", "purple"]
            for i, func in enumerate(functions):
                func_lambdified = lambdify('x', func, 'numpy')
                y_vals = func_lambdified(x_vals)
                derivative_vals = compute_derivative(func_lambdified, x_vals)
                integral_vals = compute_integral(func_lambdified, x_min, x_vals)

                if self.show_function.isChecked():
                    self.ax.plot(x_vals, y_vals, label=f"Function {i+1}", color=colors[i % len(colors)], linewidth=2)
                if self.show_derivative.isChecked():
                    self.ax.plot(x_vals, derivative_vals, label=f"Derivative {i+1}", linestyle="--", color=colors[i % len(colors)])
                if self.show_integral.isChecked():
                    self.ax.plot(x_vals, integral_vals, label=f"Integral {i+1}", linestyle=":", color=colors[i % len(colors)])

                for j in range(0, len(x_vals), 50):  # Show every 50th value for readability
                    self.results_box.append(f"{x_vals[j]:.5f}\t{y_vals[j]:.5f}\t{derivative_vals[j]:.5f}\t{integral_vals[j]:.5f}")

            self.ax.set_title("Functions, Derivatives, and Integrals", fontsize=16, fontweight="bold", color="white")
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
        self.results_box.clear()

    def generate_random_input(self):
        random_functions = random.sample(["x**2", "x**3", "sin(x)", "cos(x)", "x**2 + 3*x + 5", "exp(x)", "log(x)", "abs(x)", "sinh(x)", "cosh(x)"], 3)
        random_x_min = random.randint(-20, -1)
        random_x_max = random.randint(1, 20)

        self.function_input.setText(", ".join(random_functions))
        self.x_min_input.setText(str(random_x_min))
        self.x_max_input.setText(str(random_x_max))

    def save_graph(self):
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Graph", "", "PNG Files (*.png);;All Files (*)", options=options)
            if file_path:
                self.figure.savefig(file_path)
                QMessageBox.information(self, "Success", f"Graph saved as '{file_path}'")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save graph: {e}")