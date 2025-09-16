"""
Bisection Method implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from .base_method import BaseMethod
from utils.math_utils import safe_eval_function


class BisectionMethod(BaseMethod):
    """Bisection Method for finding roots"""

    def __init__(self):
        super().__init__(
            name="Bisection Method",
            description="Finds roots by repeatedly halving the interval"
        )

    def calculate(self, function: str, a: float, b: float, tolerance: float = 1e-8, max_iter: int = 50) -> Dict[str, Any]:
        """Calculate Bisection method"""
        try:
            # Get the function
            f_func = safe_eval_function(function)

            fa = f_func(a)
            fb = f_func(b)

            if np.isnan(fa) or np.isnan(fb) or not np.isfinite(fa) or not np.isfinite(fb):
                raise ValueError("Function is not finite at the interval endpoints")

            if fa * fb > 0:
                raise ValueError("f(a) and f(b) must have opposite signs (root bracketing)")

            iterations: List[Dict[str, Any]] = []
            a_n, b_n = a, b
            c_n = (a_n + b_n) / 2.0
            fc = f_func(c_n)

            abs_error = float('inf')

            for i in range(1, max_iter + 1):
                c_n = (a_n + b_n) / 2.0
                fc = f_func(c_n)

                # Absolute error as interval half-length
                abs_error = (b_n - a_n) / 2.0
                rel_error = abs_error / (abs(c_n) if c_n != 0 else 1.0)

                iterations.append({
                    'iteration': i,
                    'a_n': a_n,
                    'b_n': b_n,
                    'c_n': c_n,
                    'f_c_n': fc,
                    'abs_error': abs_error,
                    'rel_error': rel_error
                })

                # Check if exact root or tolerance achieved
                if abs_error < tolerance or fc == 0:
                    break

                # Decide the subinterval that contains the root
                if fa * fc < 0:
                    b_n = c_n
                    fb = fc
                else:
                    a_n = c_n
                    fa = fc

            converged = abs_error < tolerance or fc == 0

            result = {
                'root': c_n,
                'converged': converged,
                'iterations': iterations,
                'final_error': abs_error,
                'a': a,
                'b': b
            }

            self.last_result = result
            self.last_iterations = iterations

            return result

        except Exception as e:
            raise ValueError(f"Calculation failed: {str(e)}")

    def get_explanation(self, function: str = "", a: str = "0", b: str = "1", tolerance: str = "1e-8", max_iter: str = "50") -> str:
        """Get method explanation text"""
        # Handle None values and provide defaults
        function = function or "f(x)"
        a = a or "0"
        b = b or "1"
        tolerance = tolerance or "1e-8"
        max_iter = max_iter or "50"

        return f"""EXPLICACIÓN DEL MÉTODO DE BISECCIÓN

ECUACIÓN ACTUAL: f(x) = 0 donde f(x) = {function}

TEORÍA:
--------
El Método de Bisección encuentra una raíz de f(x) en un intervalo [a,b] donde f(a) y f(b)
tienen signos opuestos, garantizando por el Teorema del Valor Intermedio que existe al menos
una raíz en (a,b). Repetidamente divide el intervalo a la mitad y selecciona el subintervalo que
contiene la raíz.

REQUISITOS:
-----------
• f(x) continua en [{a}, {b}]
• f({a}) y f({b}) con signos opuestos (f(a)·f(b) < 0)

PASOS DEL ALGORITMO:
--------------------
1. Comenzar con el intervalo inicial [a, b] = [{a}, {b}]
2. Calcular el punto medio c = (a+b)/2
3. Evaluar f(c)
4. Si f(c) = 0 o el tamaño del intervalo es menor que la tolerancia {tolerance}, detener
5. Si f(a)·f(c) < 0, la raíz está en [a, c]; si no, está en [c, b]
6. Repetir hasta un máximo de {max_iter} iteraciones

PROPIEDADES:
------------
• Siempre converge si se cumplen los requisitos
• Convergencia lineal
• Error después de n iteraciones ≤ (b-a)/2^n

CRITERIOS DE PARADA:
--------------------
• Error absoluto del intervalo: (b-a)/2 < {tolerance}
• f(c) = 0 (raíz exacta)
• Máximo de iteraciones alcanzado: {max_iter}

VENTAJAS:
---------
• Robusto y simple
• Garantiza convergencia si f(a)·f(b) < 0 y f es continua

DESVENTAJAS:
------------
• Convergencia lenta comparada con Newton-Raphson
• Requiere un intervalo con cambio de signo
"""

    def plot_function_and_iterations(self, ax, function: str, iterations: List[Dict], result: Dict) -> None:
        """Plot the function and bisection intervals"""
        if not iterations:
            ax.text(0.5, 0.5, 'No data to display', 
                    transform=ax.transAxes, ha='center', va='center')
            return

        # Determine plotting range from initial interval with some padding
        a = result['a']
        b = result['b']
        width = b - a
        x_min = a - 0.1 * width
        x_max = b + 0.1 * width
        x_range = np.linspace(x_min, x_max, 1000)
        f_func = safe_eval_function(function)
        y_range = f_func(x_range)

        # Plot function and x-axis
        ax.plot(x_range, y_range, 'b-', label=f'f(x) = {function}', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

        # Plot shrinking intervals
        for i, it in enumerate(iterations):
            a_i, b_i = it['a_n'], it['b_n']
            c_i = it['c_n']
            # Interval line on x-axis
            ax.plot([a_i, b_i], [0, 0], 'r-', linewidth=max(1, 3 - 0.02 * i), alpha=0.6)
            # Mark endpoints and midpoint
            ax.scatter([a_i, b_i], [0, 0], c='red', s=30, zorder=5)
            ax.scatter(c_i, 0, c='green', s=40, zorder=6)

        # Final root marker
        final_c = iterations[-1]['c_n']
        ax.scatter(final_c, 0, c='green', s=100, marker='*', zorder=7, label=f'Root: {final_c:.6f}')

        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Bisection Method: Interval Halving')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def get_example_functions(self) -> List[Dict[str, Any]]:
        """Get example functions for Bisection method"""
        return [
            {
                'name': 'x² - 2',
                'function': 'x**2 - 2',
                'description': 'Root near √2 in [1, 2]',
                'a': 1,
                'b': 2,
                'tolerance': 1e-8,
                'max_iter': 50
            },
            {
                'name': 'cos(x) - x',
                'function': 'cos(x) - x',
                'description': 'Root near 0.739 in [0, 1]',
                'a': 0,
                'b': 1,
                'tolerance': 1e-8,
                'max_iter': 50
            },
            {
                'name': 'x³ - x - 1',
                'function': 'x**3 - x - 1',
                'description': 'Root in [1, 2]',
                'a': 1,
                'b': 2,
                'tolerance': 1e-8,
                'max_iter': 50
            }
        ]

    def get_input_fields(self) -> List[Dict[str, str]]:
        """Get input field definitions for the UI"""
        return [
            {'name': 'function', 'label': 'Function f(x): *', 'type': 'entry', 'width': 30},
            {'name': 'a', 'label': 'Lower bound (a): *', 'type': 'entry', 'width': 15},
            {'name': 'b', 'label': 'Upper bound (b): *', 'type': 'entry', 'width': 15},
            {'name': 'tolerance', 'label': 'Tolerance:', 'type': 'entry', 'width': 15},
            {'name': 'max_iter', 'label': 'Max iterations:', 'type': 'entry', 'width': 15}
        ]

    def validate_inputs(self, function: str = "", a: str = "0", b: str = "1", tolerance: str = "1e-8", max_iter: str = "50") -> Tuple[bool, str]:
        """Validate input parameters"""
        if not function.strip():
            return False, "Please enter a function"

        try:
            a_val = float(a)
        except ValueError:
            return False, "Lower bound must be a number"

        try:
            b_val = float(b)
        except ValueError:
            return False, "Upper bound must be a number"

        if a_val >= b_val:
            return False, "Lower bound must be less than upper bound"

        try:
            float(tolerance)
        except ValueError:
            return False, "Tolerance must be a number"

        try:
            mi = int(max_iter)
            if mi <= 0:
                return False, "Max iterations must be positive"
        except ValueError:
            return False, "Max iterations must be an integer"

        # Optional: sign check can only be done after parsing function; 
        # we'll do a quick safe check here but do not fail hard if evaluation fails.
        try:
            f_func = safe_eval_function(function)
            fa = f_func(a_val)
            fb = f_func(b_val)
            if not (np.isfinite(fa) and np.isfinite(fb)):
                return False, "Function must be finite at the bounds"
            if fa * fb > 0:
                return False, "f(a) and f(b) must have opposite signs"
        except Exception:
            # If function cannot be evaluated yet, let calculation surface errors
            pass

        return True, ""



