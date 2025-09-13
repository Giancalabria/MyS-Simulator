"""
Fixed Point Method implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from .base_method import BaseMethod
from utils.math_utils import safe_eval_function, check_convergence_condition

class FixedPointMethod(BaseMethod):
    """Fixed Point Method for finding roots"""
    
    def __init__(self):
        super().__init__(
            name="Fixed Point Method",
            description="Finds roots by iterating x_{n+1} = g(x_n)"
        )
    
    def calculate(self, function: str, x0: float, max_iter: int, tolerance: float) -> Dict[str, Any]:
        """Calculate Fixed Point method"""
        try:
            # Get the function
            g_func = safe_eval_function(function)
            
            # Initialize
            iterations = []
            x_n = x0
            
            for i in range(max_iter):
                # Calculate g(x_n)
                g_x_n = g_func(x_n)
                
                # Calculate errors
                if i > 0:
                    abs_error = abs(g_x_n - x_n)
                    rel_error = abs_error / abs(g_x_n) if g_x_n != 0 else float('inf')
                else:
                    abs_error = float('inf')
                    rel_error = float('inf')
                
                # Store iteration data
                iterations.append({
                    'iteration': i,
                    'x_n': x_n,
                    'g_x_n': g_x_n,
                    'abs_error': abs_error,
                    'rel_error': rel_error
                })
                
                # Check convergence
                if abs_error < tolerance:
                    break
                
                # Update for next iteration
                x_n = g_x_n
            
            # Determine if converged
            converged = abs_error < tolerance if iterations else False
            root = g_x_n if iterations else x0
            
            # Check convergence condition
            convergence_check = check_convergence_condition(g_func, x0)
            
            result = {
                'root': root,
                'converged': converged,
                'iterations': iterations,
                'convergence_check': convergence_check
            }
            
            self.last_result = result
            self.last_iterations = iterations
            
            return result
            
        except Exception as e:
            raise ValueError(f"Calculation failed: {str(e)}")
    
    def get_explanation(self, function: str, x0: str = "0.5", tolerance: str = "1e-8", max_iter: str = "50") -> str:
        """Get method explanation text"""
        try:
            if not function:
                return """EXPLICACIÓN DEL MÉTODO DE PUNTO FIJO

El Método de Punto Fijo es una técnica numérica para encontrar raíces de ecuaciones de la forma x = g(x).

TEORÍA:
--------
Dada una ecuación f(x) = 0, podemos reescribirla como x = g(x) donde g(x) = x - f(x)/f'(x) u otras transformaciones.

El método funciona:
1. Comenzando con una estimación inicial x₀
2. Calculando x₁ = g(x₀), x₂ = g(x₁), etc.
3. La secuencia {x_n} converge al punto fijo si |g'(x)| < 1

CONDICIONES DE CONVERGENCIA:
----------------------------
• |g'(x)| < 1 en una vecindad de la raíz
• La función g(x) debe ser continua
• La estimación inicial debe estar cerca de la raíz

VENTAJAS:
---------
• Simple de implementar
• A menudo converge rápidamente cuando se cumplen las condiciones
• Puede usarse para transformar ecuaciones difíciles

DESVENTAJAS:
------------
• Puede no converger si |g'(x)| ≥ 1
• La convergencia depende mucho de la elección de g(x)
• Puede ser lento o divergir con malas estimaciones iniciales

Ingresa una función para ver detalles específicos de tu ecuación."""
            else:
                # Handle None values and provide defaults
                function = function or "g(x)"
                x0 = x0 or "1.0"
                tolerance = tolerance or "1e-8"
                max_iter = max_iter or "50"
                
                return f"""EXPLICACIÓN DEL MÉTODO DE PUNTO FIJO

ECUACIÓN ACTUAL: x = g(x) donde g(x) = {function}

TEORÍA:
--------
El Método de Punto Fijo encuentra raíces iterando x_{{n+1}} = g(x_n) comenzando desde x₀ = {x0}.

Para tu ecuación: x = {function}
• Buscamos un valor x* tal que x* = {function.replace('x', 'x*')}
• Esto significa f(x*) = x* - {function.replace('x', 'x*')} = 0

PASOS DEL ALGORITMO:
--------------------
1. Comenzar con estimación inicial: x₀ = {x0}
2. Para n = 0, 1, 2, ... hasta convergencia:
   • Calcular x_{{n+1}} = g(x_n) = {function.replace('x', 'x_n')}
   • Verificar si |x_{{n+1}} - x_n| < tolerancia = {tolerance}
   • Si convergió, x_{{n+1}} es la raíz
   • De lo contrario, establecer x_n = x_{{n+1}} y continuar

ANÁLISIS DE CONVERGENCIA:
-------------------------
El método converge si |g'(x)| < 1 cerca de la raíz.

Para g(x) = {function}:
• g'(x) = {self.get_derivative_text(function)}
• Verificar si |g'(x)| < 1 en la vecindad de tu estimación inicial

TASA DE CONVERGENCIA:
---------------------
• Convergencia lineal: |x_{{n+1}} - x*| ≤ L|x_n - x*| donde L = |g'(x*)|
• Si L < 1, el método converge
• L más pequeño significa convergencia más rápida

CRITERIOS DE PARADA:
--------------------
• Error absoluto: |x_{{n+1}} - x_n| < {tolerance}
• Error relativo: |x_{{n+1}} - x_n|/|x_{{n+1}}| < {tolerance}
• Máximo de iteraciones: {max_iter}

INTERPRETACIÓN DE RESULTADOS:
------------------------------
• Si el método converge: Has encontrado una raíz de f(x) = x - g(x) = 0
• Si diverge: Prueba una transformación diferente o estimación inicial
• Verificar la condición de convergencia |g'(x)| < 1

CONSEJOS PARA EL ÉXITO:
-----------------------
• Elegir g(x) tal que |g'(x)| < 1 cerca de la raíz
• Comenzar con una buena estimación inicial cerca de la raíz esperada
• Considerar transformaciones diferentes si la convergencia es lenta
• Monitorear la reducción del error en la tabla de iteraciones"""
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def plot_function_and_iterations(self, ax, function: str, iterations: List[Dict], result: Dict) -> None:
        """Plot the function and iteration points"""
        if not iterations:
            ax.text(0.5, 0.5, 'No data to display', 
                    transform=ax.transAxes, ha='center', va='center')
            return
        
        # Extract data
        x_values = [iter_data['x_n'] for iter_data in iterations]
        g_values = [iter_data['g_x_n'] for iter_data in iterations]
        
        # Plot the original function g(x) as a continuous curve
        try:
            # Create a much wider range for better visualization
            if x_values:
                x_center = np.mean(x_values)
                x_range = np.linspace(x_center - 3, x_center + 3, 2000)
            else:
                x_range = np.linspace(-3, 3, 2000)
            
            # Get the function
            g_func = safe_eval_function(function)
            
            # Evaluate the function
            y_range = g_func(x_range)
            
            # Plot the original function as a smooth curve
            ax.plot(x_range, y_range, 'b-', label=f'g(x) = {function}', linewidth=2)
            
        except Exception as e:
            print(f"Function plotting error: {e}")
        
        # Plot iteration points as dots on the function curve
        # Each point shows (x_n, g(x_n)) which is (x_n, x_{n+1})
        for i in range(len(x_values) - 1):
            ax.plot(x_values[i], g_values[i], 'ro', markersize=6, alpha=0.7)
        
        # Mark the final converged point with a different color
        if len(x_values) > 0:
            final_x = x_values[-1]
            final_g = g_values[-1]
            ax.plot(final_x, final_g, 'go', markersize=8, 
                    label=f'Converged: ({final_x:.6f}, {final_g:.6f})', alpha=0.9)
        
        ax.set_xlabel('x')
        ax.set_ylabel('g(x)')
        ax.set_title('Fixed Point Method: Function g(x) and Iteration Points')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def get_example_functions(self) -> List[Dict[str, Any]]:
        """Get example functions for Fixed Point method"""
        return [
            {
                'name': 'cos(x)',
                'function': 'cos(x)',
                'description': 'cos(x) - converges to ~0.739',
                'x0': 0.5
            },
            {
                'name': 'sin(x)',
                'function': 'sin(x)',
                'description': 'sin(x) - converges to 0',
                'x0': 0.5
            },
            {
                'name': 'exp(-x)',
                'function': 'exp(-x)',
                'description': 'e^(-x) - converges to ~0.567',
                'x0': 0.5
            },
            {
                'name': 'x^2 - 2',
                'function': 'x**2 - 2',
                'description': 'x² - 2 - may not converge',
                'x0': 1.0
            }
        ]
    
    def get_input_fields(self) -> List[Dict[str, str]]:
        """Get input field definitions for the UI"""
        return [
            {'name': 'function', 'label': 'Function g(x):', 'type': 'entry', 'width': 30},
            {'name': 'x0', 'label': 'Initial guess (x₀):', 'type': 'entry', 'width': 15},
            {'name': 'tolerance', 'label': 'Tolerance:', 'type': 'entry', 'width': 15},
            {'name': 'max_iter', 'label': 'Max iterations:', 'type': 'entry', 'width': 15}
        ]
    
    def validate_inputs(self, function: str = "", x0: str = "0.5", tolerance: str = "1e-8", max_iter: str = "50") -> Tuple[bool, str]:
        """Validate input parameters"""
        if not function.strip():
            return False, "Please enter a function"
        
        try:
            float(x0)
        except ValueError:
            return False, "Initial guess must be a number"
        
        try:
            float(tolerance)
        except ValueError:
            return False, "Tolerance must be a number"
        
        try:
            int(max_iter)
        except ValueError:
            return False, "Max iterations must be an integer"
        
        return True, ""
    
    def get_derivative_text(self, function: str) -> str:
        """Get derivative text for common functions"""
        # Simple derivative approximations for display
        if function == "cos(x)":
            return "-sin(x)"
        elif function == "sin(x)":
            return "cos(x)"
        elif function == "exp(x)":
            return "exp(x)"
        elif function == "log(x)":
            return "1/x"
        elif function == "x**2":
            return "2*x"
        elif function == "x**3":
            return "3*x**2"
        elif function == "sqrt(x)":
            return "1/(2*sqrt(x))"
        else:
            return "g'(x) (derivative of g(x))"
