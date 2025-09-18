"""
Newton-Raphson Method implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from .base_method import BaseMethod
from utils.math_utils import safe_eval_function

class NewtonRaphsonMethod(BaseMethod):
    """Newton-Raphson Method for finding roots"""
    
    def __init__(self):
        super().__init__(
            name="Newton-Raphson Method",
            description="Finds roots using function and derivative"
        )
    
    def calculate(self, function: str, x0: float, tolerance: float = 1e-8, max_iter: int = 50) -> Dict[str, Any]:
        """Calculate Newton-Raphson method"""
        try:
            # Get the function
            f_func = safe_eval_function(function)
            
            # Calculate derivative numerically (using same method as working version)
            def f_prime(x, h=1e-6):
                return (f_func(x + h) - f_func(x - h)) / (2 * h)
            
            # Initialize
            iterations = []
            x = x0
            prev_x = None
            oscillation_count = 0
            
            for i in range(max_iter):
                # Calculate function value and derivative
                fx = f_func(x)
                dfx = f_prime(x)
                
                # Check for division by zero (using same threshold as working version)
                if abs(dfx) < 1e-14:
                    raise ValueError("Derivative is zero or very close to zero")
                
                # Newton-Raphson formula: x_{n+1} = x_n - f(x_n)/f'(x_n)
                x_next = x - fx / dfx
                
                # Calculate errors
                abs_error = abs(x_next - x)
                rel_error = abs_error / abs(x_next) if x_next != 0 else float('inf')
                
                # Store iteration data
                iterations.append({
                    'iteration': i + 1,
                    'x_n': x,
                    'f_x_n': fx,
                    'f_prime_x_n': dfx,
                    'x_next': x_next,
                    'abs_error': abs_error,
                    'rel_error': rel_error
                })
                
                # Check convergence - either small change OR function value close to zero
                fx_next = f_func(x_next)
                if abs_error < tolerance or abs(fx_next) < tolerance:
                    # Add final converged point to history (like working version)
                    dfx_next = f_prime(x_next)
                    iterations.append({
                        'iteration': i + 2,
                        'x_n': x_next,
                        'f_x_n': fx_next,
                        'f_prime_x_n': dfx_next,
                        'x_next': x_next,
                        'abs_error': 0.0,
                        'rel_error': 0.0
                    })
                    break
                
                # Check for divergence or oscillation
                if prev_x is not None:
                    if abs(x_next) > 1e10:  # Diverging to infinity
                        break
                    if abs(x_next - prev_x) < 1e-15:  # Oscillating between same values
                        oscillation_count += 1
                        if oscillation_count > 3:
                            break
                    else:
                        oscillation_count = 0
                
                prev_x = x
                
                # Update for next iteration
                x = x_next
            
            # Determine if converged
            converged = abs_error < tolerance if iterations else False
            root = x_next if iterations else x0
            
            result = {
                'root': root,
                'converged': converged,
                'iterations': iterations,
                'final_error': abs_error if iterations else None
            }
            
            self.last_result = result
            self.last_iterations = iterations
            
            return result
            
        except Exception as e:
            raise ValueError(f"Calculation failed: {str(e)}")
    
    def get_explanation(self, function: str = "", x0: str = "1.0", tolerance: str = "1e-8", max_iter: str = "50") -> str:
        """Get method explanation text"""
        # Handle None values and provide defaults
        function = function or "f(x)"
        x0 = x0 or "1.0"
        tolerance = tolerance or "1e-8"
        max_iter = max_iter or "50"
        
        return f"""EXPLICACIÓN DEL MÉTODO DE NEWTON-RAPHSON

ECUACIÓN ACTUAL: f(x) = 0 donde f(x) = {function}

TEORÍA:
--------
El método de Newton-Raphson encuentra raíces de ecuaciones usando la aproximación de la línea tangente.
Es uno de los algoritmos de búsqueda de raíces más poderosos y ampliamente utilizados.

FÓRMULA:
--------
x_{{n+1}} = x_n - f(x_n)/f'(x_n)

Donde:
• x_n es la aproximación actual
• f(x_n) es el valor de la función en x_n
• f'(x_n) es la derivada en x_n
• x_{{n+1}} es la siguiente aproximación

PASOS DEL ALGORITMO:
--------------------
1. Comenzar con estimación inicial: x₀ = {x0}
2. Para n = 0, 1, 2, ... hasta convergencia:
   • Calcular f(x_n) y f'(x_n)
   • Aplicar la fórmula de Newton-Raphson
   • Verificar si |x_{{n+1}} - x_n| < tolerancia = {tolerance}
   • Si convergió, x_{{n+1}} es la raíz
   • De lo contrario, establecer x_n = x_{{n+1}} y continuar

INTERPRETACIÓN GEOMÉTRICA:
--------------------------
• Dibujar línea tangente en el punto actual (x_n, f(x_n))
• Encontrar donde la línea tangente cruza el eje x
• Esta intersección es la siguiente aproximación
• Repetir hasta convergencia

CONDICIONES DE CONVERGENCIA:
----------------------------
• f'(x) ≠ 0 cerca de la raíz
• f''(x) es continua cerca de la raíz
• La estimación inicial está suficientemente cerca de la raíz
• La función está bien comportada

VENTAJAS:
---------
• Convergencia cuadrática (muy rápida)
• Simple de implementar
• Funciona para la mayoría de funciones
• Auto-corrector
• Puede encontrar raíces complejas

DESVENTAJAS:
------------
• Requiere cálculo de derivadas
• Puede no converger si f'(x) ≈ 0
• Sensible a la estimación inicial
• Puede divergir para puntos de partida pobres
• Puede ciclar u oscilar

TASA DE CONVERGENCIA:
---------------------
• Convergencia cuadrática: |x_{{n+1}} - x*| ≤ C|x_n - x*|²
• Muy rápida cuando converge
• El error se cuadra aproximadamente en cada iteración
• Mucho más rápida que los métodos lineales

ANÁLISIS DE ERROR:
------------------
• Error absoluto: |x_{{n+1}} - x_n|
• Error relativo: |x_{{n+1}} - x_n|/|x_{{n+1}}|
• Criterios de parada: error < tolerancia

TRAMPAS COMUNES:
----------------
• División por cero cuando f'(x) = 0
• Estimación inicial pobre lleva a divergencia
• Múltiples raíces pueden causar confusión
• Comportamiento oscilatorio cerca de puntos críticos

APLICACIONES:
-------------
• Encontrar raíces de polinomios
• Resolver ecuaciones no lineales
• Problemas de optimización
• Análisis numérico
• Cálculos de ingeniería

CONSEJOS PARA EL ÉXITO:
-----------------------
• Elegir estimación inicial cerca de la raíz esperada
• Verificar que f'(x) ≠ 0 cerca de la raíz
• Monitorear el comportamiento de convergencia
• Usar métodos de acotamiento como respaldo
• Considerar métodos de Newton modificados para casos difíciles"""
    
    def plot_function_and_iterations(self, ax, function: str, iterations: List[Dict], result: Dict) -> None:
        """Plot the function and Newton-Raphson iterations"""
        if not iterations:
            ax.text(0.5, 0.5, 'No data to display', 
                    transform=ax.transAxes, ha='center', va='center')
            return
        
        # Get parameters
        x_values = [iter_data['x_n'] for iter_data in iterations]
        f_values = [iter_data['f_x_n'] for iter_data in iterations]
        
        # Create function plot
        x_min = min(x_values) - 1
        x_max = max(x_values) + 1
        x_range = np.linspace(x_min, x_max, 1000)
        f_func = safe_eval_function(function)
        y_range = f_func(x_range)
        
        # Plot the original function
        ax.plot(x_range, y_range, 'b-', label=f'f(x) = {function}', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Plot Newton-Raphson iterations
        for i, iter_data in enumerate(iterations):
            x_n = iter_data['x_n']
            f_x_n = iter_data['f_x_n']
            f_prime_x_n = iter_data['f_prime_x_n']
            x_next = iter_data['x_next']
            
            # Plot current point
            ax.scatter(x_n, f_x_n, c='red', s=50, zorder=5)
            
            # Draw tangent line
            if i < len(iterations) - 1:  # Don't draw tangent for last iteration
                x_tangent = np.linspace(x_n - 0.5, x_n + 0.5, 100)
                y_tangent = f_x_n + f_prime_x_n * (x_tangent - x_n)
                ax.plot(x_tangent, y_tangent, 'r--', alpha=0.7, linewidth=1)
                
                # Draw vertical line to x-axis
                ax.plot([x_next, x_next], [0, f_x_n + f_prime_x_n * (x_next - x_n)], 'g--', alpha=0.7)
        
        # Mark the final root
        if iterations:
            final_root = iterations[-1]['x_next']
            ax.scatter(final_root, 0, c='green', s=100, marker='*', zorder=6, label=f'Root: {final_root:.6f}')
        
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Newton-Raphson Method: Root Finding')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def get_example_functions(self) -> List[Dict[str, Any]]:
        """Get example functions for Newton-Raphson method"""
        return [
            {
                'name': 'x² - 2',
                'function': 'x**2 - 2',
                'description': 'Find √2',
                'x0': 1.5
            },
            {
                'name': 'x³ - 1',
                'function': 'x**3 - 1',
                'description': 'Find cube root of 1',
                'x0': 2.0
            },
            {
                'name': 'cos(x) - x',
                'function': 'cos(x) - x',
                'description': 'Find where cos(x) = x',
                'x0': 1.0
            },
            {
                'name': 'exp(x) - 2',
                'function': 'exp(x) - 2',
                'description': 'Find ln(2)',
                'x0': 1.0
            }
        ]
    
    def get_input_fields(self) -> List[Dict[str, str]]:
        """Get input field definitions for the UI"""
        return [
            {'name': 'function', 'label': 'Function f(x): *', 'type': 'entry', 'width': 30},
            {'name': 'x0', 'label': 'Initial guess (x₀): *', 'type': 'entry', 'width': 15},
            {'name': 'tolerance', 'label': 'Tolerance:', 'type': 'entry', 'width': 15},
            {'name': 'max_iter', 'label': 'Max iterations:', 'type': 'entry', 'width': 15}
        ]
    
    def validate_inputs(self, function: str = "", x0: str = "1.0", tolerance: str = "1e-8", max_iter: str = "50") -> Tuple[bool, str]:
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
        
        if int(max_iter) <= 0:
            return False, "Max iterations must be positive"
        
        return True, ""
