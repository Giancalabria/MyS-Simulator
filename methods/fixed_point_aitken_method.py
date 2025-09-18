"""
Fixed Point + Aitken Method implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from .base_method import BaseMethod
from utils.math_utils import safe_eval_function, check_convergence_condition

class FixedPointAitkenMethod(BaseMethod):
    """Fixed Point Method with Aitken's delta-squared acceleration"""
    
    def __init__(self):
        super().__init__(
            name="Fixed Point + Aitken",
            description="Fixed Point method with Aitken's acceleration"
        )
    
    def calculate(self, function: str, x0: float, max_iter: int, tolerance: float) -> Dict[str, Any]:
        """Calculate Fixed Point + Aitken method"""
        try:
            # Get the function
            g_func = safe_eval_function(function)
            
            # Initialize
            iterations = []
            x = x0
            
            for i in range(max_iter):
                # Generate three consecutive terms for Aitken acceleration
                x1 = g_func(x)      # x_{n+1}
                x2 = g_func(x1)     # x_{n+2}
                
                # Apply Aitken's delta-squared acceleration
                denom = x2 - 2*x1 + x
                if abs(denom) < 1e-14:  # Numerical stability check
                    x_acc = x2  # Fallback to regular iteration
                else:
                    x_acc = x - (x1 - x)**2 / denom
                
                # Calculate errors
                abs_err = abs(x_acc - x)
                rel_err = abs_err / abs(x_acc) if x_acc != 0 else float('inf')
                
                # Store iteration data
                iterations.append({
                    'iteration': i + 1,
                    'x_n': x,
                    'g_x_n': x1,  # This is g(x_n)
                    'aitken_x': x_acc,
                    'abs_error': abs_err,
                    'rel_error': rel_err
                })
                
                # Check convergence
                if abs_err < tolerance:
                    break
                
                # Use accelerated value for next iteration
                x = x_acc
            
            # Determine if converged
            converged = abs_err < tolerance if iterations else False
            root = x if iterations else x0
            
            # Check convergence condition
            convergence_check = check_convergence_condition(g_func, x0)
            
            result = {
                'root': root,
                'converged': converged,
                'iterations': iterations,
                'convergence_check': convergence_check,
                'method': 'Fixed Point + Aitken'
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
                return """EXPLICACIÓN DEL MÉTODO DE PUNTO FIJO + AITKEN

El Método de Punto Fijo con Aceleración de Aitken combina la iteración de punto fijo con la 
aceleración de Aitken para mejorar la convergencia.

TEORÍA:
--------
Dada una ecuación f(x) = 0, reescribimos como x = g(x). El método de punto fijo genera una 
secuencia {x_n} donde x_{n+1} = g(x_n). La aceleración de Aitken mejora esta convergencia.

FÓRMULA DE AITKEN:
------------------
x* ≈ x_n - (x_{n+1} - x_n)² / (x_{n+2} - 2x_{n+1} + x_n)

Donde:
• x_n, x_{n+1}, x_{n+2} son tres iteraciones consecutivas
• x* es la estimación acelerada de la raíz

VENTAJAS:
---------
• Acelera la convergencia de métodos lineales
• Simple de implementar
• Mejora significativamente la precisión
• Funciona bien con secuencias que convergen linealmente

DESVENTAJAS:
------------
• Requiere al menos 3 iteraciones para funcionar
• Puede ser inestable si la secuencia no converge
• No garantiza convergencia si el método base diverge

Ingresa una función para ver detalles específicos de tu ecuación."""
            else:
                # Handle None values and provide defaults
                function = function or "g(x)"
                x0 = x0 or "1.0"
                tolerance = tolerance or "1e-8"
                max_iter = max_iter or "50"
                
                return f"""EXPLICACIÓN DEL MÉTODO DE PUNTO FIJO + AITKEN

ECUACIÓN ACTUAL: x = g(x) donde g(x) = {function}

TEORÍA:
--------
El Método de Punto Fijo + Aitken combina la iteración básica x_{{n+1}} = g(x_n) con la 
aceleración de Aitken para mejorar la convergencia.

PASOS DEL ALGORITMO:
--------------------
1. Comenzar con estimación inicial: x₀ = {x0}
2. Para n = 0, 1, 2, ... hasta convergencia:
   • Calcular x_{{n+1}} = g(x_n) = {function.replace('x', 'x_n')}
   • Almacenar secuencia de 3 valores consecutivos
   • Aplicar aceleración de Aitken cuando sea posible
   • Verificar convergencia usando estimación acelerada
   • Si convergió, usar estimación de Aitken como raíz

FÓRMULA DE ACELERACIÓN DE AITKEN:
---------------------------------
x* ≈ x_n - (x_{{n+1}} - x_n)² / (x_{{n+2}} - 2x_{{n+1}} + x_n)

Donde:
• x_n, x_{{n+1}}, x_{{n+2}} son tres iteraciones consecutivas
• x* es la estimación acelerada de la raíz

CONDICIONES DE CONVERGENCIA:
----------------------------
• |g'(x)| < 1 en una vecindad de la raíz (misma que punto fijo)
• La función g(x) debe ser continua
• La estimación inicial debe estar cerca de la raíz

VENTAJAS DE LA ACELERACIÓN:
---------------------------
• Mejora significativamente la tasa de convergencia
• Reduce el número de iteraciones necesarias
• Aumenta la precisión de la estimación final
• Funciona especialmente bien con convergencia lineal

TASA DE CONVERGENCIA:
---------------------
• Convergencia superlineal cuando funciona
• Puede convertir convergencia lineal en cuadrática
• Mucho más rápida que el método de punto fijo puro

CRITERIOS DE PARADA:
--------------------
• Error absoluto: |x_{{n+1}} - x_n| < {tolerance}
• Error de Aitken: |x* - x_n| < {tolerance}
• Máximo de iteraciones: {max_iter}

INTERPRETACIÓN DE RESULTADOS:
------------------------------
• Si el método converge: Has encontrado una raíz de f(x) = x - g(x) = 0
• La estimación de Aitken suele ser más precisa que la iteración básica
• Si diverge: Prueba una transformación diferente o estimación inicial

CONSEJOS PARA EL ÉXITO:
-----------------------
• Elegir g(x) tal que |g'(x)| < 1 cerca de la raíz
• Comenzar con una buena estimación inicial cerca de la raíz esperada
• La aceleración de Aitken funciona mejor con convergencia lineal
• Monitorear tanto la iteración básica como la estimación acelerada"""
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def plot_function_and_iterations(self, ax, function: str, iterations: List[Dict], result: Dict) -> None:
        """Plot the function and iteration points with Aitken estimates"""
        if not iterations:
            ax.text(0.5, 0.5, 'No data to display', 
                    transform=ax.transAxes, ha='center', va='center')
            return
        
        # Extract data
        x_values = [iter_data['x_n'] for iter_data in iterations]
        g_values = [iter_data['g_x_n'] for iter_data in iterations]
        aitken_values = [iter_data.get('aitken_x') for iter_data in iterations]
        
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
        for i in range(len(x_values) - 1):
            ax.plot(x_values[i], g_values[i], 'ro', markersize=6, alpha=0.7)
        
        # Plot Aitken estimates when available
        aitken_x_vals = [x for x in aitken_values if x is not None]
        aitken_g_vals = []
        if aitken_x_vals:
            try:
                g_func = safe_eval_function(function)
                aitken_g_vals = [g_func(x) for x in aitken_x_vals]
                ax.scatter(aitken_x_vals, aitken_g_vals, c='green', s=80, marker='s', 
                          alpha=0.8, label='Aitken Estimates', zorder=5)
            except:
                pass
        
        # Mark the final converged point with a different color
        if len(x_values) > 0:
            final_x = x_values[-1]
            final_g = g_values[-1]
            ax.plot(final_x, final_g, 'go', markersize=8, 
                    label=f'Final: ({final_x:.6f}, {final_g:.6f})', alpha=0.9)
        
        ax.set_xlabel('x')
        ax.set_ylabel('g(x)')
        ax.set_title('Fixed Point + Aitken Method: Function and Iterations')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def get_example_functions(self) -> List[Dict[str, Any]]:
        """Get example functions for Fixed Point + Aitken method"""
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
