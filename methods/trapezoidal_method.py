"""
Trapezoidal Rule Integration Method implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from .base_method import BaseMethod
from utils.math_utils import safe_eval_function

class TrapezoidalMethod(BaseMethod):
    """Trapezoidal Rule for numerical integration"""
    
    def __init__(self):
        super().__init__(
            name="Trapezoidal Rule",
            description="Approximates integrals using trapezoids"
        )
    
    def calculate(self, function: str, a: float, b: float, n: int = 10) -> Dict[str, Any]:
        """Calculate Trapezoidal Rule integration"""
        try:
            # Get the function
            f_func = safe_eval_function(function)
            
            # Calculate step size
            h = (b - a) / n
            
            # Generate points
            x_points = np.linspace(a, b, n + 1)
            f_points = f_func(x_points)
            
            # Apply trapezoidal rule
            integral = h * (0.5 * f_points[0] + np.sum(f_points[1:-1]) + 0.5 * f_points[-1])
            
            # Calculate error estimate (using second derivative)
            try:
                # Numerical second derivative
                x_test = np.linspace(a, b, 1000)
                f_test = f_func(x_test)
                f_second_deriv = np.gradient(np.gradient(f_test, x_test), x_test)
                max_f_second = np.max(np.abs(f_second_deriv))
                error_estimate = ((b - a) * h**2 / 12) * max_f_second
            except:
                error_estimate = None
            
            # Store iteration data for plotting
            iterations = []
            for i in range(n + 1):
                iterations.append({
                    'iteration': i,
                    'x': x_points[i],
                    'f_x': f_points[i],
                    'weight': 0.5 if i == 0 or i == n else 1.0
                })
            
            result = {
                'integral': integral,
                'error_estimate': error_estimate,
                'n_intervals': n,
                'h': h,
                'converged': True,
                'iterations': iterations,
                'x_points': x_points,
                'f_points': f_points,
                'a': a,
                'b': b
            }
            
            self.last_result = result
            self.last_iterations = iterations
            
            return result
            
        except Exception as e:
            raise ValueError(f"Calculation failed: {str(e)}")
    
    def get_explanation(self, function: str = "", a: str = "0", b: str = "1", n: str = "10") -> str:
        """Get method explanation text"""
        # Handle None values and provide defaults
        function = function or "f(x)"
        a = a or "0"
        b = b or "1"
        n = n or "10"
        
        return f"""EXPLICACIÓN DE LA REGLA DEL TRAPECIO

INTEGRAL ACTUAL: ∫[{a} a {b}] {function} dx

TEORÍA:
--------
La Regla del Trapecio aproxima la integral definida dividiendo el área bajo la curva
en trapecios y sumando sus áreas. Se basa en interpolación lineal entre puntos.

FÓRMULA:
--------
∫[a a b] f(x) dx ≈ h/2 [f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ)]

Donde:
• h = (b-a)/n (tamaño de paso)
• xᵢ = a + ih
• n = número de intervalos

PASOS DEL ALGORITMO:
--------------------
1. Dividir intervalo [{a}, {b}] en {n} subintervalos iguales
2. Calcular tamaño de paso: h = (b-a)/{n}
3. Evaluar función en todos los puntos: x₀, x₁, ..., xₙ
4. Aplicar fórmula del trapecio
5. Sumar todas las áreas de trapecios

INTERPRETACIÓN GEOMÉTRICA:
--------------------------
• Cada subintervalo forma un trapecio
• Área del trapecio = h × (f(xᵢ) + f(xᵢ₊₁))/2
• Área total ≈ suma de todas las áreas de trapecios

ANÁLISIS DE ERROR:
------------------
• Error = -(b-a)h²f''(ξ)/12 para algún ξ en [a,b]
• El error disminuye como h² (convergencia cuadrática)
• Más intervalos = error más pequeño
• Funciona mejor para funciones suaves

VENTAJAS:
---------
• Simple de implementar
• Siempre converge para funciones continuas
• Fácil de entender geométricamente
• Bueno para funciones suaves
• Se puede extender fácilmente a reglas compuestas

DESVENTAJAS:
------------
• Menos preciso que métodos de orden superior
• Requiere muchos intervalos para alta precisión
• El error puede ser grande para funciones oscilatorias
• No es adecuado para funciones con discontinuidades

TASA DE CONVERGENCIA:
---------------------
• O(h²) - convergencia cuadrática
• Duplicar intervalos reduce el error por factor de 4
• Buen balance entre precisión y simplicidad

APLICACIONES:
-------------
• Integración numérica básica
• Cuando la función es suave y continua
• Aproximaciones rápidas
• Propósitos educativos
• Base para métodos más avanzados

CONSEJOS PARA EL ÉXITO:
-----------------------
• Usar más intervalos para mejor precisión
• Funciona bien para funciones suaves y continuas
• Considerar la regla de Simpson para mejor precisión
• Monitorear estimaciones de error
• Usar métodos adaptativos para comportamiento variable de función"""
    
    def plot_function_and_iterations(self, ax, function: str, iterations: List[Dict], result: Dict) -> None:
        """Plot the function and trapezoidal approximation"""
        if not iterations:
            ax.text(0.5, 0.5, 'No data to display', 
                    transform=ax.transAxes, ha='center', va='center')
            return
        
        # Get parameters
        a = result['a']
        b = result['b']
        x_points = result['x_points']
        f_points = result['f_points']
        
        # Create function plot
        x_range = np.linspace(a, b, 1000)
        f_func = safe_eval_function(function)
        y_range = f_func(x_range)
        
        # Plot the original function
        ax.plot(x_range, y_range, 'b-', label=f'f(x) = {function}', linewidth=2)
        
        # Plot trapezoidal approximation
        for i in range(len(x_points) - 1):
            x1, x2 = x_points[i], x_points[i + 1]
            f1, f2 = f_points[i], f_points[i + 1]
            
            # Draw trapezoid
            ax.plot([x1, x2], [f1, f2], 'r-', linewidth=2, alpha=0.7)
            ax.fill_between([x1, x2], [0, 0], [f1, f2], alpha=0.3, color='red')
        
        # Mark evaluation points
        ax.scatter(x_points, f_points, c='red', s=50, zorder=5, label='Evaluation Points')
        
        # Fill area under curve for reference
        ax.fill_between(x_range, y_range, alpha=0.2, color='blue', label='Exact Area')
        
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Trapezoidal Rule Integration')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def get_example_functions(self) -> List[Dict[str, Any]]:
        """Get example functions for Trapezoidal Rule"""
        return [
            {
                'name': 'x²',
                'function': 'x**2',
                'description': 'x² from 0 to 2',
                'a': 0,
                'b': 2,
                'n': 10
            },
            {
                'name': 'sin(x)',
                'function': 'sin(x)',
                'description': 'sin(x) from 0 to π',
                'a': 0,
                'b': '3.14159',
                'n': 20
            },
            {
                'name': 'exp(x)',
                'function': 'exp(x)',
                'description': 'eˣ from 0 to 1',
                'a': 0,
                'b': 1,
                'n': 15
            },
            {
                'name': '1/x',
                'function': '1/x',
                'description': '1/x from 1 to 2',
                'a': 1,
                'b': 2,
                'n': 20
            }
        ]
    
    def get_input_fields(self) -> List[Dict[str, str]]:
        """Get input field definitions for the UI"""
        return [
            {'name': 'function', 'label': 'Function f(x): *', 'type': 'entry', 'width': 30},
            {'name': 'a', 'label': 'Lower limit (a): *', 'type': 'entry', 'width': 15},
            {'name': 'b', 'label': 'Upper limit (b): *', 'type': 'entry', 'width': 15},
            {'name': 'n', 'label': 'Number of intervals: *', 'type': 'entry', 'width': 15}
        ]
    
    def validate_inputs(self, function: str = "", a: str = "0", b: str = "1", n: str = "10") -> Tuple[bool, str]:
        """Validate input parameters"""
        if not function.strip():
            return False, "Please enter a function"
        
        try:
            float(a)
        except ValueError:
            return False, "Lower limit must be a number"
        
        try:
            float(b)
        except ValueError:
            return False, "Upper limit must be a number"
        
        if float(a) >= float(b):
            return False, "Lower limit must be less than upper limit"
        
        try:
            int(n)
        except ValueError:
            return False, "Number of intervals must be an integer"
        
        if int(n) <= 0:
            return False, "Number of intervals must be positive"
        
        return True, ""
