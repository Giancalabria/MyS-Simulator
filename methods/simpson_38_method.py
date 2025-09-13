"""
Simpson's 3/8 Rule Integration Method implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from .base_method import BaseMethod
from utils.math_utils import safe_eval_function

class Simpson38Method(BaseMethod):
    """Simpson's 3/8 Rule for numerical integration"""
    
    def __init__(self):
        super().__init__(
            name="Simpson's 3/8 Rule",
            description="Approximates integrals using cubic segments"
        )
    
    def calculate(self, function: str, a: float, b: float, n: int = 9) -> Dict[str, Any]:
        """Calculate Simpson's 3/8 Rule integration"""
        try:
            # Get the function
            f_func = safe_eval_function(function)
            
            # Ensure n is divisible by 3 for Simpson's 3/8 rule
            if n % 3 != 0:
                n = ((n // 3) + 1) * 3
            
            # Calculate step size
            h = (b - a) / n
            
            # Generate points
            x_points = np.linspace(a, b, n + 1)
            f_points = f_func(x_points)
            
            # Apply Simpson's 3/8 rule
            # I = 3h/8 [f(x₀) + 3f(x₁) + 3f(x₂) + 2f(x₃) + 3f(x₄) + 3f(x₅) + 2f(x₆) + ... + f(xₙ)]
            integral = 3*h/8 * (f_points[0] + 3*np.sum(f_points[1:-1:3]) + 3*np.sum(f_points[2:-1:3]) + 2*np.sum(f_points[3:-1:3]) + f_points[-1])
            
            # Calculate error estimate (using fourth derivative)
            try:
                # Numerical fourth derivative
                x_test = np.linspace(a, b, 1000)
                f_test = f_func(x_test)
                f_fourth_deriv = np.gradient(np.gradient(np.gradient(np.gradient(f_test, x_test), x_test), x_test), x_test)
                max_f_fourth = np.max(np.abs(f_fourth_deriv))
                error_estimate = ((b - a) * h**4 / 80) * max_f_fourth
            except:
                error_estimate = None
            
            # Store iteration data for plotting
            iterations = []
            for i in range(n + 1):
                if i == 0 or i == n:
                    weight = 1.0
                elif i % 3 == 1 or i % 3 == 2:
                    weight = 3.0
                else:  # i % 3 == 0
                    weight = 2.0
                
                iterations.append({
                    'iteration': i,
                    'x': x_points[i],
                    'f_x': f_points[i],
                    'weight': weight
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
    
    def get_explanation(self, function: str = "", a: str = "0", b: str = "1", n: str = "9") -> str:
        """Get method explanation text"""
        # Handle None values and provide defaults
        function = function or "f(x)"
        a = a or "0"
        b = b or "1"
        n = n or "9"
        
        return f"""EXPLICACIÓN DE LA REGLA DE SIMPSON 3/8

INTEGRAL ACTUAL: ∫[{a} a {b}] {function} dx

TEORÍA:
--------
La Regla de Simpson 3/8 aproxima la integral definida ajustando polinomios cúbicos a grupos de
cuatro puntos consecutivos e integrando estos cúbicos. Es más precisa que la Regla de Simpson 1/3.

FÓRMULA:
--------
∫[a a b] f(x) dx ≈ 3h/8 [f(x₀) + 3f(x₁) + 3f(x₂) + 2f(x₃) + 3f(x₄) + 3f(x₅) + 2f(x₆) + ... + f(xₙ)]

Donde:
• h = (b-a)/n (tamaño de paso)
• xᵢ = a + ih
• n debe ser divisible por 3
• Pesos: 1, 3, 3, 2, 3, 3, 2, ..., 3, 3, 2, 1

PASOS DEL ALGORITMO:
--------------------
1. Asegurar que n sea divisible por 3
2. Dividir intervalo [{a}, {b}] en {n} subintervalos iguales
3. Calcular tamaño de paso: h = (b-a)/{n}
4. Evaluar función en todos los puntos: x₀, x₁, ..., xₙ
5. Aplicar fórmula de Simpson 3/8 con pesos cúbicos
6. Sumar todas las áreas de segmentos cúbicos

INTERPRETACIÓN GEOMÉTRICA:
--------------------------
• Cada grupo de tres subintervalos forma un segmento cúbico
• Ajusta polinomio cúbico a través de cuatro puntos consecutivos
• Integra el cúbico exactamente
• Más preciso que la aproximación parabólica

ANÁLISIS DE ERROR:
------------------
• Error = -(b-a)h⁴f⁽⁴⁾(ξ)/80 para algún ξ en [a,b]
• El error disminuye como h⁴ (convergencia de cuarto orden)
• Más preciso que la Regla de Simpson 1/3
• Funciona mejor para funciones suaves

VENTAJAS:
---------
• Mayor precisión que Simpson 1/3
• Convergencia de cuarto orden
• Bueno para funciones suaves
• Puede manejar comportamiento más complejo
• Excelente para polinomios cúbicos

DESVENTAJAS:
------------
• Requiere n divisible por 3
• Más complejo que Simpson 1/3
• Puede ser menos estable para derivadas de alto orden
• No es adecuado para funciones con discontinuidades

TASA DE CONVERGENCIA:
---------------------
• O(h⁴) - convergencia de cuarto orden
• Duplicar intervalos reduce el error por factor de 16
• Igual que Simpson 1/3 pero con mejor constante

COMPARACIÓN CON OTROS MÉTODOS:
------------------------------
• Más preciso que la Regla de Simpson 1/3
• Menos preciso que la Regla de Boole
• Buen compromiso entre precisión y complejidad
• Mejor para funciones con comportamiento cúbico

APLICACIONES:
-------------
• Integración numérica de alta precisión
• Funciones suaves y continuas
• Cálculos de precisión de ingeniería
• Computación científica
• Cuando se necesita máxima precisión

CONSEJOS PARA EL ÉXITO:
-----------------------
• Siempre usar n divisible por 3
• Funciona mejor para funciones suaves
• Considerar la Regla de Boole para aún mayor precisión
• Monitorear estimaciones de error
• Usar métodos adaptativos para comportamiento variable"""
    
    def plot_function_and_iterations(self, ax, function: str, iterations: List[Dict], result: Dict) -> None:
        """Plot the function and Simpson's 3/8 approximation"""
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
        
        # Plot Simpson's 3/8 approximation (cubic segments)
        for i in range(0, len(x_points) - 3, 3):
            x1, x2, x3, x4 = x_points[i], x_points[i + 1], x_points[i + 2], x_points[i + 3]
            f1, f2, f3, f4 = f_points[i], f_points[i + 1], f_points[i + 2], f_points[i + 3]
            
            # Create cubic segment
            x_seg = np.linspace(x1, x4, 50)
            # Lagrange interpolation for cubic through 4 points
            y_seg = ((x_seg - x2) * (x_seg - x3) * (x_seg - x4) / ((x1 - x2) * (x1 - x3) * (x1 - x4))) * f1 + \
                   ((x_seg - x1) * (x_seg - x3) * (x_seg - x4) / ((x2 - x1) * (x2 - x3) * (x2 - x4))) * f2 + \
                   ((x_seg - x1) * (x_seg - x2) * (x_seg - x4) / ((x3 - x1) * (x3 - x2) * (x3 - x4))) * f3 + \
                   ((x_seg - x1) * (x_seg - x2) * (x_seg - x3) / ((x4 - x1) * (x4 - x2) * (x4 - x3))) * f4
            
            # Draw cubic segment
            ax.plot(x_seg, y_seg, 'r-', linewidth=2, alpha=0.7)
            ax.fill_between(x_seg, 0, y_seg, alpha=0.3, color='red')
        
        # Mark evaluation points
        ax.scatter(x_points, f_points, c='red', s=50, zorder=5, label='Evaluation Points')
        
        # Fill area under curve for reference
        ax.fill_between(x_range, y_range, alpha=0.2, color='blue', label='Exact Area')
        
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title("Simpson's 3/8 Rule Integration")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def get_example_functions(self) -> List[Dict[str, Any]]:
        """Get example functions for Simpson's 3/8 Rule"""
        return [
            {
                'name': 'x⁶',
                'function': 'x**6',
                'description': 'x⁶ from 0 to 2',
                'a': 0,
                'b': 2,
                'n': 9
            },
            {
                'name': 'sin(x)',
                'function': 'sin(x)',
                'description': 'sin(x) from 0 to π',
                'a': 0,
                'b': '3.14159',
                'n': 21
            },
            {
                'name': 'exp(x)',
                'function': 'exp(x)',
                'description': 'eˣ from 0 to 2',
                'a': 0,
                'b': 2,
                'n': 15
            },
            {
                'name': 'x⁵ + x³',
                'function': 'x**5 + x**3',
                'description': 'x⁵ + x³ from 0 to 1',
                'a': 0,
                'b': 1,
                'n': 12
            }
        ]
    
    def get_input_fields(self) -> List[Dict[str, str]]:
        """Get input field definitions for the UI"""
        return [
            {'name': 'function', 'label': 'Function f(x): *', 'type': 'entry', 'width': 30},
            {'name': 'a', 'label': 'Lower limit (a): *', 'type': 'entry', 'width': 15},
            {'name': 'b', 'label': 'Upper limit (b): *', 'type': 'entry', 'width': 15},
            {'name': 'n', 'label': 'Number of intervals (divisible by 3): *', 'type': 'entry', 'width': 15}
        ]
    
    def validate_inputs(self, function: str = "", a: str = "0", b: str = "1", n: str = "9") -> Tuple[bool, str]:
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
            n_int = int(n)
        except ValueError:
            return False, "Number of intervals must be an integer"
        
        if n_int <= 0:
            return False, "Number of intervals must be positive"
        
        if n_int < 3:
            return False, "Number of intervals must be at least 3"
        
        if n_int % 3 != 0:
            return False, "Number of intervals must be divisible by 3"
        
        return True, ""
