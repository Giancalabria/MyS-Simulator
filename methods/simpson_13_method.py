"""
Simpson's 1/3 Rule Integration Method implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from .base_method import BaseMethod
from utils.math_utils import safe_eval_function

class Simpson13Method(BaseMethod):
    """Simpson's 1/3 Rule for numerical integration"""
    
    def __init__(self):
        super().__init__(
            name="Simpson's 1/3 Rule",
            description="Approximates integrals using parabolic segments"
        )
    
    def calculate(self, function: str, a: float, b: float, n: int = 10) -> Dict[str, Any]:
        """Calculate Simpson's 1/3 Rule integration"""
        try:
            # Get the function
            f_func = safe_eval_function(function)
            
            # Ensure n is even for Simpson's 1/3 rule
            if n % 2 != 0:
                n += 1
            
            # Calculate step size
            h = (b - a) / n
            
            # Generate points
            x_points = np.linspace(a, b, n + 1)
            f_points = f_func(x_points)
            
            # Apply Simpson's 1/3 rule
            # I = h/3 [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 4f(xₙ₋₁) + f(xₙ)]
            integral = h/3 * (f_points[0] + 4*np.sum(f_points[1::2]) + 2*np.sum(f_points[2:-1:2]) + f_points[-1])
            
            # Calculate error estimate (using fourth derivative)
            try:
                # Numerical fourth derivative
                x_test = np.linspace(a, b, 1000)
                f_test = f_func(x_test)
                f_fourth_deriv = np.gradient(np.gradient(np.gradient(np.gradient(f_test, x_test), x_test), x_test), x_test)
                max_f_fourth = np.max(np.abs(f_fourth_deriv))
                error_estimate = ((b - a) * h**4 / 180) * max_f_fourth
            except:
                error_estimate = None
            
            # Store iteration data for plotting
            iterations = []
            for i in range(n + 1):
                if i == 0 or i == n:
                    weight = 1.0
                elif i % 2 == 1:
                    weight = 4.0
                else:
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
    
    def get_explanation(self, function: str = "", a: str = "0", b: str = "1", n: str = "10") -> str:
        """Get method explanation text"""
        # Handle None values and provide defaults
        function = function or "f(x)"
        a = a or "0"
        b = b or "1"
        n = n or "10"
        
        return f"""EXPLICACIÓN DE LA REGLA DE SIMPSON 1/3

INTEGRAL ACTUAL: ∫[{a} a {b}] {function} dx

TEORÍA:
--------
La Regla de Simpson 1/3 aproxima la integral definida ajustando parábolas a grupos de tres
puntos consecutivos e integrando estas parábolas. Es más precisa que la Regla del Trapecio.

FÓRMULA:
--------
∫[a a b] f(x) dx ≈ h/3 [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 4f(xₙ₋₁) + f(xₙ)]

Donde:
• h = (b-a)/n (tamaño de paso)
• xᵢ = a + ih
• n debe ser par
• Pesos: 1, 4, 2, 4, 2, ..., 4, 1

PASOS DEL ALGORITMO:
--------------------
1. Asegurar que n sea par (agregar 1 si es impar)
2. Dividir intervalo [{a}, {b}] en {n} subintervalos iguales
3. Calcular tamaño de paso: h = (b-a)/{n}
4. Evaluar función en todos los puntos: x₀, x₁, ..., xₙ
5. Aplicar fórmula de Simpson 1/3 con pesos alternantes
6. Sumar todas las áreas de segmentos parabólicos

INTERPRETACIÓN GEOMÉTRICA:
--------------------------
• Cada par de subintervalos forma un segmento parabólico
• Ajusta parábola a través de tres puntos consecutivos
• Integra la parábola exactamente
• Más preciso que la aproximación lineal

ANÁLISIS DE ERROR:
------------------
• Error = -(b-a)h⁴f⁽⁴⁾(ξ)/180 para algún ξ en [a,b]
• El error disminuye como h⁴ (convergencia de cuarto orden)
• Mucho más preciso que la Regla del Trapecio
• Funciona mejor para funciones suaves

VENTAJAS:
---------
• Alta precisión (cuarto orden)
• Eficiente para funciones suaves
• Ampliamente usado en la práctica
• Buen balance entre precisión y simplicidad
• Excelente para funciones polinómicas

DESVENTAJAS:
------------
• Requiere número par de intervalos
• Menos preciso para funciones oscilatorias
• Puede ser inestable para derivadas de alto orden
• No es adecuado para funciones con discontinuidades

TASA DE CONVERGENCIA:
---------------------
• O(h⁴) - convergencia de cuarto orden
• Duplicar intervalos reduce el error por factor de 16
• Mucho más rápido que la Regla del Trapecio

COMPARACIÓN CON OTROS MÉTODOS:
------------------------------
• Más preciso que la Regla del Trapecio
• Menos preciso que la Regla de Simpson 3/8
• Buen compromiso entre precisión y simplicidad
• Método de Simpson más comúnmente usado

APLICACIONES:
-------------
• Integración numérica general
• Funciones suaves y continuas
• Cálculos de ingeniería
• Computación científica
• Cuando se necesita alta precisión

CONSEJOS PARA EL ÉXITO:
-----------------------
• Siempre usar número par de intervalos
• Funciona mejor para funciones suaves
• Considerar Simpson 3/8 para aún mejor precisión
• Monitorear estimaciones de error
• Usar métodos adaptativos para comportamiento variable de función"""
    
    def plot_function_and_iterations(self, ax, function: str, iterations: List[Dict], result: Dict) -> None:
        """Plot the function and Simpson's 1/3 approximation"""
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
        
        # Plot Simpson's 1/3 approximation (parabolic segments)
        for i in range(0, len(x_points) - 2, 2):
            x1, x2, x3 = x_points[i], x_points[i + 1], x_points[i + 2]
            f1, f2, f3 = f_points[i], f_points[i + 1], f_points[i + 2]
            
            # Create parabolic segment
            x_seg = np.linspace(x1, x3, 50)
            # Lagrange interpolation for parabola through 3 points
            y_seg = ((x_seg - x2) * (x_seg - x3) / ((x1 - x2) * (x1 - x3))) * f1 + \
                   ((x_seg - x1) * (x_seg - x3) / ((x2 - x1) * (x2 - x3))) * f2 + \
                   ((x_seg - x1) * (x_seg - x2) / ((x3 - x1) * (x3 - x2))) * f3
            
            # Draw parabolic segment
            ax.plot(x_seg, y_seg, 'r-', linewidth=2, alpha=0.7)
            ax.fill_between(x_seg, 0, y_seg, alpha=0.3, color='red')
        
        # Mark evaluation points
        ax.scatter(x_points, f_points, c='red', s=50, zorder=5, label='Evaluation Points')
        
        # Fill area under curve for reference
        ax.fill_between(x_range, y_range, alpha=0.2, color='blue', label='Exact Area')
        
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title("Simpson's 1/3 Rule Integration")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def get_example_functions(self) -> List[Dict[str, Any]]:
        """Get example functions for Simpson's 1/3 Rule"""
        return [
            {
                'name': 'x⁴',
                'function': 'x**4',
                'description': 'x⁴ from 0 to 2',
                'a': 0,
                'b': 2,
                'n': 10
            },
            {
                'name': 'sin(x)',
                'function': 'sin(x)',
                'description': 'sin(x) from 0 to π',
                'a': 0,
                'b': 'pi',
                'n': 20
            },
            {
                'name': 'cos(x)',
                'function': 'cos(x)',
                'description': 'cos(x) from 0 to π/2',
                'a': 0,
                'b': 'pi/2',
                'n': 16
            },
            {
                'name': '1/(1+x²)',
                'function': '1/(1+x**2)',
                'description': '1/(1+x²) from 0 to 1',
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
            {'name': 'n', 'label': 'Number of intervals (even): *', 'type': 'entry', 'width': 15}
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
            n_int = int(n)
        except ValueError:
            return False, "Number of intervals must be an integer"
        
        if n_int <= 0:
            return False, "Number of intervals must be positive"
        
        if n_int < 2:
            return False, "Number of intervals must be at least 2"
        
        return True, ""
