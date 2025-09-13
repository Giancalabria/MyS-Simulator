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
        
        return f"""TRAPEZOIDAL RULE EXPLANATION

CURRENT INTEGRAL: ∫[{a} to {b}] {function} dx

THEORY:
--------
The Trapezoidal Rule approximates the definite integral by dividing the area under the curve
into trapezoids and summing their areas. It's based on linear interpolation between points.

FORMULA:
--------
∫[a to b] f(x) dx ≈ h/2 [f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ)]

Where:
• h = (b-a)/n (step size)
• xᵢ = a + ih
• n = number of intervals

ALGORITHM STEPS:
---------------
1. Divide interval [{a}, {b}] into {n} equal subintervals
2. Calculate step size: h = (b-a)/{n}
3. Evaluate function at all points: x₀, x₁, ..., xₙ
4. Apply trapezoidal formula
5. Sum all trapezoid areas

GEOMETRIC INTERPRETATION:
-------------------------
• Each subinterval forms a trapezoid
• Area of trapezoid = h × (f(xᵢ) + f(xᵢ₊₁))/2
• Total area ≈ sum of all trapezoid areas

ERROR ANALYSIS:
--------------
• Error = -(b-a)h²f''(ξ)/12 for some ξ in [a,b]
• Error decreases as h² (quadratic convergence)
• More intervals = smaller error
• Works best for smooth functions

ADVANTAGES:
-----------
• Simple to implement
• Always converges for continuous functions
• Easy to understand geometrically
• Good for smooth functions
• Can be easily extended to composite rules

DISADVANTAGES:
--------------
• Less accurate than higher-order methods
• Requires many intervals for high accuracy
• Error can be large for oscillatory functions
• Not suitable for functions with discontinuities

CONVERGENCE RATE:
-----------------
• O(h²) - quadratic convergence
• Doubling intervals reduces error by factor of 4
• Good balance between accuracy and simplicity

APPLICATIONS:
-------------
• Basic numerical integration
• When function is smooth and continuous
• Quick approximations
• Educational purposes
• Foundation for more advanced methods

TIPS FOR SUCCESS:
----------------
• Use more intervals for better accuracy
• Works well for smooth, continuous functions
• Consider Simpson's rule for better accuracy
• Monitor error estimates
• Use adaptive methods for varying function behavior"""
    
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
