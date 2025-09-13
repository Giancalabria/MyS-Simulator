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
            
            # Calculate derivative numerically
            def f_prime(x, h=1e-6):
                return (f_func(x + h) - f_func(x - h)) / (2 * h)
            
            # Initialize
            iterations = []
            x_n = x0
            
            for i in range(max_iter):
                # Calculate function value and derivative
                f_x = f_func(x_n)
                f_prime_x = f_prime(x_n)
                
                # Check for division by zero
                if abs(f_prime_x) < 1e-12:
                    raise ValueError("Derivative is zero or very close to zero")
                
                # Newton-Raphson formula: x_{n+1} = x_n - f(x_n)/f'(x_n)
                x_next = x_n - f_x / f_prime_x
                
                # Calculate errors
                abs_error = abs(x_next - x_n)
                rel_error = abs_error / abs(x_next) if x_next != 0 else float('inf')
                
                # Store iteration data
                iterations.append({
                    'iteration': i + 1,
                    'x_n': x_n,
                    'f_x_n': f_x,
                    'f_prime_x_n': f_prime_x,
                    'x_next': x_next,
                    'abs_error': abs_error,
                    'rel_error': rel_error
                })
                
                # Check convergence
                if abs_error < tolerance:
                    break
                
                # Update for next iteration
                x_n = x_next
            
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
        
        return f"""NEWTON-RAPHSON METHOD EXPLANATION

CURRENT EQUATION: f(x) = 0 where f(x) = {function}

THEORY:
--------
The Newton-Raphson method finds roots of equations by using the tangent line approximation.
It's one of the most powerful and widely used root-finding algorithms.

FORMULA:
--------
x_{{n+1}} = x_n - f(x_n)/f'(x_n)

Where:
• x_n is the current approximation
• f(x_n) is the function value at x_n
• f'(x_n) is the derivative at x_n
• x_{{n+1}} is the next approximation

ALGORITHM STEPS:
---------------
1. Start with initial guess: x₀ = {x0}
2. For n = 0, 1, 2, ... until convergence:
   • Calculate f(x_n) and f'(x_n)
   • Apply Newton-Raphson formula
   • Check if |x_{{n+1}} - x_n| < tolerance = {tolerance}
   • If converged, x_{{n+1}} is the root
   • Otherwise, set x_n = x_{{n+1}} and continue

GEOMETRIC INTERPRETATION:
-------------------------
• Draw tangent line at current point (x_n, f(x_n))
• Find where tangent line crosses x-axis
• This intersection is the next approximation
• Repeat until convergence

CONVERGENCE CONDITIONS:
-----------------------
• f'(x) ≠ 0 near the root
• f''(x) is continuous near the root
• Initial guess is close enough to the root
• Function is well-behaved

ADVANTAGES:
-----------
• Quadratic convergence (very fast)
• Simple to implement
• Works for most functions
• Self-correcting
• Can find complex roots

DISADVANTAGES:
--------------
• Requires derivative calculation
• May not converge if f'(x) ≈ 0
• Sensitive to initial guess
• Can diverge for poor starting points
• May cycle or oscillate

CONVERGENCE RATE:
-----------------
• Quadratic convergence: |x_{{n+1}} - x*| ≤ C|x_n - x*|²
• Very fast when it converges
• Error roughly squares each iteration
• Much faster than linear methods

ERROR ANALYSIS:
--------------
• Absolute error: |x_{{n+1}} - x_n|
• Relative error: |x_{{n+1}} - x_n|/|x_{{n+1}}|
• Stopping criteria: error < tolerance

COMMON PITFALLS:
----------------
• Division by zero when f'(x) = 0
• Poor initial guess leads to divergence
• Multiple roots can cause confusion
• Oscillatory behavior near critical points

APPLICATIONS:
-------------
• Finding roots of polynomials
• Solving nonlinear equations
• Optimization problems
• Numerical analysis
• Engineering calculations

TIPS FOR SUCCESS:
----------------
• Choose initial guess close to expected root
• Check that f'(x) ≠ 0 near the root
• Monitor convergence behavior
• Use bracketing methods as backup
• Consider modified Newton methods for difficult cases"""
    
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
