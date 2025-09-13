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
                return """FIXED POINT METHOD EXPLANATION

The Fixed Point Method is a numerical technique for finding roots of equations of the form x = g(x).

THEORY:
--------
Given an equation f(x) = 0, we can rewrite it as x = g(x) where g(x) = x - f(x)/f'(x) or other transformations.

The method works by:
1. Starting with an initial guess x₀
2. Computing x₁ = g(x₀), x₂ = g(x₁), etc.
3. The sequence {x_n} converges to the fixed point if |g'(x)| < 1

CONVERGENCE CONDITIONS:
-----------------------
• |g'(x)| < 1 in a neighborhood of the root
• The function g(x) must be continuous
• The initial guess should be close to the root

ADVANTAGES:
-----------
• Simple to implement
• Often converges quickly when conditions are met
• Can be used to transform difficult equations

DISADVANTAGES:
--------------
• May not converge if |g'(x)| ≥ 1
• Convergence depends heavily on the choice of g(x)
• Can be slow or diverge with poor initial guesses

Enter a function to see specific details for your equation."""
            else:
                # Handle None values and provide defaults
                function = function or "g(x)"
                x0 = x0 or "1.0"
                tolerance = tolerance or "1e-8"
                max_iter = max_iter or "50"
                
                return f"""FIXED POINT METHOD EXPLANATION

CURRENT EQUATION: x = g(x) where g(x) = {function}

THEORY:
--------
The Fixed Point Method finds roots by iterating x_{{n+1}} = g(x_n) starting from x₀ = {x0}.

For your equation: x = {function}
• We seek a value x* such that x* = {function.replace('x', 'x*')}
• This means f(x*) = x* - {function.replace('x', 'x*')} = 0

ALGORITHM STEPS:
---------------
1. Start with initial guess: x₀ = {x0}
2. For n = 0, 1, 2, ... until convergence:
   • Compute x_{{n+1}} = g(x_n) = {function.replace('x', 'x_n')}
   • Check if |x_{{n+1}} - x_n| < tolerance = {tolerance}
   • If converged, x_{{n+1}} is the root
   • Otherwise, set x_n = x_{{n+1}} and continue

CONVERGENCE ANALYSIS:
--------------------
The method converges if |g'(x)| < 1 near the root.

For g(x) = {function}:
• g'(x) = {self.get_derivative_text(function)}
• Check if |g'(x)| < 1 in the neighborhood of your initial guess

CONVERGENCE RATE:
-----------------
• Linear convergence: |x_{{n+1}} - x*| ≤ L|x_n - x*| where L = |g'(x*)|
• If L < 1, the method converges
• Smaller L means faster convergence

STOPPING CRITERIA:
------------------
• Absolute error: |x_{{n+1}} - x_n| < {tolerance}
• Relative error: |x_{{n+1}} - x_n|/|x_{{n+1}}| < {tolerance}
• Maximum iterations: {max_iter}

INTERPRETATION OF RESULTS:
-------------------------
• If the method converges: You've found a root of f(x) = x - g(x) = 0
• If it diverges: Try a different transformation or initial guess
• Check the convergence condition |g'(x)| < 1

TIPS FOR SUCCESS:
----------------
• Choose g(x) such that |g'(x)| < 1 near the root
• Start with a good initial guess close to the expected root
• Consider different transformations if convergence is slow
• Monitor the error reduction in the iterations table"""
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
