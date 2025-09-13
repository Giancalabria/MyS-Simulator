"""
Runge-Kutta Method implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from .base_method import BaseMethod
from utils.math_utils import safe_eval_function_2d

class RungeKuttaMethod(BaseMethod):
    """Runge-Kutta Method for solving ordinary differential equations"""
    
    def __init__(self):
        super().__init__(
            name="Runge-Kutta Method",
            description="Solves ODEs using weighted average of slopes"
        )
    
    def calculate(self, function: str, x0: float, y0: float, h: float = 0.1, n_steps: int = 10, method: str = "rk4") -> Dict[str, Any]:
        """Calculate Runge-Kutta method"""
        try:
            # Get the function (dy/dx = f(x, y))
            f_func = safe_eval_function_2d(function)
            
            # Initialize
            x_values = [x0]
            y_values = [y0]
            iterations = []
            
            x = x0
            y = y0
            
            for i in range(n_steps):
                if method == "rk1":
                    # Euler's method (RK1)
                    k1 = f_func(x, y)
                    y_next = y + h * k1
                    
                    iterations.append({
                        'iteration': i + 1,
                        'x': x,
                        'y': y,
                        'k1': k1,
                        'k2': None,
                        'k3': None,
                        'k4': None,
                        'y_next': y_next
                    })
                
                elif method == "rk2":
                    # RK2 (Midpoint method)
                    k1 = f_func(x, y)
                    k2 = f_func(x + h/2, y + h/2 * k1)
                    y_next = y + h * k2
                    
                    iterations.append({
                        'iteration': i + 1,
                        'x': x,
                        'y': y,
                        'k1': k1,
                        'k2': k2,
                        'k3': None,
                        'k4': None,
                        'y_next': y_next
                    })
                
                elif method == "rk4":
                    # RK4 (Classical Runge-Kutta)
                    k1 = f_func(x, y)
                    k2 = f_func(x + h/2, y + h/2 * k1)
                    k3 = f_func(x + h/2, y + h/2 * k2)
                    k4 = f_func(x + h, y + h * k3)
                    y_next = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                    
                    iterations.append({
                        'iteration': i + 1,
                        'x': x,
                        'y': y,
                        'k1': k1,
                        'k2': k2,
                        'k3': k3,
                        'k4': k4,
                        'y_next': y_next
                    })
                
                # Update for next step
                x = x + h
                y = y_next
                x_values.append(x)
                y_values.append(y)
            
            result = {
                'x_values': x_values,
                'y_values': y_values,
                'final_x': x,
                'final_y': y,
                'h': h,
                'n_steps': n_steps,
                'method': method,
                'converged': True,
                'iterations': iterations
            }
            
            self.last_result = result
            self.last_iterations = iterations
            
            return result
            
        except Exception as e:
            raise ValueError(f"Calculation failed: {str(e)}")
    
    def get_explanation(self, function: str = "", x0: str = "0", y0: str = "1", h: str = "0.1", n_steps: str = "10", method: str = "rk4") -> str:
        """Get method explanation text"""
        # Handle None values and provide defaults
        function = function or "f(x,y)"
        x0 = x0 or "0"
        y0 = y0 or "1"
        h = h or "0.1"
        n_steps = n_steps or "10"
        method = method or "rk4"
        
        return f"""RUNGE-KUTTA METHOD EXPLANATION

CURRENT ODE: dy/dx = f(x,y) where f(x,y) = {function}

THEORY:
--------
Runge-Kutta methods are numerical techniques for solving ordinary differential equations.
They use weighted averages of function values at different points to approximate the solution.

INITIAL CONDITIONS:
------------------
• x₀ = {x0} (initial x value)
• y₀ = {y0} (initial y value)
• Step size: h = {h}
• Number of steps: {n_steps}

COMMON RUNGE-KUTTA METHODS:
---------------------------
• RK1 (Euler): y_{{n+1}} = y_n + h·f(x_n, y_n)
• RK2 (Midpoint): Uses slope at midpoint
• RK4 (Classical): Uses 4 slopes with optimal weights

RK4 FORMULA:
------------
k₁ = f(x_n, y_n)
k₂ = f(x_n + h/2, y_n + h·k₁/2)
k₃ = f(x_n + h/2, y_n + h·k₂/2)
k₄ = f(x_n + h, y_n + h·k₃)
y_{{n+1}} = y_n + h/6·(k₁ + 2k₂ + 2k₃ + k₄)

ALGORITHM STEPS:
---------------
1. Start with initial conditions: x₀ = {x0}, y₀ = {y0}
2. For n = 0, 1, 2, ..., {n_steps}-1:
   • Calculate slopes k₁, k₂, k₃, k₄
   • Compute weighted average
   • Update: x_{{n+1}} = x_n + h, y_{{n+1}} = y_n + weighted_sum
3. Return solution points (x_n, y_n)

GEOMETRIC INTERPRETATION:
-------------------------
• RK1: Uses slope at current point
• RK2: Uses slope at midpoint
• RK4: Uses 4 slopes with optimal weights
• More slopes = better approximation

ERROR ANALYSIS:
--------------
• RK1: O(h) - first order
• RK2: O(h²) - second order  
• RK4: O(h⁴) - fourth order
• Smaller h = better accuracy

ADVANTAGES:
-----------
• High accuracy (especially RK4)
• Self-starting (no previous values needed)
• Easy to implement
• Good for most ODEs
• Widely used in practice

DISADVANTAGES:
--------------
• Fixed step size (unless adaptive)
• Can be unstable for stiff ODEs
• Requires function evaluation
• May need small steps for accuracy

STABILITY:
----------
• Conditionally stable
• Stability depends on step size h
• May require very small h for stiff problems
• Consider adaptive methods for varying behavior

APPLICATIONS:
-------------
• Physics simulations
• Engineering problems
• Population dynamics
• Chemical kinetics
• Mechanical systems

TIPS FOR SUCCESS:
----------------
• Choose appropriate step size h
• Use RK4 for most problems
• Consider adaptive methods for efficiency
• Monitor solution behavior
• Check stability for stiff problems"""
    
    def plot_function_and_iterations(self, ax, function: str, iterations: List[Dict], result: Dict) -> None:
        """Plot the Runge-Kutta solution"""
        if not iterations:
            ax.text(0.5, 0.5, 'No data to display', 
                    transform=ax.transAxes, ha='center', va='center')
            return
        
        # Get parameters
        x_values = result['x_values']
        y_values = result['y_values']
        method = result['method']
        
        # Plot the solution
        ax.plot(x_values, y_values, 'b-o', label=f'RK{method[-1]} Solution', linewidth=2, markersize=4)
        
        # Mark initial point
        ax.scatter(x_values[0], y_values[0], c='green', s=100, marker='*', zorder=6, label=f'Initial: ({x_values[0]}, {y_values[0]})')
        
        # Mark final point
        ax.scatter(x_values[-1], y_values[-1], c='red', s=100, marker='*', zorder=6, label=f'Final: ({x_values[-1]:.3f}, {y_values[-1]:.3f})')
        
        # Add slope field (if possible)
        try:
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)
            
            # Create grid for slope field
            x_grid = np.linspace(x_min, x_max, 20)
            y_grid = np.linspace(y_min, y_max, 20)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            # Calculate slopes
            f_func = safe_eval_function_2d(function)
            slopes = f_func(X, Y)
            
            # Normalize slopes for visualization
            U = np.ones_like(slopes)
            V = slopes
            norm = np.sqrt(U**2 + V**2)
            U = U / norm * 0.1
            V = V / norm * 0.1
            
            # Plot slope field
            ax.quiver(X, Y, U, V, alpha=0.3, color='gray', label='Slope Field')
        except:
            pass  # Skip slope field if function evaluation fails
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Runge-Kutta Method: dy/dx = {function}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def get_example_functions(self) -> List[Dict[str, Any]]:
        """Get example functions for Runge-Kutta method"""
        return [
            {
                'name': 'y',
                'function': 'y',
                'description': 'dy/dx = y (exponential growth)',
                'x0': 0,
                'y0': 1,
                'h': 0.1,
                'n_steps': 10
            },
            {
                'name': '-y',
                'function': '-y',
                'description': 'dy/dx = -y (exponential decay)',
                'x0': 0,
                'y0': 1,
                'h': 0.1,
                'n_steps': 10
            },
            {
                'name': 'x + y',
                'function': 'x + y',
                'description': 'dy/dx = x + y',
                'x0': 0,
                'y0': 1,
                'h': 0.1,
                'n_steps': 10
            },
            {
                'name': 'x*y',
                'function': 'x*y',
                'description': 'dy/dx = xy',
                'x0': 0,
                'y0': 1,
                'h': 0.05,
                'n_steps': 20
            }
        ]
    
    def get_input_fields(self) -> List[Dict[str, str]]:
        """Get input field definitions for the UI"""
        return [
            {'name': 'function', 'label': 'Function f(x,y): *', 'type': 'entry', 'width': 30},
            {'name': 'x0', 'label': 'Initial x (x₀): *', 'type': 'entry', 'width': 15},
            {'name': 'y0', 'label': 'Initial y (y₀): *', 'type': 'entry', 'width': 15},
            {'name': 'h', 'label': 'Step size (h): *', 'type': 'entry', 'width': 15},
            {'name': 'n_steps', 'label': 'Number of steps:', 'type': 'entry', 'width': 15},
            {'name': 'method', 'label': 'Method:', 'type': 'combobox', 'width': 15, 'values': ['rk1', 'rk2', 'rk4']}
        ]
    
    def validate_inputs(self, function: str = "", x0: str = "0", y0: str = "1", h: str = "0.1", n_steps: str = "10", method: str = "rk4") -> Tuple[bool, str]:
        """Validate input parameters"""
        if not function.strip():
            return False, "Please enter a function"
        
        try:
            float(x0)
        except ValueError:
            return False, "Initial x must be a number"
        
        try:
            float(y0)
        except ValueError:
            return False, "Initial y must be a number"
        
        try:
            float(h)
        except ValueError:
            return False, "Step size must be a number"
        
        if float(h) <= 0:
            return False, "Step size must be positive"
        
        try:
            int(n_steps)
        except ValueError:
            return False, "Number of steps must be an integer"
        
        if int(n_steps) <= 0:
            return False, "Number of steps must be positive"
        
        return True, ""
