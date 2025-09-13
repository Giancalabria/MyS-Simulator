"""
Newton-Cotes Integration Method implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from .base_method import BaseMethod
from utils.math_utils import safe_eval_function

class NewtonCotesMethod(BaseMethod):
    """Newton-Cotes Method for numerical integration"""
    
    def __init__(self):
        super().__init__(
            name="Newton-Cotes Method",
            description="Approximates integrals using polynomial interpolation"
        )
    
    def calculate(self, function: str, a: float, b: float, n: int = 5, method: str = "closed") -> Dict[str, Any]:
        """Calculate Newton-Cotes integration"""
        try:
            # Get the function
            f_func = safe_eval_function(function)
            
            # Generate points
            if method == "closed":
                x_points = np.linspace(a, b, n + 1)
            else:  # open
                x_points = np.linspace(a, b, n + 3)[1:-1]  # Exclude endpoints
            
            f_points = f_func(x_points)
            
            # Calculate weights based on Newton-Cotes formulas
            if n == 1:
                # Trapezoidal rule
                weights = np.array([0.5, 0.5])
                if method == "open":
                    weights = np.array([1.0])
            elif n == 2:
                # Simpson's 1/3 rule
                weights = np.array([1/3, 4/3, 1/3])
                if method == "open":
                    weights = np.array([2/3, 2/3])
            elif n == 3:
                # Simpson's 3/8 rule
                weights = np.array([3/8, 9/8, 9/8, 3/8])
                if method == "open":
                    weights = np.array([4/3, -2/3, 4/3])
            elif n == 4:
                # Boole's rule
                weights = np.array([7/90, 32/90, 12/90, 32/90, 7/90])
                if method == "open":
                    weights = np.array([11/24, -14/24, 26/24, -14/24, 11/24])
            else:
                # General Newton-Cotes (approximate)
                weights = np.ones(n + 1) / (n + 1)
                if method == "open":
                    weights = np.ones(n + 1) / n
            
            # Calculate step size
            h = (b - a) / n
            
            # Apply Newton-Cotes formula
            integral = h * np.sum(weights * f_points)
            
            # Calculate error estimate
            try:
                # Numerical derivative for error estimation
                x_test = np.linspace(a, b, 1000)
                f_test = f_func(x_test)
                f_deriv = np.gradient(f_test, x_test)
                max_f_deriv = np.max(np.abs(f_deriv))
                error_estimate = h**(n+1) * max_f_deriv
            except:
                error_estimate = None
            
            # Store iteration data for plotting
            iterations = []
            for i in range(len(x_points)):
                iterations.append({
                    'iteration': i,
                    'x': x_points[i],
                    'f_x': f_points[i],
                    'weight': weights[i] if i < len(weights) else 1.0
                })
            
            result = {
                'integral': integral,
                'error_estimate': error_estimate,
                'n_points': n + 1,
                'h': h,
                'method': method,
                'converged': True,
                'iterations': iterations,
                'x_points': x_points,
                'f_points': f_points,
                'weights': weights,
                'a': a,
                'b': b
            }
            
            self.last_result = result
            self.last_iterations = iterations
            
            return result
            
        except Exception as e:
            raise ValueError(f"Calculation failed: {str(e)}")
    
    def get_explanation(self, function: str = "", a: str = "0", b: str = "1", n: str = "5", method: str = "closed") -> str:
        """Get method explanation text"""
        # Handle None values and provide defaults
        function = function or "f(x)"
        a = a or "0"
        b = b or "1"
        n = n or "5"
        method = method or "closed"
        
        return f"""NEWTON-COTES METHOD EXPLANATION

CURRENT INTEGRAL: ∫[{a} to {b}] {function} dx

THEORY:
--------
Newton-Cotes methods approximate definite integrals by replacing the integrand with
a polynomial that interpolates the function at equally spaced points, then integrating
the polynomial exactly.

FORMULA:
--------
∫[a to b] f(x) dx ≈ h ∑ᵢ wᵢ f(xᵢ)

Where:
• h = (b-a)/n (step size)
• wᵢ are the Newton-Cotes weights
• xᵢ are the interpolation points
• n is the degree of the polynomial

COMMON NEWTON-COTES FORMULAS:
-----------------------------
• n=1: Trapezoidal Rule (linear)
• n=2: Simpson's 1/3 Rule (quadratic)
• n=3: Simpson's 3/8 Rule (cubic)
• n=4: Boole's Rule (quartic)
• n=5: Higher-order formulas

CLOSED vs OPEN FORMULAS:
-----------------------
• CLOSED: Uses endpoints a and b
• OPEN: Excludes endpoints (useful for singularities)

ALGORITHM STEPS:
---------------
1. Choose degree n and method type ({method})
2. Generate {n+1} equally spaced points
3. Calculate Newton-Cotes weights
4. Evaluate function at all points
5. Apply weighted sum formula
6. Calculate error estimate

WEIGHT CALCULATION:
------------------
Weights are determined by integrating Lagrange basis polynomials:
wᵢ = ∫[a to b] Lᵢ(x) dx

Where Lᵢ(x) is the i-th Lagrange polynomial.

ERROR ANALYSIS:
--------------
• Error = O(h^(n+1)) for n-th degree polynomial
• Higher degree = better accuracy
• Error depends on (n+1)-th derivative
• Can be unstable for high degrees

ADVANTAGES:
-----------
• High accuracy for smooth functions
• Systematic approach
• Well-established theory
• Good for regular intervals
• Can achieve high precision

DISADVANTAGES:
--------------
• Can be unstable for high degrees
• Requires smooth functions
• Not suitable for singularities
• Runge's phenomenon for high degrees
• Fixed point spacing

CONVERGENCE RATE:
-----------------
• O(h^(n+1)) where n is polynomial degree
• Very fast for smooth functions
• May not converge for rough functions

APPLICATIONS:
-------------
• High-precision integration
• Smooth, continuous functions
• Regular integration intervals
• Scientific computing
• Engineering calculations

TIPS FOR SUCCESS:
----------------
• Use appropriate degree for function smoothness
• Consider open formulas for singularities
• Monitor error estimates
• Avoid very high degrees
• Use adaptive methods for varying behavior"""
    
    def plot_function_and_iterations(self, ax, function: str, iterations: List[Dict], result: Dict) -> None:
        """Plot the function and Newton-Cotes approximation"""
        if not iterations:
            ax.text(0.5, 0.5, 'No data to display', 
                    transform=ax.transAxes, ha='center', va='center')
            return
        
        # Get parameters
        a = result['a']
        b = result['b']
        x_points = result['x_points']
        f_points = result['f_points']
        weights = result['weights']
        method = result['method']
        
        # Create function plot
        x_range = np.linspace(a, b, 1000)
        f_func = safe_eval_function(function)
        y_range = f_func(x_range)
        
        # Plot the original function
        ax.plot(x_range, y_range, 'b-', label=f'f(x) = {function}', linewidth=2)
        
        # Plot Newton-Cotes approximation
        # Create polynomial interpolation
        if len(x_points) > 1:
            # Lagrange interpolation
            x_interp = np.linspace(a, b, 200)
            y_interp = np.zeros_like(x_interp)
            
            for i, (x_i, f_i) in enumerate(zip(x_points, f_points)):
                # Calculate Lagrange basis polynomial
                L_i = np.ones_like(x_interp)
                for j, x_j in enumerate(x_points):
                    if i != j:
                        L_i *= (x_interp - x_j) / (x_i - x_j)
                y_interp += f_i * L_i
            
            # Plot interpolating polynomial
            ax.plot(x_interp, y_interp, 'r-', linewidth=2, alpha=0.7, label='Interpolating Polynomial')
            ax.fill_between(x_interp, 0, y_interp, alpha=0.3, color='red')
        
        # Mark evaluation points with weights
        for i, (x, f_x, weight) in enumerate(zip(x_points, f_points, weights)):
            size = 50 + weight * 100  # Size based on weight
            ax.scatter(x, f_x, c='red', s=size, zorder=5, alpha=0.8)
            ax.annotate(f'w={weight:.2f}', (x, f_x), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        # Fill area under curve for reference
        ax.fill_between(x_range, y_range, alpha=0.2, color='blue', label='Exact Area')
        
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Newton-Cotes Integration ({method})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def get_example_functions(self) -> List[Dict[str, Any]]:
        """Get example functions for Newton-Cotes method"""
        return [
            {
                'name': 'x⁴',
                'function': 'x**4',
                'description': 'x⁴ from 0 to 2',
                'a': 0,
                'b': 2,
                'n': 5
            },
            {
                'name': 'sin(x)',
                'function': 'sin(x)',
                'description': 'sin(x) from 0 to π',
                'a': 0,
                'b': '3.14159',
                'n': 4
            },
            {
                'name': 'exp(x)',
                'function': 'exp(x)',
                'description': 'eˣ from 0 to 1',
                'a': 0,
                'b': 1,
                'n': 3
            },
            {
                'name': '1/(1+x²)',
                'function': '1/(1+x**2)',
                'description': '1/(1+x²) from 0 to 1',
                'a': 0,
                'b': 1,
                'n': 5
            }
        ]
    
    def get_input_fields(self) -> List[Dict[str, str]]:
        """Get input field definitions for the UI"""
        return [
            {'name': 'function', 'label': 'Function f(x): *', 'type': 'entry', 'width': 30},
            {'name': 'a', 'label': 'Lower limit (a): *', 'type': 'entry', 'width': 15},
            {'name': 'b', 'label': 'Upper limit (b): *', 'type': 'entry', 'width': 15},
            {'name': 'n', 'label': 'Degree (n): *', 'type': 'entry', 'width': 15},
            {'name': 'method', 'label': 'Method:', 'type': 'combobox', 'width': 15, 'values': ['closed', 'open']}
        ]
    
    def validate_inputs(self, function: str = "", a: str = "0", b: str = "1", n: str = "5", method: str = "closed") -> Tuple[bool, str]:
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
            return False, "Degree must be an integer"
        
        if n_int <= 0:
            return False, "Degree must be positive"
        
        if n_int > 10:
            return False, "Degree should not exceed 10 (unstable)"
        
        return True, ""
