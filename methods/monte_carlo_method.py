"""
Monte Carlo Integration Method implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from .base_method import BaseMethod
from utils.math_utils import safe_eval_function

class MonteCarloMethod(BaseMethod):
    """Monte Carlo Method for numerical integration"""
    
    def __init__(self):
        super().__init__(
            name="Monte Carlo Integration",
            description="Estimates integrals using random sampling"
        )
    
    def calculate(self, function: str, a: float, b: float, n_samples: int = 1000, method: str = "hit_or_miss") -> Dict[str, Any]:
        """Calculate Monte Carlo integration"""
        try:
            # Get the function
            f_func = safe_eval_function(function)
            
            # Generate random samples
            x_samples = np.random.uniform(a, b, n_samples)
            f_samples = f_func(x_samples)
            
            # Calculate integral using different methods
            if method == "hit_or_miss":
                # Find max value for hit-or-miss
                x_test = np.linspace(a, b, 1000)
                f_test = f_func(x_test)
                f_max = np.max(f_test)
                f_min = np.min(f_test)
                
                # Generate random points in rectangle
                y_samples = np.random.uniform(f_min, f_max, n_samples)
                hits = np.sum(f_samples >= y_samples)
                
                # Calculate integral
                integral = (b - a) * (f_max - f_min) * (hits / n_samples)
                
            elif method == "average_value":
                # Average value method
                integral = (b - a) * np.mean(f_samples)
            
            # Calculate error estimate
            if method == "hit_or_miss":
                error_estimate = (b - a) * (f_max - f_min) * np.sqrt(hits * (n_samples - hits)) / n_samples
            else:
                error_estimate = (b - a) * np.std(f_samples) / np.sqrt(n_samples)
            
            # Store sample data for plotting
            iterations = []
            for i in range(min(100, n_samples)):  # Store first 100 samples for display
                iterations.append({
                    'iteration': i + 1,
                    'x_sample': x_samples[i],
                    'f_sample': f_samples[i],
                    'y_sample': y_samples[i] if method == "hit_or_miss" else None,
                    'hit': f_samples[i] >= y_samples[i] if method == "hit_or_miss" else None
                })
            
            result = {
                'integral': integral,
                'error_estimate': error_estimate,
                'n_samples': n_samples,
                'method': method,
                'converged': True,  # Monte Carlo always "converges"
                'iterations': iterations,
                'x_samples': x_samples,
                'f_samples': f_samples,
                'y_samples': y_samples if method == "hit_or_miss" else None,
                'a': a,
                'b': b,
                'f_max': f_max if method == "hit_or_miss" else None,
                'f_min': f_min if method == "hit_or_miss" else None
            }
            
            self.last_result = result
            self.last_iterations = iterations
            
            return result
            
        except Exception as e:
            raise ValueError(f"Calculation failed: {str(e)}")
    
    def get_explanation(self, function: str = "", a: str = "0", b: str = "1", n_samples: str = "1000", method: str = "hit_or_miss") -> str:
        """Get method explanation text"""
        # Handle None values and provide defaults
        function = function or "f(x)"
        a = a or "0"
        b = b or "1"
        n_samples = n_samples or "1000"
        method = method or "hit_or_miss"
        
        return f"""MONTE CARLO INTEGRATION EXPLANATION

CURRENT INTEGRAL: ∫[{a} to {b}] {function} dx

THEORY:
--------
Monte Carlo integration estimates the value of definite integrals using random sampling.
Instead of using deterministic methods, it uses random numbers to approximate the integral.

METHODS:
--------
1. HIT-OR-MISS METHOD:
   • Generates random points in a rectangle containing the function
   • Counts points that fall under the curve
   • Integral ≈ (Rectangle Area) × (Hits / Total Points)

2. AVERAGE VALUE METHOD:
   • Samples random points from the integration interval
   • Calculates function values at these points
   • Integral ≈ (b-a) × Average(f(x))

ALGORITHM STEPS:
---------------
1. Define integration interval: [{a}, {b}]
2. Generate {n_samples} random samples
3. Evaluate function at sample points
4. Apply chosen method to estimate integral
5. Calculate error estimate

CONVERGENCE:
------------
• Error decreases as O(1/√N) where N is number of samples
• More samples = better accuracy
• Independent of function complexity
• Works well for high-dimensional integrals

ADVANTAGES:
-----------
• Simple to implement
• Works for any integrable function
• Naturally parallelizable
• Handles complex domains easily
• No need for derivatives

DISADVANTAGES:
--------------
• Slow convergence rate
• Requires many samples for high accuracy
• Results are probabilistic
• Can be computationally expensive

ERROR ESTIMATION:
----------------
• Hit-or-miss: σ = (b-a)(f_max-f_min)√(hits×(N-hits))/N
• Average value: σ = (b-a)σ_f/√N
• Confidence improves with √N

APPLICATIONS:
-------------
• High-dimensional integrals
• Complex integration domains
• Functions with discontinuities
• Physics simulations
• Financial mathematics

TIPS FOR SUCCESS:
----------------
• Use more samples for better accuracy
• Hit-or-miss works well for bounded functions
• Average value is more efficient for smooth functions
• Consider variance reduction techniques"""
    
    def plot_function_and_iterations(self, ax, function: str, iterations: List[Dict], result: Dict) -> None:
        """Plot the function and Monte Carlo samples"""
        if not iterations:
            ax.text(0.5, 0.5, 'No data to display', 
                    transform=ax.transAxes, ha='center', va='center')
            return
        
        # Get parameters
        a = result['a']
        b = result['b']
        method = result['method']
        
        # Create function plot
        x_range = np.linspace(a, b, 1000)
        f_func = safe_eval_function(function)
        y_range = f_func(x_range)
        
        # Plot the function
        ax.plot(x_range, y_range, 'b-', label=f'f(x) = {function}', linewidth=2)
        
        if method == "hit_or_miss":
            # Plot rectangle for hit-or-miss
            f_max = result['f_max']
            f_min = result['f_min']
            ax.axhline(y=f_max, color='r', linestyle='--', alpha=0.5, label=f'y = {f_max:.2f}')
            ax.axhline(y=f_min, color='r', linestyle='--', alpha=0.5, label=f'y = {f_min:.2f}')
            
            # Plot samples
            x_samples = result['x_samples'][:1000]  # Limit for display
            f_samples = result['f_samples'][:1000]
            y_samples = result['y_samples'][:1000]
            
            # Separate hits and misses
            hits = f_samples >= y_samples
            misses = ~hits
            
            # Plot hits in green, misses in red
            ax.scatter(x_samples[hits], y_samples[hits], c='green', alpha=0.6, s=10, label='Hits')
            ax.scatter(x_samples[misses], y_samples[misses], c='red', alpha=0.6, s=10, label='Misses')
            
        else:  # average_value
            # Plot sample points
            x_samples = result['x_samples'][:1000]
            f_samples = result['f_samples'][:1000]
            ax.scatter(x_samples, f_samples, c='green', alpha=0.6, s=10, label='Samples')
        
        # Fill area under curve
        ax.fill_between(x_range, y_range, alpha=0.3, label='Integral Area')
        
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Monte Carlo Integration: {method.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def get_example_functions(self) -> List[Dict[str, Any]]:
        """Get example functions for Monte Carlo integration"""
        return [
            {
                'name': 'sin(x)',
                'function': 'sin(x)',
                'description': 'sin(x) from 0 to π',
                'a': 0,
                'b': 'pi',
                'n_samples': 1000
            },
            {
                'name': 'x²',
                'function': 'x**2',
                'description': 'x² from 0 to 2',
                'a': 0,
                'b': 2,
                'n_samples': 1000
            },
            {
                'name': 'exp(-x²)',
                'function': 'exp(-x**2)',
                'description': 'Gaussian function from -2 to 2',
                'a': -2,
                'b': 2,
                'n_samples': 2000
            },
            {
                'name': '1/(1+x²)',
                'function': '1/(1+x**2)',
                'description': '1/(1+x²) from 0 to 1',
                'a': 0,
                'b': 1,
                'n_samples': 1000
            }
        ]
    
    def get_input_fields(self) -> List[Dict[str, str]]:
        """Get input field definitions for the UI"""
        return [
            {'name': 'function', 'label': 'Function f(x): *', 'type': 'entry', 'width': 30},
            {'name': 'a', 'label': 'Lower limit (a): *', 'type': 'entry', 'width': 15},
            {'name': 'b', 'label': 'Upper limit (b): *', 'type': 'entry', 'width': 15},
            {'name': 'n_samples', 'label': 'Number of samples: *', 'type': 'entry', 'width': 15},
            {'name': 'method', 'label': 'Method:', 'type': 'combobox', 'width': 15, 'values': ['hit_or_miss', 'average_value']}
        ]
    
    def validate_inputs(self, function: str = "", a: str = "0", b: str = "1", n_samples: str = "1000", method: str = "hit_or_miss") -> Tuple[bool, str]:
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
            int(n_samples)
        except ValueError:
            return False, "Number of samples must be an integer"
        
        if int(n_samples) <= 0:
            return False, "Number of samples must be positive"
        
        return True, ""
