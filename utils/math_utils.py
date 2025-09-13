"""
Mathematical utilities for safe expression evaluation
"""

import sympy as sp
import numpy as np
from typing import Callable, Optional

def safe_eval_function(expr_str: str) -> Callable[[float], float]:
    """
    Safely convert a mathematical expression string to a function.
    
    Args:
        expr_str: Mathematical expression as string (e.g., "cos(x)", "x**2 + 1")
    
    Returns:
        Function that takes x and returns f(x)
    
    Raises:
        ValueError: If expression is invalid or unsafe
    """
    try:
        # Parse the expression using sympy
        x = sp.Symbol('x')
        expr = sp.sympify(expr_str)
        
        # Convert to a lambda function for numerical evaluation
        func = sp.lambdify(x, expr, modules=['numpy', 'math'])
        
        # Test the function to make sure it works
        test_value = func(1.0)
        if not np.isfinite(test_value):
            raise ValueError("Function produces non-finite values")
            
        return func
    except Exception as e:
        raise ValueError(f"Invalid function expression: {str(e)}")

def safe_eval_function_2d(expr_str: str) -> Callable[[float, float], float]:
    """
    Safely convert a mathematical expression string to a two-variable function.
    
    Args:
        expr_str: Mathematical expression as string (e.g., "x + y", "x*y", "x**2 + y**2")
    
    Returns:
        Function that takes x, y and returns f(x, y)
    
    Raises:
        ValueError: If expression is invalid or unsafe
    """
    try:
        # Parse the expression using sympy
        x, y = sp.symbols('x y')
        expr = sp.sympify(expr_str)
        
        # Convert to a lambda function for numerical evaluation
        func = sp.lambdify((x, y), expr, modules=['numpy', 'math'])
        
        # Test the function to make sure it works
        test_value = func(1.0, 1.0)
        if not np.isfinite(test_value):
            raise ValueError("Function produces non-finite values")
            
        return func
    except Exception as e:
        raise ValueError(f"Invalid function expression: {str(e)}")

def check_convergence_condition(g_func: Callable[[float], float], x0: float, h: float = 0.1) -> dict:
    """
    Check if |g'(x)| < 1 in the neighborhood of x0.
    This is a necessary condition for convergence of Fixed Point method.
    
    Args:
        g_func: Function g(x)
        x0: Point to check around
        h: Size of neighborhood to check
    
    Returns:
        Dictionary with convergence information
    """
    try:
        # Create symbolic version for derivative
        x = sp.Symbol('x')
        # We need to convert the function back to symbolic form
        # For now, we'll use numerical differentiation
        test_points = np.linspace(x0 - h, x0 + h, 5)
        derivatives = []
        
        for pt in test_points:
            # Numerical derivative: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
            h_num = 1e-6
            try:
                df = (g_func(pt + h_num) - g_func(pt - h_num)) / (2 * h_num)
                if np.isfinite(df):
                    derivatives.append(abs(df))
            except:
                continue
        
        if not derivatives:
            return {
                'max_derivative': None,
                'likely_converges': None,
                'warning': 'Could not compute derivative'
            }
        
        max_deriv = max(derivatives)
        return {
            'max_derivative': max_deriv,
            'likely_converges': max_deriv < 1,
            'warning': f"|g'(x)| ≈ {max_deriv:.3f} {'< 1' if max_deriv < 1 else '≥ 1'}"
        }
    except Exception as e:
        return {
            'max_derivative': None,
            'likely_converges': None,
            'warning': f'Error checking convergence: {str(e)}'
        }
