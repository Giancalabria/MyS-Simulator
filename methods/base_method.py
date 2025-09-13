"""
Base class for numerical methods
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from utils.math_utils import safe_eval_function

class BaseMethod(ABC):
    """Base class for numerical methods"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.last_result = None
        self.last_iterations = []
    
    @abstractmethod
    def calculate(self, **kwargs) -> Dict[str, Any]:
        """Calculate the method with given parameters"""
        pass
    
    @abstractmethod
    def get_explanation(self, function: str, **kwargs) -> str:
        """Get method explanation text"""
        pass
    
    @abstractmethod
    def plot_function_and_iterations(self, ax, function: str, iterations: List[Dict], result: Dict) -> None:
        """Plot the function and iteration points"""
        pass
    
    def get_example_functions(self) -> List[Dict[str, Any]]:
        """Get example functions for this method"""
        return []
    
    def clear(self):
        """Clear results"""
        self.last_result = None
        self.last_iterations = []
    
    def get_input_fields(self) -> List[Dict[str, str]]:
        """Get input field definitions for the UI"""
        return []
    
    def validate_inputs(self, **kwargs) -> Tuple[bool, str]:
        """Validate input parameters"""
        return True, ""
