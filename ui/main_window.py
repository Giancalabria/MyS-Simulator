#!/usr/bin/env python3
"""
Main GUI application for Simple Numerical Methods Simulator
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import os
import sys

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.fixed_point_method import FixedPointMethod
from methods.monte_carlo_method import MonteCarloMethod
from methods.trapezoidal_method import TrapezoidalMethod
from methods.simpson_13_method import Simpson13Method
from methods.simpson_38_method import Simpson38Method
from methods.newton_raphson_method import NewtonRaphsonMethod
from methods.newton_cotes_method import NewtonCotesMethod
from methods.runge_kutta_method import RungeKuttaMethod
from methods.bisection_method import BisectionMethod

class SimpleSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Numerical Methods Simulator")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # Initialize methods
        self.methods = {
            "Fixed Point Method": FixedPointMethod(),
            "Bisection Method": BisectionMethod(),
            "Monte Carlo Integration": MonteCarloMethod(),
            "Trapezoidal Rule": TrapezoidalMethod(),
            "Simpson's 1/3 Rule": Simpson13Method(),
            "Simpson's 3/8 Rule": Simpson38Method(),
            "Newton-Raphson Method": NewtonRaphsonMethod(),
            "Newton-Cotes Method": NewtonCotesMethod(),
            "Runge-Kutta Method": RungeKuttaMethod()
        }
        self.current_method = "Fixed Point Method"
        
        # Variables
        self.function_var = tk.StringVar(value="cos(x)")
        self.x0_var = tk.StringVar(value="0.5")
        self.tolerance_var = tk.StringVar(value="1e-8")
        self.max_iter_var = tk.StringVar(value="50")
        self.decimals_var = tk.IntVar(value=8)
        
        # Results storage
        self.last_result = None
        self.last_iterations = []
        
        # Row numbers for layout
        self.input_row = 2
        self.results_row = 2
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Numerical Methods Simulator", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Method selection
        method_frame = ttk.Frame(main_frame)
        method_frame.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT, padx=(0, 5))
        self.method_var = tk.StringVar(value=self.current_method)
        method_combo = ttk.Combobox(method_frame, textvariable=self.method_var, 
                                   values=list(self.methods.keys()), state="readonly", width=20)
        method_combo.pack(side=tk.LEFT, padx=(0, 10))
        method_combo.bind('<<ComboboxSelected>>', self.on_method_changed)
        
        # Input section
        self.setup_input_section(main_frame)
        
        # Results section
        self.setup_results_section(main_frame)
        
    def setup_input_section(self, parent):
        """Setup input controls"""
        # Input frame
        self.input_frame = ttk.LabelFrame(parent, text="Parameters", padding="10")
        self.input_frame.grid(row=self.input_row, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Create input fields dynamically
        self.create_input_fields()
        
        # Buttons
        button_frame = ttk.Frame(self.input_frame)
        button_frame.grid(row=100, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Calculate", command=self.calculate).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Example", command=self.load_example).pack(side=tk.LEFT, padx=5)
        
        # Examples dropdown
        self.setup_examples_dropdown()
    
    def create_input_fields(self):
        """Create input fields based on current method"""
        # Clear existing fields
        for widget in self.input_frame.winfo_children():
            if isinstance(widget, (ttk.Entry, ttk.Spinbox, ttk.Label)):
                widget.destroy()
        
        # Get current method
        method = self.methods[self.current_method]
        fields = method.get_input_fields()
        
        # Create fields
        for i, field in enumerate(fields):
            ttk.Label(self.input_frame, text=field['label']).grid(row=i, column=0, sticky=tk.W, pady=2)
            
            if field['type'] == 'entry':
                # Create variable if it doesn't exist
                if not hasattr(self, f"{field['name']}_var"):
                    setattr(self, f"{field['name']}_var", tk.StringVar())
                entry = ttk.Entry(self.input_frame, textvariable=getattr(self, f"{field['name']}_var"), 
                                width=field.get('width', 20))
                entry.grid(row=i, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
            elif field['type'] == 'spinbox':
                if not hasattr(self, f"{field['name']}_var"):
                    setattr(self, f"{field['name']}_var", tk.StringVar())
                spinbox = ttk.Spinbox(self.input_frame, from_=field.get('from_', 0), 
                                    to=field.get('to', 100), textvariable=getattr(self, f"{field['name']}_var"), 
                                    width=field.get('width', 15))
                spinbox.grid(row=i, column=1, sticky=tk.W, pady=2, padx=(5, 0))
            elif field['type'] == 'combobox':
                if not hasattr(self, f"{field['name']}_var"):
                    setattr(self, f"{field['name']}_var", tk.StringVar())
                combo = ttk.Combobox(self.input_frame, textvariable=getattr(self, f"{field['name']}_var"),
                                   values=field.get('values', []), state="readonly", width=field.get('width', 15))
                combo.grid(row=i, column=1, sticky=tk.W, pady=2, padx=(5, 0))
                # Set default value
                if field.get('values'):
                    combo.set(field['values'][0])
    
    def setup_examples_dropdown(self):
        """Setup examples dropdown"""
        ttk.Label(self.input_frame, text="Examples:").grid(row=99, column=0, sticky=tk.W, pady=(10, 2))
        self.example_var = tk.StringVar()
        examples = self.methods[self.current_method].get_example_functions()
        example_names = [ex['name'] for ex in examples]
        example_combo = ttk.Combobox(self.input_frame, textvariable=self.example_var, 
                                   values=example_names, state="readonly", width=25)
        example_combo.grid(row=99, column=1, sticky=(tk.W, tk.E), pady=(10, 2), padx=(5, 0))
        example_combo.bind('<<ComboboxSelected>>', self.on_example_selected)
        
        
    def setup_results_section(self, parent):
        """Setup results display"""
        # Results frame
        self.results_frame = ttk.LabelFrame(parent, text="Results", padding="10")
        self.results_frame.grid(row=self.results_row, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(1, weight=1)
        
        # Results info - Made bigger with larger font
        self.result_info_var = tk.StringVar(value="No calculations yet")
        result_info = ttk.Label(self.results_frame, textvariable=self.result_info_var, 
                               font=("Arial", 14, "bold"), wraplength=500)
        result_info.grid(row=0, column=0, pady=(0, 15))
        
        # Notebook for tabs
        notebook = ttk.Notebook(self.results_frame)
        notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Iterations table tab
        self.setup_iterations_table_tab(notebook)
        
        # Graph tab
        self.setup_graph_tab(notebook)
        
        # Explanation tab
        self.setup_explanation_tab(notebook)
        
    def setup_iterations_table_tab(self, notebook):
        """Setup detailed iterations table tab"""
        table_frame = ttk.Frame(notebook)
        notebook.add(table_frame, text="Iterations Table")
        
        # Create a frame for the table and scrollbar
        table_container = ttk.Frame(table_frame)
        table_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview for iterations with more columns
        columns = ("Iter", "x_n", "g(x_n)", "|x_{n+1} - x_n|", "Rel Error", "Convergence")
        self.tree = ttk.Treeview(table_container, columns=columns, show="headings", height=20)
        
        # Configure columns with better widths
        column_widths = {"Iter": 60, "x_n": 120, "g(x_n)": 120, "|x_{n+1} - x_n|": 120, "Rel Error": 120, "Convergence": 100}
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths[col], anchor=tk.E)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add summary frame below table
        summary_frame = ttk.LabelFrame(table_frame, text="Summary", padding="5")
        summary_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        self.summary_var = tk.StringVar(value="No calculations yet")
        summary_label = ttk.Label(summary_frame, textvariable=self.summary_var, 
                                 font=("Arial", 10), wraplength=600)
        summary_label.pack(anchor=tk.W)
        
    def setup_graph_tab(self, notebook):
        """Setup graph tab"""
        graph_frame = ttk.Frame(notebook)
        notebook.add(graph_frame, text="Graph")
        
        # Create a container frame for canvas and toolbar
        canvas_frame = ttk.Frame(graph_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        toolbar.update()
        
    def setup_explanation_tab(self, notebook):
        """Setup explanation tab with method theory and details"""
        explanation_frame = ttk.Frame(notebook)
        notebook.add(explanation_frame, text="Method Explanation")
        
        # Create scrollable text widget
        text_frame = ttk.Frame(explanation_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Text widget for explanation
        self.explanation_text = tk.Text(text_frame, wrap=tk.WORD, font=("Arial", 11), 
                                       height=25, width=80)
        self.explanation_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar for text
        text_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.explanation_text.yview)
        self.explanation_text.configure(yscrollcommand=text_scrollbar.set)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Make text read-only
        self.explanation_text.config(state=tk.DISABLED)
        
        # Update explanation when function changes
        self.function_var.trace('w', self.update_explanation_display)
        self.update_explanation_display()
    
    def on_method_changed(self, event):
        """Handle method selection change"""
        self.current_method = self.method_var.get()
        self.create_input_fields()
        self.setup_examples_dropdown()
        self.clear()
        self.load_example_for_method()
        self.update_explanation_display()
        
    def calculate(self):
        """Calculate using current method"""
        try:
            # Get current method
            method = self.methods[self.current_method]
            
            # Get parameters dynamically based on method
            params = {}
            for field in method.get_input_fields():
                field_name = field['name']
                if hasattr(self, f"{field_name}_var"):
                    value = getattr(self, f"{field_name}_var").get()
                    # Convert mathematical constants
                    value = self.convert_math_constants(value)
                    # Convert to appropriate type
                    if field_name in ['x0', 'y0', 'a', 'b', 'h', 'tolerance']:
                        params[field_name] = float(value)
                    elif field_name in ['n', 'n_samples', 'n_steps', 'max_iter']:
                        params[field_name] = int(value)
                    else:
                        params[field_name] = value
            
            # Validate inputs
            is_valid, error_msg = method.validate_inputs(**params)
            
            if not is_valid:
                messagebox.showerror("Error", error_msg)
                return
            
            # Update UI
            self.root.update()
            
            # Calculate
            result = method.calculate(**params)
            self.last_result = result
            self.last_iterations = result.get('iterations', [])
            
            # Update display
            self.update_results_display()
            self.update_iterations_table()
            self.update_graph()
            self.update_explanation_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Calculation failed: {str(e)}")
    
    def convert_math_constants(self, value: str) -> str:
        """Convert mathematical constants in string to their numerical values"""
        if not isinstance(value, str):
            return value
        
        # Dictionary of mathematical constants
        constants = {
            'pi': '3.141592653589793',
            'π': '3.141592653589793',
            'e': '2.718281828459045',
            'E': '2.718281828459045',
        }
        
        # Replace constants with their numerical values
        result = value
        for constant, numerical_value in constants.items():
            # Replace standalone constants (not part of other words)
            import re
            pattern = r'\b' + re.escape(constant) + r'\b'
            result = re.sub(pattern, numerical_value, result)
        
        return result
    
    def update_results_display(self):
        """Update the results info display"""
        if not self.last_result:
            self.result_info_var.set("No calculations yet")
            return
        
        result = self.last_result
        decimals = self.decimals_var.get()
        
        # Handle different result types
        if 'root' in result:
            # Root finding methods
            info = f"Root: {result['root']:.{decimals}f}\n"
            info += f"Iterations: {len(result['iterations'])}\n"
            info += f"Converged: {'Yes' if result['converged'] else 'No'}\n"
            
            if 'convergence_check' in result and 'warning' in result['convergence_check']:
                info += f"Convergence: {result['convergence_check']['warning']}"
        
        elif 'integral' in result:
            # Integration methods
            info = f"Integral: {result['integral']:.{decimals}f}\n"
            info += f"Intervals/Steps: {result.get('n_intervals', result.get('n_points', result.get('n_samples', 'N/A')))}\n"
            if 'error_estimate' in result and result['error_estimate']:
                info += f"Error Estimate: {result['error_estimate']:.{decimals}e}\n"
            if 'method' in result:
                info += f"Method: {result['method']}\n"
        
        elif 'final_y' in result:
            # ODE methods
            info = f"Final Value: y({result['final_x']:.{decimals}f}) = {result['final_y']:.{decimals}f}\n"
            info += f"Steps: {result['n_steps']}\n"
            info += f"Step Size: {result['h']:.{decimals}f}\n"
            info += f"Method: {result['method']}\n"
        
        else:
            # Generic result
            info = "Calculation completed\n"
            for key, value in result.items():
                if key != 'iterations':
                    info += f"{key}: {value}\n"
        
        self.result_info_var.set(info)
    
    def update_iterations_table(self):
        """Update the detailed iterations table"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if not self.last_iterations:
            return
        
        # Update table columns based on method
        method = self.methods[self.current_method]
        if 'Monte Carlo' in self.current_method:
            columns = ("Sample", "x", "f(x)", "y", "Hit", "Weight")
        elif 'Trapezoidal' in self.current_method or 'Simpson' in self.current_method or 'Newton-Cotes' in self.current_method:
            columns = ("Point", "x", "f(x)", "Weight", "Error", "Status")
        elif 'Newton-Raphson' in self.current_method:
            columns = ("Iter", "x_n", "f(x_n)", "f'(x_n)", "x_next", "Error")
        elif 'Bisection' in self.current_method:
            columns = ("Iter", "a_n", "b_n", "c_n", "f(c_n)", "Error")
        elif 'Runge-Kutta' in self.current_method:
            columns = ("Step", "x", "y", "k1", "k2", "k3", "k4")
        else:
            columns = ("Iter", "x_n", "g(x_n)", "Error", "Rel Error", "Status")
        
        # Update tree columns
        self.tree['columns'] = columns
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.E)
        
        # Add iterations with method-specific data
        decimals = self.decimals_var.get()
        for i, iter_data in enumerate(self.last_iterations):
            if 'Monte Carlo' in self.current_method:
                values = (
                    iter_data['iteration'],
                    f"{iter_data['x_sample']:.{decimals}f}",
                    f"{iter_data['f_sample']:.{decimals}f}",
                    f"{iter_data.get('y_sample', 0):.{decimals}f}",
                    "✓" if iter_data.get('hit') else "✗",
                    f"{iter_data.get('weight', 1):.{decimals}f}"
                )
            elif 'Trapezoidal' in self.current_method or 'Simpson' in self.current_method or 'Newton-Cotes' in self.current_method:
                values = (
                    iter_data['iteration'],
                    f"{iter_data['x']:.{decimals}f}",
                    f"{iter_data['f_x']:.{decimals}f}",
                    f"{iter_data['weight']:.{decimals}f}",
                    "",
                    ""
                )
            elif 'Newton-Raphson' in self.current_method:
                values = (
                    iter_data['iteration'],
                    f"{iter_data['x_n']:.{decimals}f}",
                    f"{iter_data['f_x_n']:.{decimals}f}",
                    f"{iter_data['f_prime_x_n']:.{decimals}f}",
                    f"{iter_data['x_next']:.{decimals}f}",
                    f"{iter_data['abs_error']:.{decimals}e}"
                )
            elif 'Bisection' in self.current_method:
                values = (
                    iter_data['iteration'],
                    f"{iter_data['a_n']:.{decimals}f}",
                    f"{iter_data['b_n']:.{decimals}f}",
                    f"{iter_data['c_n']:.{decimals}f}",
                    f"{iter_data['f_c_n']:.{decimals}f}",
                    f"{iter_data['abs_error']:.{decimals}e}"
                )
            elif 'Runge-Kutta' in self.current_method:
                # Handle None values for k2, k3, k4 (they are None for RK1 and RK2)
                k2_val = iter_data.get('k2')
                k3_val = iter_data.get('k3')
                k4_val = iter_data.get('k4')
                
                values = (
                    iter_data['iteration'],
                    f"{iter_data['x']:.{decimals}f}",
                    f"{iter_data['y']:.{decimals}f}",
                    f"{iter_data['k1']:.{decimals}f}",
                    f"{k2_val:.{decimals}f}" if k2_val is not None else "N/A",
                    f"{k3_val:.{decimals}f}" if k3_val is not None else "N/A",
                    f"{k4_val:.{decimals}f}" if k4_val is not None else "N/A"
                )
            else:
                # Default for Fixed Point
                convergence_status = "✓" if iter_data.get('abs_error', 0) <= 1e-8 else "✗"
                values = (
                    iter_data['iteration'],
                    f"{iter_data['x_n']:.{decimals}f}",
                    f"{iter_data['g_x_n']:.{decimals}f}",
                    f"{iter_data.get('abs_error', 0):.{decimals}e}",
                    f"{iter_data.get('rel_error', 0):.{decimals}e}",
                    convergence_status
                )
            
            self.tree.insert("", tk.END, values=values)
        
        # Update summary
        self.update_table_summary()
    
    def update_table_summary(self):
        """Update the table summary"""
        if not self.last_result or not self.last_iterations:
            self.summary_var.set("No calculations yet")
            return
        
        result = self.last_result
        iterations = self.last_iterations
        decimals = self.decimals_var.get()
        
        # Handle different result types
        if 'root' in result:
            # Root finding methods
            summary = f"Total Iterations: {len(iterations)} | "
            summary += f"Converged: {'Yes' if result['converged'] else 'No'} | "
            summary += f"Final Root: {result['root']:.{decimals}f} | "
            
            if iterations and 'abs_error' in iterations[0]:
                first_error = iterations[0]['abs_error']
                last_error = iterations[-1]['abs_error']
                summary += f"Error Reduction: {first_error:.2e} → {last_error:.2e}"
        
        elif 'integral' in result:
            # Integration methods
            summary = f"Total Points: {len(iterations)} | "
            summary += f"Integral: {result['integral']:.{decimals}f} | "
            summary += f"Intervals: {result.get('n_intervals', result.get('n_points', 'N/A'))} | "
            if 'error_estimate' in result and result['error_estimate']:
                summary += f"Error: {result['error_estimate']:.2e}"
        
        elif 'final_y' in result:
            # ODE methods
            summary = f"Total Steps: {len(iterations)} | "
            summary += f"Final Value: {result['final_y']:.{decimals}f} | "
            summary += f"Method: {result['method']} | "
            summary += f"Step Size: {result['h']:.{decimals}f}"
        
        else:
            # Generic result
            summary = f"Total Steps: {len(iterations)} | "
            summary += f"Completed: {'Yes' if result.get('converged', True) else 'No'}"
        
        self.summary_var.set(summary)
    
    def update_graph(self):
        """Update the graph"""
        self.ax.clear()
        
        if not self.last_iterations:
            self.ax.text(0.5, 0.5, 'No data to display', 
                        transform=self.ax.transAxes, ha='center', va='center')
            self.canvas.draw()
            return
        
        # Get current method and plot using its method
        method = self.methods[self.current_method]
        function_expr = self.function_var.get().strip()
        
        method.plot_function_and_iterations(self.ax, function_expr, self.last_iterations, self.last_result)
        
        self.canvas.draw()
    
    def update_explanation_display(self, *args):
        """Update the method explanation display"""
        try:
            # Get current method
            method = self.methods[self.current_method]
            
            # Get parameters dynamically
            params = {}
            for field in method.get_input_fields():
                field_name = field['name']
                if hasattr(self, f"{field_name}_var"):
                    params[field_name] = getattr(self, f"{field_name}_var").get()
            
            # Get explanation from method
            explanation_text = method.get_explanation(**params)
            
            self.explanation_text.config(state=tk.NORMAL)
            self.explanation_text.delete(1.0, tk.END)
            self.explanation_text.insert(1.0, explanation_text)
            self.explanation_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.explanation_text.config(state=tk.NORMAL)
            self.explanation_text.delete(1.0, tk.END)
            self.explanation_text.insert(1.0, f"Error generating explanation: {str(e)}")
            self.explanation_text.config(state=tk.DISABLED)
    
    def clear(self):
        """Clear all results"""
        self.last_result = None
        self.last_iterations = []
        
        # Clear table
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Clear graph
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'No data to display', 
                    transform=self.ax.transAxes, ha='center', va='center')
        self.canvas.draw()
        
        # Clear info and summary
        self.result_info_var.set("No calculations yet")
        self.summary_var.set("No calculations yet")
        
        # Update explanation
        self.update_explanation_display()
    
    def load_example(self):
        """Load example function"""
        method = self.methods[self.current_method]
        examples = method.get_example_functions()
        if not examples:
            messagebox.showwarning("Warning", "No examples available")
            return
        
        # Simple dialog to select example
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Example")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Listbox with examples
        listbox = tk.Listbox(dialog, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for i, example in enumerate(examples):
            listbox.insert(tk.END, f"{example['name']}: {example['description']}")
        
        def on_select():
            selection = listbox.curselection()
            if selection:
                example = examples[selection[0]]
                # Set all the example values dynamically
                for field in method.get_input_fields():
                    field_name = field['name']
                    if field_name in example:
                        if hasattr(self, f"{field_name}_var"):
                            value = str(example[field_name])
                            # Convert mathematical constants for display
                            value = self.convert_math_constants(value)
                            getattr(self, f"{field_name}_var").set(value)
                dialog.destroy()
        
        ttk.Button(dialog, text="Load", command=on_select).pack(pady=5)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=5)
    
    def load_example_for_method(self):
        """Automatically load the first example for the current method"""
        try:
            method = self.methods[self.current_method]
            examples = method.get_example_functions()
            
            if examples:
                # Load the first example
                example = examples[0]
                
                # Set all the example values
                for field in method.get_input_fields():
                    field_name = field['name']
                    if field_name in example:
                        if hasattr(self, f"{field_name}_var"):
                            value = str(example[field_name])
                            # Convert mathematical constants for display
                            value = self.convert_math_constants(value)
                            getattr(self, f"{field_name}_var").set(value)
                
                # Update the example dropdown to show the loaded example
                self.example_var.set(example['name'])
                
        except Exception as e:
            # If loading example fails, just continue without error
            print(f"Could not load example: {e}")
    
    def on_example_selected(self, event):
        """Handle example selection from dropdown"""
        method = self.methods[self.current_method]
        examples = method.get_example_functions()
        selected = self.example_var.get()
        
        for example in examples:
            if example['name'] == selected:
                # Set all the example values dynamically
                for field in method.get_input_fields():
                    field_name = field['name']
                    if field_name in example:
                        if hasattr(self, f"{field_name}_var"):
                            value = str(example[field_name])
                            # Convert mathematical constants for display
                            value = self.convert_math_constants(value)
                            getattr(self, f"{field_name}_var").set(value)
                break
    
def main():
    """Main function"""
    root = tk.Tk()
    app = SimpleSimulator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
