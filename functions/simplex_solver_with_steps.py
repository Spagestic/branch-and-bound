import numpy as np
import streamlit as st
from tabulate import tabulate


def simplex_solver_with_steps(c, A, b, bounds):
    """
    Solve LP using simplex method and display full tableau at each step
    
    Parameters:
    - c: Objective coefficients (for maximizing c'x)
    - A: Constraint coefficients matrix
    - b: Right-hand side of constraints
    - bounds: Variable bounds as [(lower_1, upper_1), (lower_2, upper_2), ...]
    
    Returns:
    - x: Optimal solution
    - optimal_value: Optimal objective value
    """
    st.markdown("\n--- Starting Simplex Method ---")
    st.text(f"Objective: Maximize {' + '.join([f'{c[i]}x_{i}' for i in range(len(c))])}")
    st.text(f"Constraints:")
    for i in range(len(b)):
        constraint_str = ' + '.join([f"{A[i,j]}x_{j}" for j in range(A.shape[1])])
        st.text(f"  {constraint_str} <= {b[i]}")
    
    # Convert problem to standard form (for tableau method)
    # First handle bounds by adding necessary constraints
    A_with_bounds = A.copy()
    b_with_bounds = b.copy()
    
    for i, (lb, ub) in enumerate(bounds):
        if lb is not None and lb > 0:
            # For variables with lower bounds > 0, we'll substitute x_i = x_i' + lb
            # This affects all constraints where x_i appears
            for j in range(A.shape[0]):
                b_with_bounds[j] -= A[j, i] * lb
    
    # Number of variables and constraints
    n_vars = len(c)
    n_constraints = A.shape[0]
    
    # Add slack variables to create standard form
    # The tableau will have: [objective row | RHS]
    #                        [-------------|----]
    #                        [constraints  | RHS]
    
    # Initial tableau: 
    # First row is -c (negative of objective coefficients) and 0s for slack variables, then 0 (for max)
    # The rest are constraint coefficients, then identity matrix for slack variables, then RHS
    tableau = np.zeros((n_constraints + 1, n_vars + n_constraints + 1))
    
    # Set the objective row (negated for maximization)
    tableau[0, :n_vars] = -c
    
    # Set the constraint coefficients
    tableau[1:, :n_vars] = A_with_bounds
    
    # Set the slack variable coefficients (identity matrix)
    for i in range(n_constraints):
        tableau[i + 1, n_vars + i] = 1
    
    # Set the RHS
    tableau[1:, -1] = b_with_bounds
    
    # Base and non-base variables
    base_vars = list(range(n_vars, n_vars + n_constraints))  # Slack variables are initially basic
    
    # Function to print current tableau
    def print_tableau(tableau, base_vars):
        headers = [f"x_{j}" for j in range(n_vars)] + [f"s_{j}" for j in range(n_constraints)] + ["RHS"]
        rows = []
        row_labels = ["z"] + [f"eq_{i}" for i in range(n_constraints)]
        
        for i, row in enumerate(tableau):
            rows.append([row_labels[i]] + [f"{val:.3f}" for val in row])
        
        st.text("\nCurrent Tableau:")
        st.text(tabulate(rows, headers=headers, tablefmt="grid"))
        st.text(f"Basic variables: {[f'x_{v}' if v < n_vars else f's_{v-n_vars}' for v in base_vars]}")
    
    # Print initial tableau
    st.text("\nInitial tableau:")
    print_tableau(tableau, base_vars)
    
    # Main simplex loop
    iteration = 0
    max_iterations = 100  # Prevent infinite loops
    
    while iteration < max_iterations:
        iteration += 1
        st.text(f"\n--- Iteration {iteration} ---")
        
        # Find the entering variable (most negative coefficient in objective row for maximization)
        entering_col = np.argmin(tableau[0, :-1])
        if tableau[0, entering_col] >= -1e-10:  # Small negative numbers due to floating-point errors
            st.text("Optimal solution reached - no negative coefficients in objective row")
            break
        
        st.text(f"Entering variable: {'x_' + str(entering_col) if entering_col < n_vars else 's_' + str(entering_col - n_vars)}")
        
        # Find the leaving variable using min ratio test
        ratios = []
        for i in range(1, n_constraints + 1):
            if tableau[i, entering_col] <= 0:
                ratios.append(np.inf)  # Avoid division by zero or negative
            else:
                ratios.append(tableau[i, -1] / tableau[i, entering_col])
        
        if all(r == np.inf for r in ratios):
            st.text("Unbounded solution - no leaving variable found")
            return None, float('inf')  # Problem is unbounded
        
        # Find the row with minimum ratio
        leaving_row = np.argmin(ratios) + 1  # +1 because we skip the objective row
        leaving_var = base_vars[leaving_row - 1]
        
        st.text(f"Leaving variable: {'x_' + str(leaving_var) if leaving_var < n_vars else 's_' + str(leaving_var - n_vars)}")
        st.text(f"Pivot element: {tableau[leaving_row, entering_col]:.3f} at row {leaving_row}, column {entering_col}")
        
        # Perform pivot operation
        # First, normalize the pivot row
        pivot = tableau[leaving_row, entering_col]
        tableau[leaving_row] = tableau[leaving_row] / pivot
        
        # Update other rows
        for i in range(tableau.shape[0]):
            if i != leaving_row:
                factor = tableau[i, entering_col]
                tableau[i] = tableau[i] - factor * tableau[leaving_row]
        
        # Update basic variables
        base_vars[leaving_row - 1] = entering_col
        
        # Print updated tableau
        st.text("\nAfter pivot:")
        print_tableau(tableau, base_vars)
    
    if iteration == max_iterations:
        st.text("Max iterations reached without convergence")
        return None, None
    
    # Extract solution
    x = np.zeros(n_vars)
    for i, var in enumerate(base_vars):
        if var < n_vars:  # If it's an original variable and not a slack
            x[var] = tableau[i + 1, -1]
    
    # Account for variable substitutions (if lower bounds were applied)
    for i, (lb, _) in enumerate(bounds):
        if lb is not None and lb > 0:
            x[i] += lb
    
    # Calculate objective value
    optimal_value = np.dot(c, x)
    
    st.markdown("\n--- Simplex Method Complete ---")
    st.text(f"Optimal solution found: {x}")
    st.text(f"Optimal objective value: {optimal_value}")
    
    return x, optimal_value

