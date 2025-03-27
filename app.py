# Import necessary libraries
import streamlit as st
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
import math
from queue import PriorityQueue
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate
from functions.BranchAndBoundSolver import BranchAndBoundSolver
from functions.DualSimplexSolver import DualSimplexSolver
from functions.simplex_solver_with_steps import simplex_solver_with_steps

# --- Streamlit App Layout ---

# Title and solver selection
st.title("Linear Programming Solver")

# Sidebar for solver selection and inputs
st.sidebar.header("Select Solver")
solver_type = st.sidebar.selectbox(
    "Choose a solver method",
    ["Branch and Bound (Integer LP)", "Dual Simplex", "Primal Simplex"]
)

st.sidebar.header("Input Problem Parameters")

# Common inputs
if solver_type == "Branch and Bound (Integer LP)":
    # Objective Function Coefficients
    c_input = st.sidebar.text_input("Objective Coefficients (comma-separated)", "5,1")
    c = np.array([float(x.strip()) for x in c_input.split(",")])

    # Constraint Matrix (A)
    A_input = st.sidebar.text_area(
        "Constraint Matrix (rows separated by newlines, values comma-separated)",
        "-1,2\n1,-1\n4,1"
    )
    A = np.array([[float(x.strip()) for x in row.split(",")] for row in A_input.split("\n")])

    # Right-hand Side (b)
    b_input = st.sidebar.text_input("Right-hand Side Values (comma-separated)", "4.5,1.5,12")
    b = np.array([float(x.strip()) for x in b_input.split(",")])

    # Variable types
    var_type = st.sidebar.radio("Variable Type", ["Integer", "Binary", "Mixed"])
    
    if var_type == "Mixed":
        integer_vars_input = st.sidebar.text_input("Integer Variable Indices (comma-separated, 0-based)", "0,1")
        integer_vars = [int(x.strip()) for x in integer_vars_input.split(",") if x.strip()]
        binary_vars = []
    elif var_type == "Binary":
        integer_vars = []
        binary_vars = list(range(len(c)))
    else:  # Integer
        integer_vars = list(range(len(c)))
        binary_vars = []

    # Maximization or Minimization
    maximize = st.sidebar.checkbox("Maximize", value=True)

elif solver_type == "Dual Simplex":
    # Objective Function Coefficients
    c_input = st.sidebar.text_input("Objective Coefficients (comma-separated)", "-1,2")
    c = [float(x.strip()) for x in c_input.split(",")]

    # Number of constraints
    num_constraints = st.sidebar.number_input("Number of Constraints", min_value=1, value=2)
    
    # Constraint Matrix (A), Relations, and RHS
    A = []
    relations = []
    b = []
    
    for i in range(int(num_constraints)):
        col1, col2, col3 = st.sidebar.columns(3)
        
        with col1:
            A_row_input = st.text_input(f"Constraint {i+1} coefficients", value="5,4" if i==0 else "1,5")
            A.append([float(x.strip()) for x in A_row_input.split(",")])
        
        with col2:
            rel = st.selectbox(f"Relation {i+1}", options=["<=", "=", ">="], index=2 if i==0 else 1)
            relations.append(rel)
        
        with col3:
            b_val = st.text_input(f"RHS {i+1}", value="20" if i==0 else "10")
            b.append(float(b_val))
    
    # Objective type (max/min)
    obj_type = st.sidebar.radio("Objective Type", ["max", "min"], index=0)

else:  # Primal Simplex
    # Objective Function Coefficients
    c_input = st.sidebar.text_input("Objective Coefficients (comma-separated)", "5,1")
    c = np.array([float(x.strip()) for x in c_input.split(",")])

    # Constraint Matrix (A)
    A_input = st.sidebar.text_area(
        "Constraint Matrix (rows separated by newlines, values comma-separated)",
        "-1,2\n1,-1\n4,1"
    )
    A = np.array([[float(x.strip()) for x in row.split(",")] for row in A_input.split("\n")])

    # Right-hand Side (b)
    b_input = st.sidebar.text_input("Right-hand Side Values (comma-separated)", "4.5,1.5,12")
    b = np.array([float(x.strip()) for x in b_input.split(",")])

    # Bounds
    bounds = [(0, None) for _ in range(len(c))]

# Solve Button
if st.sidebar.button("Solve"):
    if solver_type == "Branch and Bound (Integer LP)":
        # Initialize the Branch and Bound Solver
        solver = BranchAndBoundSolver(c, A, b, 
                                     integer_vars=integer_vars,
                                     binary_vars=binary_vars, 
                                     maximize=maximize)

        # Solve the problem
        with st.spinner("Solving the integer programming problem..."):
            solution, objective = solver.solve(verbose=False)

        # Display Results
        st.header("Results")
        if solution is not None:
            st.subheader("Optimal Solution")
            st.write(f"Objective Value: {objective:.6f}")
            st.write("Solution:")
            for i, val in enumerate(solution):
                st.write(f"x_{i} = {val:.6f}")
        else:
            st.write("No feasible integer solution found.")

        # Display Steps Table
        st.subheader("Steps Table")
        st.table(solver.steps_table)

        # Visualize Branch and Bound Tree
        st.subheader("Branch and Bound Tree")
        solver.visualize_graph()

        # Validation with CVXPY's Integer Solver
        st.subheader("Validation with CVXPY's Integer Solver")
        try:
            x = cp.Variable(len(c), integer=bool(integer_vars) or None, boolean=bool(binary_vars) or None)
            objective_fn = cp.Maximize(c @ x) if maximize else cp.Minimize(c @ x)
            constraints = [A @ x <= b, x >= 0]
            
            if binary_vars:
                constraints.append(x <= 1)
                
            int_prob = cp.Problem(objective_fn, constraints)
            int_result = int_prob.solve()
            st.write(f"CVXPY Integer Optimizer Objective: {int_result}")
            st.write(f"CVXPY Integer Optimizer Solution: {x.value}")
        except Exception as e:
            st.write(f"CVXPY validation failed: {str(e)}")

        # Continuous Solution for Comparison
        st.subheader("Continuous Solution for Comparison")
        try:
            x_cont = cp.Variable(len(c))
            objective_cont = cp.Maximize(c @ x_cont) if maximize else cp.Minimize(c @ x_cont)
            constraints_cont = [A @ x_cont <= b, x_cont >= 0]
            cont_prob = cp.Problem(objective_cont, constraints_cont)
            cont_result = cont_prob.solve()
            st.write(f"Optimal Continuous Objective Value: {cont_result}")
            st.write(f"Optimal Continuous Solution: {x_cont.value}")
        except Exception as e:
            st.write(f"Continuous solution calculation failed: {str(e)}")

    elif solver_type == "Dual Simplex":
        # Display the problem
        st.header("Dual Simplex Method")
        st.subheader("Problem Formulation")
        
        obj_str = " + ".join([f"{coef}x{i}" for i, coef in enumerate(c)])
        st.write(f"{'Maximize' if obj_type == 'max' else 'Minimize'} {obj_str}")
        
        st.write("Subject to:")
        for i in range(len(A)):
            constraint_str = " + ".join([f"{coef}x{j}" for j, coef in enumerate(A[i])])
            st.write(f"{constraint_str} {relations[i]} {b[i]}")
        
        st.write("x_i ≥ 0 for all i")
        
        # Solve using DualSimplexSolver
        with st.spinner("Solving with Dual Simplex Method..."):
            try:
                solver = DualSimplexSolver(obj_type, c, A, relations, b)
                st.subheader("Solution Process")
                # Create a placeholder for capturing outputs
                solution_steps = st.empty()
                with st.container():
                    result = solver.solve()
                
                # Display solution
                st.subheader("Final Solution")
                if result:
                    st.write(f"Optimal value: {solver.optimal_value}")
                    st.write("Optimal solution:")
                    for i, val in enumerate(solver.optimal_solution):
                        st.write(f"x_{i} = {val}")
                else:
                    st.write("The problem has no feasible solution or is unbounded.")
                
            except Exception as e:
                st.error(f"Error solving the problem: {str(e)}")
    
    else:  # Primal Simplex
        # Solve using simplex_solver_with_steps
        with st.spinner("Solving with Primal Simplex Method..."):
            solution, objective = simplex_solver_with_steps(c, A, b, bounds)
            
        if solution is not None:
            st.header("Final Solution")
            st.write(f"Optimal objective value: {objective:.6f}")
            st.write("Optimal solution:")
            for i, val in enumerate(solution):
                st.write(f"x_{i} = {val:.6f}")
        else:
            st.error("The problem has no feasible solution or is unbounded.")

# Add explanation and examples section
with st.expander("About the Solvers"):
    st.markdown("""
    ## About the LP Solvers
    
    ### Branch and Bound (Integer LP)
    The Branch and Bound algorithm is used to solve integer and mixed-integer linear programming problems.
    It works by solving a series of LP relaxations and "branching" on integer variables that have fractional values.
    
    ### Dual Simplex
    The Dual Simplex method is particularly efficient for re-optimizing problems where constraints have been added
    or the right-hand side has been modified. It maintains dual feasibility while working toward primal feasibility.
    
    ### Primal Simplex
    The Primal Simplex method is the classic approach for solving linear programming problems.
    It maintains primal feasibility while improving the objective function value at each iteration.
    
    ## Example Problems
    
    ### For Branch and Bound:
    - Objective: Maximize 5x₀ + 1x₁
    - Constraints:
      - -x₀ + 2x₁ ≤ 4.5
      - x₀ - x₁ ≤ 1.5
      - 4x₀ + x₁ ≤ 12
    - x₀, x₁ are integers
    
    ### For Dual Simplex:
    - Objective: Maximize -x₀ + 2x₁
    - Constraints:
      - 5x₀ + 4x₁ ≥ 20
      - x₀ + 5x₁ = 10
    - x₀, x₁ ≥ 0
    """)