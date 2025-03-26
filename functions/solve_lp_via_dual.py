
import numpy as np
from scipy.optimize import linprog # Using SciPy for robust LP solving
import warnings
from functions.solve_primal_directly import solve_primal_directly

def solve_lp_via_dual(objective_type, c, A, relations, b, TOLERANCE=1e-6):
    """
    Solves an LP problem by formulating and solving the dual, then using
    complementary slackness.

    Args:
        objective_type (str): 'max' or 'min'.
        c (list or np.array): Primal objective coefficients.
        A (list of lists or np.array): Primal constraint matrix LHS.
        relations (list): Primal constraint relations ('<=', '>=', '=').
        b (list or np.array): Primal constraint RHS.

    Returns:
        A tuple: (status, message, primal_solution, dual_solution, objective_value)
        status: 0 for success, non-zero for failure.
        message: Description of the outcome.
        primal_solution: Dictionary of primal variable values (x).
        dual_solution: Dictionary of dual variable values (p).
        objective_value: Optimal objective value of the primal.
    """
    primal_c = np.array(c, dtype=float)
    primal_A = np.array(A, dtype=float)
    primal_b = np.array(b, dtype=float)
    primal_relations = relations
    num_primal_vars = len(primal_c)
    num_primal_constraints = len(primal_b)

    print("--- Step 1: Formulate the Dual Problem ---")

    # Standardize Primal for consistent dual formulation: Convert to Min problem
    original_obj_type = objective_type.lower()
    if original_obj_type == 'max':
        print("   Primal is Max. Converting to Min by negating objective coefficients.")
        primal_c_std = -primal_c
        objective_sign_flip = -1.0
    else:
        print("   Primal is Min. Using original objective coefficients.")
        primal_c_std = primal_c
        objective_sign_flip = 1.0

    # Handle constraint relations for dual formulation
    # We'll formulate the dual based on a standard primal form:
    # Min c'x s.t. Ax >= b, x >= 0
    # Dual: Max p'b s.t. p'A <= c', p >= 0
    # Let's adjust the input primal constraints to fit the Ax >= b form needed for this dual pair.

    A_geq = []
    b_geq = []
    print("   Adjusting primal constraints to >= form for dual formulation:")
    for i in range(num_primal_constraints):
        rel = primal_relations[i]
        a_row = primal_A[i]
        b_val = primal_b[i]

        if rel == '<=':
            print(f"      Constraint {i+1}: Multiplying by -1 ( {a_row} <= {b_val} -> {-a_row} >= {-b_val} )")
            A_geq.append(-a_row)
            b_geq.append(-b_val)
        elif rel == '>=':
            print(f"      Constraint {i+1}: Keeping as >= ( {a_row} >= {b_val} )")
            A_geq.append(a_row)
            b_geq.append(b_val)
        elif rel == '=':
            # Represent equality as two inequalities: >= and <=
            print(f"      Constraint {i+1}: Splitting '=' into >= and <= ")
            # >= part
            print(f"         Part 1: {a_row} >= {b_val}")
            A_geq.append(a_row)
            b_geq.append(b_val)
            # <= part -> multiply by -1 to get >=
            print(f"         Part 2: {-a_row} >= {-b_val} (from {a_row} <= {b_val})")
            A_geq.append(-a_row)
            b_geq.append(-b_val)
        else:
            return 1, f"Invalid relation '{rel}' in constraint {i+1}", None, None, None

    primal_A_std = np.array(A_geq, dtype=float)
    primal_b_std = np.array(b_geq, dtype=float)
    num_dual_vars = primal_A_std.shape[0] # One dual var per standardized constraint

    # Now formulate the dual: Max p' * primal_b_std s.t. p' * primal_A_std <= primal_c_std, p >= 0
    dual_c = primal_b_std  # Coefficients for Maximize objective
    dual_A = primal_A_std.T # Transpose A
    dual_b = primal_c_std  # RHS for dual constraints (<= type)

    print("\n   Dual Problem Formulation:")
    print(f"      Objective: Maximize p * [{', '.join(f'{bi:.2f}' for bi in dual_c)}]")
    print(f"      Subject to:")
    for j in range(dual_A.shape[0]): # Iterate through dual constraints
         print(f"         {' + '.join(f'{dual_A[j, i]:.2f}*p{i+1}' for i in range(num_dual_vars))} <= {dual_b[j]:.2f}")
    print(f"      p_i >= 0 for i=1..{num_dual_vars}")

    print("\n--- Step 2: Solve the Dual Problem using SciPy linprog ---")
    # linprog solves Min problems, so we Max p*b by Min -p*b
    # Constraints for linprog: A_ub @ x <= b_ub, A_eq @ x == b_eq
    # Our dual is Max p*b s.t. p*A <= c. Let p be x for linprog.
    # Maximize dual_c @ p => Minimize -dual_c @ p
    # Subject to: dual_A @ p <= dual_b
    #             p >= 0 (default bounds)

    c_linprog = -dual_c
    A_ub_linprog = dual_A
    b_ub_linprog = dual_b

    # Use method='highs' which is the default and generally robust
    # Options can be added for more control if needed
    try:
        result_dual = linprog(c_linprog, A_ub=A_ub_linprog, b_ub=b_ub_linprog, bounds=[(0, None)] * num_dual_vars, method='highs') # Using HiGHS solver
    except ValueError as e:
         # Sometimes specific solvers are not available or fail
         print(f"   SciPy linprog(method='highs') failed: {e}. Trying 'simplex'.")
         try:
            result_dual = linprog(c_linprog, A_ub=A_ub_linprog, b_ub=b_ub_linprog, bounds=[(0, None)] * num_dual_vars, method='simplex')
         except Exception as e_simplex:
             return 1, f"SciPy linprog failed for dual problem with both 'highs' and 'simplex': {e_simplex}", None, None, None


    if not result_dual.success:
        # Check status for specific reasons
        if result_dual.status == 2: # Infeasible
             msg = "Dual problem is infeasible. Primal problem is unbounded (or infeasible)."
        elif result_dual.status == 3: # Unbounded
             msg = "Dual problem is unbounded. Primal problem is infeasible."
        else:
             msg = f"Failed to solve the dual problem. Status: {result_dual.status} - {result_dual.message}"
        return result_dual.status, msg, None, None, None

    # Optimal dual solution found
    optimal_dual_p = result_dual.x
    optimal_dual_obj = -result_dual.fun # Negate back to get Max value

    dual_solution_dict = {f'p{i+1}': optimal_dual_p[i] for i in range(num_dual_vars)}

    print("\n   Optimal Dual Solution Found:")
    print(f"      Dual Variables (p*):")
    for i in range(num_dual_vars):
        print(f"         p{i+1} = {optimal_dual_p[i]:.6f}")
    print(f"      Optimal Dual Objective Value (Max p*b): {optimal_dual_obj:.6f}")

    print("\n--- Step 3: Check Strong Duality ---")
    # The optimal objective value of the dual should equal the optimal objective
    # value of the primal (after adjusting for Min/Max conversion).
    expected_primal_obj = optimal_dual_obj * objective_sign_flip
    print(f"   Strong duality implies the optimal primal objective value should be: {expected_primal_obj:.6f}")

    print("\n--- Step 4 & 5: Use Complementary Slackness to find Primal Variables ---")

    # Calculate Dual Slacks: dual_slack = dual_b - dual_A @ optimal_dual_p
    # dual_b is primal_c_std
    # dual_A is primal_A_std.T
    # dual_slack_j = primal_c_std_j - (optimal_dual_p @ primal_A_std)_j
    dual_slacks = dual_b - dual_A @ optimal_dual_p
    print("   Calculating Dual Slacks (c'_j - p* A'_j):")
    for j in range(num_primal_vars):
        print(f"      Dual Slack for primal var x{j+1}: {dual_slacks[j]:.6f}")


    # Identify conditions from Complementary Slackness (CS)
    # 1. Dual Slackness: If dual_slack_j > TOLERANCE, then primal x*_j = 0
    # 2. Primal Slackness: If optimal_dual_p_i > TOLERANCE, then i-th standardized primal constraint is binding
    #    (primal_A_std[i] @ x* = primal_b_std[i])

    binding_constraints_indices = []
    zero_primal_vars_indices = []

    print("\n   Applying Complementary Slackness Conditions:")
    # Dual Slackness
    print("      From Dual Slackness (if c'_j - p* A'_j > 0, then x*_j = 0):")
    for j in range(num_primal_vars):
        if dual_slacks[j] > TOLERANCE:
            print(f"         Dual Slack for x{j+1} is {dual_slacks[j]:.4f} > 0 => x{j+1}* = 0")
            zero_primal_vars_indices.append(j)
        else:
             print(f"         Dual Slack for x{j+1} is {dual_slacks[j]:.4f} approx 0 => x{j+1}* may be non-zero")


    # Primal Slackness
    print("      From Primal Slackness (if p*_i > 0, then primal constraint i is binding):")
    for i in range(num_dual_vars):
        if optimal_dual_p[i] > TOLERANCE:
            print(f"         p*{i+1} = {optimal_dual_p[i]:.4f} > 0 => Primal constraint {i+1} (standardized) is binding.")
            binding_constraints_indices.append(i)
        else:
            print(f"         p*{i+1} = {optimal_dual_p[i]:.4f} approx 0 => Primal constraint {i+1} (standardized) may be non-binding.")

    # Construct system of equations for non-zero primal variables
    # Equations come from binding primal constraints.
    # Variables are x*_j where j is NOT in zero_primal_vars_indices.

    active_primal_vars_indices = [j for j in range(num_primal_vars) if j not in zero_primal_vars_indices]
    num_active_primal_vars = len(active_primal_vars_indices)

    print(f"\n   Identifying system for active primal variables ({[f'x{j+1}' for j in active_primal_vars_indices]}):")

    if num_active_primal_vars == 0:
        # All primal vars are zero
        primal_x_star = np.zeros(num_primal_vars)
        print("      All primal variables determined to be 0 by dual slackness.")
    elif len(binding_constraints_indices) < num_active_primal_vars:
         print(f"      Warning: Number of binding constraints ({len(binding_constraints_indices)}) identified is less than the number of potentially non-zero primal variables ({num_active_primal_vars}).")
         print("      Complementary slackness alone might not be sufficient, or there might be degeneracy/multiple solutions.")
         print("      Attempting to solve using available binding constraints, but result might be unreliable.")
         # Pad with zero rows if necessary, or indicate underspecified system. For now, proceed cautiously.
         matrix_A_sys = primal_A_std[binding_constraints_indices][:, active_primal_vars_indices]
         vector_b_sys = primal_b_std[binding_constraints_indices]

    else:
        # We have at least as many binding constraints as active variables.
        # Select num_active_primal_vars binding constraints to form a square system (if possible).
        # If more binding constraints exist, they should be consistent.
        # We take the first 'num_active_primal_vars' binding constraints.
        if len(binding_constraints_indices) > num_active_primal_vars:
             print(f"      More binding constraints ({len(binding_constraints_indices)}) than active variables ({num_active_primal_vars}). Using the first {num_active_primal_vars}.")

        matrix_A_sys = primal_A_std[binding_constraints_indices[:num_active_primal_vars]][:, active_primal_vars_indices]
        vector_b_sys = primal_b_std[binding_constraints_indices[:num_active_primal_vars]]

        print("      System Ax = b to solve:")
        for r in range(matrix_A_sys.shape[0]):
            print(f"         {' + '.join(f'{matrix_A_sys[r, c]:.2f}*x{active_primal_vars_indices[c]+1}' for c in range(num_active_primal_vars))} = {vector_b_sys[r]:.2f}")


    # Solve the system if possible
    if num_active_primal_vars > 0:
        try:
            # Use numpy.linalg.solve for square systems, lstsq for potentially non-square
            if matrix_A_sys.shape[0] == matrix_A_sys.shape[1]:
                 solved_active_vars = np.linalg.solve(matrix_A_sys, vector_b_sys)
            elif matrix_A_sys.shape[0] > matrix_A_sys.shape[1]: # Overdetermined
                 print("      System is overdetermined. Using least squares solution.")
                 solved_active_vars, residuals, rank, s = np.linalg.lstsq(matrix_A_sys, vector_b_sys, rcond=None)
                 # Check if residuals are close to zero for consistency
                 if residuals and np.sum(residuals**2) > TOLERANCE * len(vector_b_sys):
                       print(f"      Warning: Least squares solution has significant residuals ({np.sqrt(np.sum(residuals**2)):.4f}), CS conditions might be inconsistent?")
            else: # Underdetermined
                 # Cannot uniquely solve. This shouldn't happen if dual was optimal and non-degenerate.
                 # Could use lstsq which gives one possible solution (minimum norm).
                 print("      System is underdetermined. Using least squares (minimum norm) solution.")
                 solved_active_vars, residuals, rank, s = np.linalg.lstsq(matrix_A_sys, vector_b_sys, rcond=None)


            # Assign solved values back to the full primal_x_star vector
            primal_x_star = np.zeros(num_primal_vars)
            for i, active_idx in enumerate(active_primal_vars_indices):
                primal_x_star[active_idx] = solved_active_vars[i]

            print("\n      Solved values for active primal variables:")
            for i, active_idx in enumerate(active_primal_vars_indices):
                print(f"         x{active_idx+1}* = {solved_active_vars[i]:.6f}")

        except np.linalg.LinAlgError:
            print("      Error: Could not solve the system of equations derived from binding constraints (matrix may be singular).")
            # Attempt to use linprog on the original primal as a fallback/check
            print("      Attempting to solve primal directly with linprog as a fallback...")
            primal_fallback_status, _, primal_fallback_sol, _, primal_fallback_obj = solve_primal_directly(
                original_obj_type, c, A, relations, b)
            if primal_fallback_status == 0:
                 print("      Fallback solution found.")
                 return 0, "Solved primal using fallback direct method after CS failure", primal_fallback_sol, dual_solution_dict, primal_fallback_obj
            else:
                 return 1, "Failed to solve system from CS, and fallback primal solve also failed.", None, dual_solution_dict, None


    # Assemble final primal solution dictionary
    primal_solution_dict = {f'x{j+1}': primal_x_star[j] for j in range(num_primal_vars)}

    # --- Step 6: Verify Primal Feasibility and Objective Value ---
    print("\n--- Step 6: Verify Primal Solution ---")
    feasible = True
    print("   Checking primal constraints:")
    for i in range(num_primal_constraints):
        lhs_val = primal_A[i] @ primal_x_star
        rhs_val = primal_b[i]
        rel = primal_relations[i]
        constraint_met = False
        if rel == '<=':
            constraint_met = lhs_val <= rhs_val + TOLERANCE
        elif rel == '>=':
            constraint_met = lhs_val >= rhs_val - TOLERANCE
        elif rel == '=':
            constraint_met = abs(lhs_val - rhs_val) < TOLERANCE

        status_str = "Satisfied" if constraint_met else "VIOLATED"
        print(f"      Constraint {i+1}: {lhs_val:.4f} {rel} {rhs_val:.4f} -> {status_str}")
        if not constraint_met:
            feasible = False

    print("   Checking non-negativity (x >= 0):")
    non_negative = np.all(primal_x_star >= -TOLERANCE)
    print(f"      All x_j >= 0: {non_negative}")
    if not non_negative:
        feasible = False
        print(f"Violating variables: {[f'x{j+1}={primal_x_star[j]:.4f}' for j in range(len(primal_x_star)) if primal_x_star[j] < -TOLERANCE]}")

    final_primal_obj = primal_c @ primal_x_star # Using original primal c
    print(f"\n   Calculated Primal Objective Value: {final_primal_obj:.6f}")
    print(f"   Expected Primal Objective Value (from dual): {expected_primal_obj:.6f}")

    if abs(final_primal_obj - expected_primal_obj) > TOLERANCE * (1 + abs(expected_primal_obj)):
        print("   Warning: Calculated primal objective value significantly differs from the dual objective value!")
        feasible = False # Consider this a failure if strong duality doesn't hold

    if feasible:
        print("\n--- Primal Solution Found Successfully via Dual ---")
        return 0, "Optimal solution found via dual.", primal_solution_dict, dual_solution_dict, final_primal_obj
    else:
        print("\n--- Failed to Find Feasible Primal Solution via Dual ---")
        print("   The derived primal solution violates constraints or non-negativity, or strong duality failed.")
        # You might want to return the possibly incorrect solution for inspection or None
        return 1, "Derived primal solution is infeasible or inconsistent.", primal_solution_dict, dual_solution_dict, final_primal_obj

