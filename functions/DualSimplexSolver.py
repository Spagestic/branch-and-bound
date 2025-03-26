import numpy as np
import sys
from functions.solve_lp_via_dual import solve_lp_via_dual
from functions.solve_primal_directly import solve_primal_directly

TOLERANCE = 1e-9

class DualSimplexSolver:
    """
    Solves a Linear Programming problem using the Dual Simplex Method.

    Assumes the problem is provided in the form:
    Maximize/Minimize c^T * x
    Subject to:
        A * x <= / >= / = b
        x >= 0

    The algorithm works best when the initial tableau (after converting all
    constraints to <=) is dual feasible (objective row coefficients >= 0 for Max)
    but primal infeasible (some RHS values are negative).
    """

    def __init__(self, objective_type, c, A, relations, b):
        """
        Initializes the solver.

        Args:
            objective_type (str): 'max' or 'min'.
            c (list or np.array): Coefficients of the objective function.
            A (list of lists or np.array): Coefficients of the constraints LHS.
            relations (list): List of strings ('<=', '>=', '=') for each constraint.
            b (list or np.array): RHS values of the constraints.
        """
        self.objective_type = objective_type.lower()
        self.original_c = np.array(c, dtype=float)
        self.original_A = np.array(A, dtype=float)
        self.original_relations = relations
        self.original_b = np.array(b, dtype=float)

        self.num_original_vars = len(c)
        self.num_constraints = len(b)

        self.tableau = None
        self.basic_vars = [] # Indices of basic variables (column index)
        self.var_names = []  # Names like 'x1', 's1', etc.
        self.is_minimized_problem = False # Flag to adjust final Z

        self._preprocess()

    def _preprocess(self):
        """
        Converts the problem to the standard form for Dual Simplex:
        - Maximization objective
        - All constraints are <=
        - Adds slack variables
        - Builds the initial tableau
        """
        # --- 1. Handle Objective Function ---
        if self.objective_type == 'min':
            self.is_minimized_problem = True
            current_c = -self.original_c
        else:
            current_c = self.original_c

        # --- 2. Handle Constraints and Slack Variables ---
        num_slacks_added = 0
        processed_A = []
        processed_b = []
        self.basic_vars = [] # Will store column indices of basic vars

        # Create variable names
        self.var_names = [f'x{i+1}' for i in range(self.num_original_vars)]
        slack_var_names = []

        for i in range(self.num_constraints):
            A_row = self.original_A[i]
            b_val = self.original_b[i]
            relation = self.original_relations[i]

            if relation == '>=':
                # Multiply by -1 to convert to <=
                processed_A.append(-A_row)
                processed_b.append(-b_val)
            elif relation == '=':
                # Convert Ax = b into Ax <= b and Ax >= b
                # First: Ax <= b
                processed_A.append(A_row)
                processed_b.append(b_val)
                # Second: Ax >= b --> -Ax <= -b
                processed_A.append(-A_row)
                processed_b.append(-b_val)
            elif relation == '<=':
                processed_A.append(A_row)
                processed_b.append(b_val)
            else:
                raise ValueError(f"Invalid relation symbol: {relation}")

        # Update number of effective constraints after handling '='
        effective_num_constraints = len(processed_b)

        # Add slack variables for all processed constraints (which are now all <=)
        num_slack_vars = effective_num_constraints
        final_A = np.zeros((effective_num_constraints, self.num_original_vars + num_slack_vars))
        final_b = np.array(processed_b, dtype=float)

        # Populate original variable coefficients
        final_A[:, :self.num_original_vars] = np.array(processed_A, dtype=float)

        # Add slack variable identity matrix part and names
        for i in range(effective_num_constraints):
            slack_col_index = self.num_original_vars + i
            final_A[i, slack_col_index] = 1
            slack_var_names.append(f's{i+1}')
            self.basic_vars.append(slack_col_index) # Initially, slacks are basic

        self.var_names.extend(slack_var_names)

        # --- 3. Build the Tableau ---
        num_total_vars = self.num_original_vars + num_slack_vars
        # Rows: 1 for objective + number of constraints
        # Cols: 1 for Z + number of total vars + 1 for RHS
        self.tableau = np.zeros((effective_num_constraints + 1, num_total_vars + 2))

        # Row 0 (Objective Z): [1, -c, 0_slacks, 0_rhs]
        self.tableau[0, 0] = 1 # Z coefficient
        self.tableau[0, 1:self.num_original_vars + 1] = -current_c
        # Slack coefficients in objective are 0 initially
        # RHS of objective row is 0 initially

        # Rows 1 to m (Constraints): [0, A_final, b_final]
        self.tableau[1:, 1:num_total_vars + 1] = final_A
        self.tableau[1:, -1] = final_b

        # Ensure the initial objective row is dual feasible (non-negative coeffs for Max)
        # We rely on the user providing a problem where this holds after conversion.
        if np.any(self.tableau[0, 1:-1] < -TOLERANCE):
             print("\nWarning: Initial tableau is not dual feasible (objective row has negative coefficients).")
             print("The standard Dual Simplex method might not apply directly or may require Phase I.")
             # For this implementation, we'll proceed, but it might fail if assumption is violated.


    def _print_tableau(self, iteration):
        """Prints the current state of the tableau."""
        print(f"\n--- Iteration {iteration} ---")
        header = ["BV"] + ["Z"] + self.var_names + ["RHS"]
        print(" ".join(f"{h:>8}" for h in header))
        print("-" * (len(header) * 9))

        basic_var_map = {idx: name for idx, name in enumerate(self.var_names)}
        row_basic_vars = ["Z"] + [basic_var_map.get(bv_idx, f'col{bv_idx}') for bv_idx in self.basic_vars]

        for i, row_bv_name in enumerate(row_basic_vars):
             row_str = [f"{row_bv_name:>8}"]
             row_str.extend([f"{val: >8.3f}" for val in self.tableau[i]])
             print(" ".join(row_str))
        print("-" * (len(header) * 9))


    def _find_pivot_row(self):
        """Finds the index of the leaving variable (pivot row)."""
        rhs_values = self.tableau[1:, -1]
        # Find the index of the most negative RHS value (among constraints)
        if np.all(rhs_values >= -TOLERANCE):
            return -1 # All RHS non-negative, current solution is feasible (and optimal)

        pivot_row_index = np.argmin(rhs_values) + 1 # +1 because we skip obj row 0
        # Check if the minimum value is actually negative
        if self.tableau[pivot_row_index, -1] >= -TOLERANCE:
             return -1 # Should not happen if np.all check passed, but safety check

        print(f"\nStep: Select Pivot Row (Leaving Variable)")
        print(f"   RHS values (b): {rhs_values}")
        leaving_var_idx = self.basic_vars[pivot_row_index - 1]
        leaving_var_name = self.var_names[leaving_var_idx]
        print(f"   Most negative RHS is {self.tableau[pivot_row_index, -1]:.3f} in Row {pivot_row_index} (Basic Var: {leaving_var_name}).")
        print(f"   Leaving Variable: {leaving_var_name} (Row {pivot_row_index})")
        return pivot_row_index

    def _find_pivot_col(self, pivot_row_index):
        """Finds the index of the entering variable (pivot column)."""
        pivot_row = self.tableau[pivot_row_index, 1:-1] # Exclude Z and RHS cols
        objective_row = self.tableau[0, 1:-1]           # Exclude Z and RHS cols

        ratios = {}
        min_ratio = float('inf')
        pivot_col_index = -1

        print(f"\nStep: Select Pivot Column (Entering Variable) using Ratio Test")
        print(f"   Pivot Row (Row {pivot_row_index}) coefficients (excluding Z, RHS): {pivot_row}")
        print(f"   Objective Row coefficients (excluding Z, RHS): {objective_row}")
        print(f"   Calculating ratios = ObjCoeff / abs(PivotRowCoeff) for PivotRowCoeff < 0:")

        found_negative_coeff = False
        for j, coeff in enumerate(pivot_row):
            col_var_index = j # This is the index within the var_names list
            col_tableau_index = j + 1 # This is the index in the full tableau row

            if coeff < -TOLERANCE: # Must be strictly negative
                found_negative_coeff = True
                obj_coeff = objective_row[j]
                # Ratio calculation: obj_coeff / abs(coeff) or obj_coeff / -coeff
                ratio = obj_coeff / (-coeff)
                ratios[col_var_index] = ratio
                print(f"      Var {self.var_names[col_var_index]} (Col {col_tableau_index}): Coeff={coeff:.3f}, ObjCoeff={obj_coeff:.3f}, Ratio = {obj_coeff:.3f} / {-coeff:.3f} = {ratio:.3f}")

                # Update minimum ratio
                if ratio < min_ratio:
                    min_ratio = ratio
                    pivot_col_index = col_tableau_index # Store the tableau column index

        if not found_negative_coeff:
            print("   No negative coefficients found in the pivot row.")
            return -1 # Indicates primal infeasibility (dual unboundedness)

        # Handle potential ties in minimum ratio (choose smallest column index - Bland's rule simplified)
        min_ratio_vars = [idx for idx, r in ratios.items() if abs(r - min_ratio) < TOLERANCE]
        if len(min_ratio_vars) > 1:
            print(f"   Tie detected for minimum ratio ({min_ratio:.3f}) among variables: {[self.var_names[idx] for idx in min_ratio_vars]}.")
            # Apply Bland's rule: choose the variable with the smallest index
            pivot_col_index = min(min_ratio_vars) + 1 # +1 for tableau index
            print(f"   Applying Bland's rule: Choosing variable with smallest index: {self.var_names[pivot_col_index - 1]}.")
        elif pivot_col_index != -1:
             entering_var_name = self.var_names[pivot_col_index - 1] # -1 to get var_name index
             print(f"   Minimum ratio is {min_ratio:.3f} for variable {entering_var_name} (Column {pivot_col_index}).")
             print(f"   Entering Variable: {entering_var_name} (Column {pivot_col_index})")
        else:
             # This case should technically not be reached if found_negative_coeff was true
             print("Error in ratio calculation or tie-breaking.")
             return -2 # Error indicator

        return pivot_col_index


    def _pivot(self, pivot_row_index, pivot_col_index):
        """Performs the pivot operation."""
        pivot_element = self.tableau[pivot_row_index, pivot_col_index]

        print(f"\nStep: Pivot Operation")
        print(f"   Pivot Element: {pivot_element:.3f} at (Row {pivot_row_index}, Col {pivot_col_index})")

        if abs(pivot_element) < TOLERANCE:
            print("Error: Pivot element is zero. Cannot proceed.")
            # This might indicate an issue with the problem formulation or numerical instability.
            raise ZeroDivisionError("Pivot element is too close to zero.")

        # 1. Normalize the pivot row
        print(f"   Normalizing Pivot Row {pivot_row_index} by dividing by {pivot_element:.3f}")
        self.tableau[pivot_row_index, :] /= pivot_element

        # 2. Eliminate other entries in the pivot column
        print(f"   Eliminating other entries in Pivot Column {pivot_col_index}:")
        for i in range(self.tableau.shape[0]):
            if i != pivot_row_index:
                factor = self.tableau[i, pivot_col_index]
                if abs(factor) > TOLERANCE: # Only perform if factor is non-zero
                    print(f"      Row {i} = Row {i} - ({factor:.3f}) * (New Row {pivot_row_index})")
                    self.tableau[i, :] -= factor * self.tableau[pivot_row_index, :]

        # 3. Update basic variables list
        # The variable corresponding to pivot_col_index becomes basic for pivot_row_index
        old_basic_var_index = self.basic_vars[pivot_row_index - 1]
        new_basic_var_index = pivot_col_index - 1 # Convert tableau col index to var_names index
        self.basic_vars[pivot_row_index - 1] = new_basic_var_index
        print(f"   Updating Basic Variables: {self.var_names[new_basic_var_index]} replaces {self.var_names[old_basic_var_index]} in the basis for Row {pivot_row_index}.")


    def solve(self, use_fallbacks=True):
        """
        Executes the Dual Simplex algorithm.
        
        Args:
            use_fallbacks (bool): If True, will attempt to use alternative solvers
                                  when the dual simplex method encounters issues
                                  
        Returns:
            tuple: (tableau, basic_vars) if successful using dual simplex,
                   or a dictionary of results if fallback solvers were used
        """
        print("--- Starting Dual Simplex Method ---")
        if self.tableau is None:
            print("Error: Tableau not initialized.")
            return None

        iteration = 0
        self._print_tableau(iteration)

        while iteration < 100: # Safety break for too many iterations
            iteration += 1

            # 1. Check for Optimality (Primal Feasibility)
            pivot_row_index = self._find_pivot_row()
            if pivot_row_index == -1:
                print("\n--- Optimal Solution Found ---")
                print("   All RHS values are non-negative.")
                self._print_results()
                return self.tableau, self.basic_vars

            # 2. Select Entering Variable (Pivot Column)
            pivot_col_index = self._find_pivot_col(pivot_row_index)

            # 3. Check for Primal Infeasibility (Dual Unboundedness)
            if pivot_col_index == -1:
                print("\n--- Primal Problem Infeasible ---")
                print(f"   All coefficients in Pivot Row {pivot_row_index} are non-negative, but RHS is negative.")
                print("   The dual problem is unbounded, implying the primal problem has no feasible solution.")
                
                if use_fallbacks:
                    return self._try_fallback_solvers("primal_infeasible")
                return None, None # Indicate infeasibility
                
            elif pivot_col_index == -2:
                 # Error during pivot column selection
                 print("\n--- Error during pivot column selection ---")
                 
                 if use_fallbacks:
                    return self._try_fallback_solvers("pivot_error")
                 return None, None

            # 4. Perform Pivot Operation
            try:
                self._pivot(pivot_row_index, pivot_col_index)
            except ZeroDivisionError as e:
                print(f"\n--- Error during pivot operation: {e} ---")
                
                if use_fallbacks:
                    return self._try_fallback_solvers("numerical_instability")
                return None, None

            # Print the tableau after pivoting
            self._print_tableau(iteration)

        print("\n--- Maximum Iterations Reached ---")
        print("   The algorithm did not converge within the iteration limit.")
        print("   This might indicate cycling or a very large problem.")
        
        if use_fallbacks:
            return self._try_fallback_solvers("iteration_limit")
        return None, None # Indicate non-convergence

    def _try_fallback_solvers(self, error_type):
        """
        Tries alternative solvers when the dual simplex method fails.
        
        Args:
            error_type (str): Type of error encountered in the dual simplex method
            
        Returns:
            dict: Results from fallback solvers
        """
        print(f"\n--- Using Fallback Solvers due to '{error_type}' ---")
        
        results = {
            "error_type": error_type,
            "dual_simplex_result": None,
            "dual_approach_result": None,
            "direct_solver_result": None
        }
        
        # First try using solve_lp_via_dual (which uses complementary slackness)
        print("\n=== Attempting to solve via Dual Approach with Complementary Slackness ===")
        status, message, primal_sol, dual_sol, obj_val = solve_lp_via_dual(
            self.objective_type, 
            self.original_c, 
            self.original_A, 
            self.original_relations, 
            self.original_b
        )
        
        results["dual_approach_result"] = {
            "status": status,
            "message": message,
            "primal_solution": primal_sol,
            "dual_solution": dual_sol,
            "objective_value": obj_val
        }
        
        print(f"Dual Approach Result: {message}")
        if status == 0 and primal_sol:
            print(f"Objective Value: {obj_val}")
            return results
        
        # If that fails, try direct method (most robust)
        print("\n=== Attempting direct solution using SciPy's linprog solver ===")
        status, message, primal_sol, _, obj_val = solve_primal_directly(
            self.objective_type, 
            self.original_c, 
            self.original_A, 
            self.original_relations, 
            self.original_b
        )
        
        results["direct_solver_result"] = {
            "status": status,
            "message": message,
            "primal_solution": primal_sol,
            "objective_value": obj_val
        }
        
        print(f"Direct Solver Result: {message}")
        if status == 0 and primal_sol:
            print(f"Objective Value: {obj_val}")
        
        return results

    def _print_results(self):
        """Prints the final solution."""
        print("\n--- Final Solution ---")
        self._print_tableau("Final")

        # Objective Value
        final_obj_value = self.tableau[0, -1]
        if self.is_minimized_problem:
            final_obj_value = -final_obj_value # Correct for Min Z = -Max(-Z)
            print(f"Optimal Objective Value (Min Z): {final_obj_value:.6f}")
        else:
            print(f"Optimal Objective Value (Max Z): {final_obj_value:.6f}")

        # Variable Values
        solution = {}
        num_total_vars = len(self.var_names)
        final_solution_vector = np.zeros(num_total_vars)

        for i, basis_col_idx in enumerate(self.basic_vars):
            # basis_col_idx is the index in the var_names list
            # The corresponding tableau row is i + 1
            final_solution_vector[basis_col_idx] = self.tableau[i + 1, -1]

        print("Optimal Variable Values:")
        for i in range(self.num_original_vars):
             var_name = self.var_names[i]
             value = final_solution_vector[i]
             print(f"   {var_name}: {value:.6f}")
             solution[var_name] = value

        # Optionally print slack variable values
        print("Slack/Surplus Variable Values:")
        for i in range(self.num_original_vars, num_total_vars):
             var_name = self.var_names[i]
             value = final_solution_vector[i]
             # Only print non-zero slacks for brevity, or all if needed
             if abs(value) > TOLERANCE:
                 print(f"   {var_name}: {value:.6f}")

