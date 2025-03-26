
import numpy as np
from scipy.optimize import linprog # Using SciPy for robust LP solving
import warnings

def solve_primal_directly(objective_type, c, A, relations, b):
    """Helper to solve primal directly using linprog for comparison/fallback"""
    primal_c = np.array(c, dtype=float)
    primal_A = np.array(A, dtype=float)
    primal_b = np.array(b, dtype=float)
    num_vars = len(c)

    A_ub, b_ub, A_eq, b_eq = [], [], [], []

    sign_flip = 1.0
    if objective_type.lower() == 'max':
        primal_c = -primal_c
        sign_flip = -1.0

    for i in range(len(relations)):
        if relations[i] == '<=':
            A_ub.append(primal_A[i])
            b_ub.append(primal_b[i])
        elif relations[i] == '>=':
            A_ub.append(-primal_A[i]) # Convert >= to <= for linprog A_ub
            b_ub.append(-primal_b[i])
        elif relations[i] == '=':
            A_eq.append(primal_A[i])
            b_eq.append(primal_b[i])

    # Convert lists to arrays, handling empty cases
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None

    try:
         result_primal = linprog(primal_c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * num_vars, method='highs') # 'highs' is default and robust

         if result_primal.success:
             primal_sol_vals = result_primal.x
             primal_obj_val = result_primal.fun * sign_flip # Adjust obj value back if Max
             primal_sol_dict = {f'x{j+1}': primal_sol_vals[j] for j in range(num_vars)}
             return 0, "Solved directly", primal_sol_dict, None, primal_obj_val
         else:
             return result_primal.status, f"Direct solve failed: {result_primal.message}", None, None, None
    except Exception as e:
         return -1, f"Direct solve exception: {e}", None, None, None

