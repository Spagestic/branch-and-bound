def get_user_input():
    """Gets the LP problem details from the user."""
    # (Identical to the input function in the previous Dual Simplex example)
    print("Enter the Linear Programming Problem:")
    while True:
        obj_type = input("Is it a 'max' or 'min' problem? ").strip().lower()
        if obj_type in ['max', 'min']:
            break
        print("Invalid input. Please enter 'max' or 'min'.")
    c_str = input("Enter objective function coefficients (space-separated): ").strip()
    c = list(map(float, c_str.split()))
    num_vars = len(c)
    A = []
    relations = []
    b = []
    i = 1
    print(f"Enter constraints (one per line). There are {num_vars} variables (x1, x2,...).")
    print("Format: coeff1 coeff2 ... relation rhs (e.g., '3 1 >= 3')")
    print("Type 'done' when finished.")
    while True:
        line = input(f"Constraint {i}: ").strip()
        if line.lower() == 'done':
            if i == 1:
                print("Error: At least one constraint is required.")
                continue
            break
        parts = line.split()
        if len(parts) != num_vars + 2:
            print(f"Error: Expected {num_vars} coefficients, 1 relation, and 1 RHS value.")
            continue
        try:
            coeffs = list(map(float, parts[:num_vars]))
            relation = parts[num_vars]
            rhs = float(parts[num_vars + 1])
            if relation not in ['<=', '>=', '=']:
                print("Error: Invalid relation symbol. Use '<=', '>=', or '='.")
                continue
            A.append(coeffs)
            relations.append(relation)
            b.append(rhs)
            i += 1
        except ValueError:
            print("Error: Invalid number format for coefficients or RHS.")
            continue
    return obj_type, c, A, relations, b

