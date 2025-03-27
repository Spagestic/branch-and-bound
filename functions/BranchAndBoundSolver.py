import cvxpy as cp
import numpy as np
import math
from queue import PriorityQueue
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.optimize import linprog


class BranchAndBoundSolver:
    def __init__(self, c, A, b, integer_vars=None, binary_vars=None, maximize=True):
        """
        Initialize the Branch and Bound solver
        
        Parameters:
        - c: Objective coefficients (for max c'x)
        - A, b: Constraints Ax <= b
        - integer_vars: Indices of variables that must be integers
        - binary_vars: Indices of variables that must be binary (0 or 1)
        - maximize: True for maximization, False for minimization
        """
        self.c = c
        self.A = A
        self.b = b
        self.n = len(c)
        
        # Process binary and integer variables
        self.binary_vars = [] if binary_vars is None else binary_vars
        
        # If integer_vars not specified, assume all non-binary variables are integers
        if integer_vars is None:
            self.integer_vars = list(range(self.n))
        else:
            self.integer_vars = integer_vars.copy()
        
        # Add binary variables to integer variables list if they're not already there
        for idx in self.binary_vars:
            if idx not in self.integer_vars:
                self.integer_vars.append(idx)
        
        # Best solution found so far
        self.best_solution = None
        self.best_objective = float('-inf') if maximize else float('inf')
        self.maximize = maximize
        
        # Track nodes explored
        self.nodes_explored = 0
        
        # Graph for visualization
        self.graph = nx.DiGraph()
        self.node_id = 0
        
        # For tabular display of steps
        self.steps_table = []
        
        # Set of active nodes
        self.active_nodes = set()
    
    def is_integer_feasible(self, x):
        """Check if the solution satisfies integer constraints"""
        if x is None:
            return False
        
        for idx in self.integer_vars:
            if abs(round(x[idx]) - x[idx]) > 1e-6:
                return False
        return True
    
    def get_branching_variable(self, x):
        """Select most fractional variable to branch on"""
        max_fractional = -1
        branching_var = -1
        
        for idx in self.integer_vars:
            fractional_part = abs(x[idx] - round(x[idx]))
            if fractional_part > max_fractional and fractional_part > 1e-6:
                max_fractional = fractional_part
                branching_var = idx
        
        return branching_var
    
    def solve_relaxation(self, lower_bounds, upper_bounds):
        """Solve the continuous relaxation with given bounds"""
        x = cp.Variable(self.n)
        
        # Set the objective - maximize c'x or minimize -c'x
        if self.maximize:
            objective = cp.Maximize(self.c @ x)
        else:
            objective = cp.Minimize(self.c @ x)
        
        # Basic constraints Ax <= b
        constraints = [self.A @ x <= self.b]
        
        # Add bounds
        for i in range(self.n):
            if lower_bounds[i] is not None:
                constraints.append(x[i] >= lower_bounds[i])
            if upper_bounds[i] is not None:
                constraints.append(x[i] <= upper_bounds[i])
        
        prob = cp.Problem(objective, constraints)
        
        try:
            objective_value = prob.solve()
            return x.value, objective_value
        except:
            return None, float('-inf') if self.maximize else float('inf')
    
    def add_node_to_graph(self, node_name, objective_value, x_value, parent=None, branch_var=None, branch_cond=None):
        """Add a node to the branch and bound graph"""
        self.graph.add_node(node_name, obj=objective_value, x=x_value, 
                        branch_var=branch_var, branch_cond=branch_cond)
        
        if parent is not None:
            # Use branch_var + 1 to show 1-indexed variables in the display
            label = f"x_{branch_var + 1} {branch_cond}"
            self.graph.add_edge(parent, node_name, label=label)
        
        return node_name
    
    def visualize_graph(self):
        """Visualize the branch and bound graph"""
        fig = plt.figure(figsize=(20, 8))
        pos = nx.spring_layout(self.graph)  # Use spring layout instead of graphviz

        # Node labels: Node name, Objective value and solution
        labels = {}
        for node, data in self.graph.nodes(data=True):
            if data.get('x') is not None:
                x_str = ', '.join([f"{x:.2f}" for x in data['x']])
                labels[node] = f"{node}\n({data['obj']:.2f}, ({x_str}))"
            else:
                labels[node] = f"{node}\nInfeasible"
        
        # Edge labels: Branching conditions
        edge_labels = nx.get_edge_attributes(self.graph, 'label')
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=2000, node_color='skyblue')
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, width=1.5, arrowsize=20, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=10, font_family='sans-serif')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=10, font_family='sans-serif')
        
        plt.title("Branch and Bound Tree", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        return fig  # Return the figure instead of showing it

    
    def display_steps_table(self):
        """Display the steps in tabular format"""
        headers = ["Node", "z", "x", "z*", "x*", "UB", "LB", "Z at end of stage"]
        print(tabulate(self.steps_table, headers=headers, tablefmt="grid"))
    
    def solve(self, verbose=True):
        """Solve the problem using branch and bound"""
        # Initialize bounds
        lower_bounds = [0] * self.n
        upper_bounds = [None] * self.n  # None means unbounded
        
        # Set upper bounds for binary variables
        for idx in self.binary_vars:
            upper_bounds[idx] = 1
        
        # Create a priority queue for nodes (max heap for maximization, min heap for minimization)
        # We use negative values for maximization to simulate max heap with Python's min heap
        node_queue = PriorityQueue()
        
        # Solve the root relaxation
        print("Step 1: Solving root relaxation (continuous problem)")
        x_root, obj_root = self.solve_relaxation(lower_bounds, upper_bounds)
        
        if x_root is None:
            print("Root problem infeasible")
            return None, float('-inf') if self.maximize else float('inf')
        
        # Add root node to the graph
        root_node = "S0"
        self.add_node_to_graph(root_node, obj_root, x_root)
        
        print(f"Root relaxation objective: {obj_root:.6f}")
        print(f"Root solution: {x_root}")
        
        # Initial upper bound is the root objective
        upper_bound = obj_root
        
        # Check if the root solution is already integer-feasible
        if self.is_integer_feasible(x_root):
            print("Root solution is integer-feasible! No need for branching.")
            self.best_solution = x_root
            self.best_objective = obj_root
            
            # Add to steps table
            active_nodes_str = "∅" if not self.active_nodes else "{" + ", ".join(self.active_nodes) + "}"
            self.steps_table.append([
                root_node, f"{obj_root:.2f}", f"({', '.join([f'{x:.2f}' for x in x_root])})", 
                f"{self.best_objective:.2f}", f"({', '.join([f'{x:.2f}' for x in self.best_solution])})",
                f"{upper_bound:.2f}", f"{self.best_objective:.2f}", active_nodes_str
            ])
            
            self.display_steps_table()
            self.visualize_graph()
            return x_root, obj_root
        
        # Add root node to the queue and active nodes set
        priority = -obj_root if self.maximize else obj_root
        node_queue.put((priority, self.nodes_explored, root_node, lower_bounds.copy(), upper_bounds.copy()))
        self.active_nodes.add(root_node)
        
        # Add entry to steps table for root node
        active_nodes_str = "{" + ", ".join(self.active_nodes) + "}"
        lb_str = "-" if self.best_objective == float('-inf') else f"{self.best_objective:.2f}"
        x_star_str = "-" if self.best_solution is None else f"({', '.join([f'{x:.2f}' for x in self.best_solution])})"
        
        self.steps_table.append([
            root_node, f"{obj_root:.2f}", f"({', '.join([f'{x:.2f}' for x in x_root])})", 
            lb_str, x_star_str, f"{upper_bound:.2f}", lb_str, active_nodes_str
        ])
        
        print("\nStarting branch and bound process:")
        node_counter = 1
        
        while not node_queue.empty():
            # Get the node with the highest objective (for maximization)
            priority, _, node_name, node_lower_bounds, node_upper_bounds = node_queue.get()
            self.nodes_explored += 1
            
            print(f"\nStep {self.nodes_explored + 1}: Exploring node {node_name}")
            
            # Remove from active nodes
            self.active_nodes.remove(node_name)
            
            # Branch on most fractional variable
            branch_var = self.get_branching_variable(self.graph.nodes[node_name]['x'])
            branch_val = self.graph.nodes[node_name]['x'][branch_var]
            
            # For binary variables, always branch with x=0 and x=1
            if branch_var in self.binary_vars:
                floor_val = 0
                ceil_val = 1
                print(f"  Branching on binary variable x_{branch_var + 1} with value {branch_val:.6f}")
                print(f"  Creating two branches: x_{branch_var + 1} = 0 and x_{branch_var + 1} = 1")
            else:
                floor_val = math.floor(branch_val)
                ceil_val = math.ceil(branch_val)
                print(f"  Branching on variable x_{branch_var + 1} with value {branch_val:.6f}")
                print(f"  Creating two branches: x_{branch_var + 1} ≤ {floor_val} and x_{branch_var + 1} ≥ {ceil_val}")
            
            # Process left branch (floor)
            left_node = f"S{node_counter}"
            node_counter += 1
            
            # Create the "floor" branch
            floor_lower_bounds = node_lower_bounds.copy()
            floor_upper_bounds = node_upper_bounds.copy()
            
            # For binary variables, set both bounds to 0 (x=0)
            if branch_var in self.binary_vars:
                floor_lower_bounds[branch_var] = 0
                floor_upper_bounds[branch_var] = 0
                branch_cond = f"= 0"
            else:
                floor_upper_bounds[branch_var] = floor_val
                branch_cond = f"≤ {floor_val}"
            
            # Solve the relaxation for this node
            x_floor, obj_floor = self.solve_relaxation(floor_lower_bounds, floor_upper_bounds)
            
            # Add node to graph
            self.add_node_to_graph(left_node, obj_floor if x_floor is not None else float('-inf'), 
                                x_floor, node_name, branch_var, branch_cond)
            
            # Process the floor branch
            if x_floor is None:
                print(f"  {left_node} is infeasible")
            else:
                print(f"  {left_node} relaxation objective: {obj_floor:.6f}")
                print(f"  {left_node} solution: {x_floor}")
                
                # Check if integer feasible and update best solution if needed
                if self.is_integer_feasible(x_floor) and ((self.maximize and obj_floor > self.best_objective) or 
                                                        (not self.maximize and obj_floor < self.best_objective)):
                    self.best_solution = x_floor.copy()
                    self.best_objective = obj_floor
                    print(f"  Found new best integer solution with objective {self.best_objective:.6f}")
                
                # Add to queue if not fathomed
                if ((self.maximize and obj_floor > self.best_objective) or 
                    (not self.maximize and obj_floor < self.best_objective)):
                    if not self.is_integer_feasible(x_floor):  # Only branch if not integer feasible
                        priority = -obj_floor if self.maximize else obj_floor
                        node_queue.put((priority, self.nodes_explored, left_node, 
                                       floor_lower_bounds.copy(), floor_upper_bounds.copy()))
                        self.active_nodes.add(left_node)
            
            # Process right branch (ceil)
            right_node = f"S{node_counter}"
            node_counter += 1
            
            # Create the "ceil" branch
            ceil_lower_bounds = node_lower_bounds.copy()
            ceil_upper_bounds = node_upper_bounds.copy()
            
            # For binary variables, set both bounds to 1 (x=1)
            if branch_var in self.binary_vars:
                ceil_lower_bounds[branch_var] = 1
                ceil_upper_bounds[branch_var] = 1
                branch_cond = f"= 1"
            else:
                ceil_lower_bounds[branch_var] = ceil_val
                branch_cond = f"≥ {ceil_val}"
            
            # Solve the relaxation for this node
            x_ceil, obj_ceil = self.solve_relaxation(ceil_lower_bounds, ceil_upper_bounds)
            
            # Add node to graph
            self.add_node_to_graph(right_node, obj_ceil if x_ceil is not None else float('-inf'), 
                                x_ceil, node_name, branch_var, branch_cond)
            
            # Process the ceil branch
            if x_ceil is None:
                print(f"  {right_node} is infeasible")
            else:
                print(f"  {right_node} relaxation objective: {obj_ceil:.6f}")
                print(f"  {right_node} solution: {x_ceil}")
                
                # Check if integer feasible and update best solution if needed
                if self.is_integer_feasible(x_ceil) and ((self.maximize and obj_ceil > self.best_objective) or 
                                                       (not self.maximize and obj_ceil < self.best_objective)):
                    self.best_solution = x_ceil.copy()
                    self.best_objective = obj_ceil
                    print(f"  Found new best integer solution with objective {self.best_objective:.6f}")
                
                # Add to queue if not fathomed
                if ((self.maximize and obj_ceil > self.best_objective) or 
                    (not self.maximize and obj_ceil < self.best_objective)):
                    if not self.is_integer_feasible(x_ceil):  # Only branch if not integer feasible
                        priority = -obj_ceil if self.maximize else obj_ceil
                        node_queue.put((priority, self.nodes_explored, right_node, 
                                      ceil_lower_bounds.copy(), ceil_upper_bounds.copy()))
                        self.active_nodes.add(right_node)
            
            # Update upper bound as the best objective in the remaining nodes
            if not node_queue.empty():
                # Upper bound is the best possible objective in the remaining nodes
                next_priority = node_queue.queue[0][0]
                upper_bound = -next_priority if self.maximize else next_priority
            else:
                upper_bound = self.best_objective
            
            # Add to steps table
            active_nodes_str = "∅" if not self.active_nodes else "{" + ", ".join(self.active_nodes) + "}"
            lb_str = f"{self.best_objective:.2f}" if self.best_objective != float('-inf') else "-"
            x_star_str = "-" if self.best_solution is None else f"({', '.join([f'{x:.2f}' for x in self.best_solution])})"
            
            self.steps_table.append([
                node_name, 
                f"{self.graph.nodes[node_name]['obj']:.2f}", 
                f"({', '.join([f'{x:.2f}' for x in self.graph.nodes[node_name]['x']])})",
                lb_str, x_star_str, f"{upper_bound:.2f}", lb_str, active_nodes_str
            ])
        
        print("\nBranch and bound completed!")
        print(f"Nodes explored: {self.nodes_explored}")
        
        if self.best_solution is not None:
            print(f"Optimal objective: {self.best_objective:.6f}")
            print(f"Optimal solution: {self.best_solution}")
        else:
            print("No feasible integer solution found")
        
        # Display steps table
        self.display_steps_table()
        
        # Visualize the graph
        self.visualize_graph()
        
        return self.best_solution, self.best_objective