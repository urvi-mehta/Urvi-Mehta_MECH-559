import numpy as np

def spring_blackbox(x):
    """
    Evaluates the tension/compression spring design problem.
    Returns: [f, [c1, c2, c3, c4]]
    """
    x1, x2, x3 = float(x[0]), float(x[1]), float(x[2])
    
    # Objective function: weight of the spring
    f = (x3 + 2.0) * x2 * (x1**2)
    
    # Constraint functions c_j(x) <= 0
    c1 = 1.0 - (x2**3 * x3) / (71785.0 * x1**4)
    
    term2_num = 4.0 * x2**2 - x1 * x2
    term2_den = 12566.0 * (x2 * x1**3 - x1**4)
    c2 = (term2_num / term2_den) + (1.0 / (5108.0 * x1**2)) - 1.0
    
    c3 = 1.0 - (140.45 * x1) / (x2**2 * x3)
    
    c4 = ((x1 + x2) / 1.5) - 1.0
    
    return [f, [c1, c2, c3, c4]]

def compute_h(constraints):
    """Computes constraint violation h(x)."""
    return sum(max(c, 0.0)**2 for c in constraints)

# Starting points
x_A = np.array([0.1, 1.0, 10.0])
x_B = np.array([0.05, 1.0, 2.0])
x_C = np.array([0.06, 0.5, 10.0])

for name, pt in zip(["xA", "xB", "xC"], [x_A, x_B, x_C]):
    f, constrs = spring_blackbox(pt)
    h = compute_h(constrs)
    is_feasible = "Feasible" if h == 0.0 else "Infeasible"
    print(f"Point {name}: f = {f:.6f}, h = {h:.6f} ({is_feasible})")
    print(f"  Constraints: [c1={constrs[0]:.4f}, c2={constrs[1]:.4f}, c3={constrs[2]:.4f}, c4={constrs[3]:.4f}]\n")

