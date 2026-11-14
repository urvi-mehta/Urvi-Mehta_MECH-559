import os
import numpy as np
import pandas as pd
from OMADS import mads

# =====================================================================
# 1. DEFINE SPRING PHYSICS PROBLEM FORMULAS
# =====================================================================
def spring_blackbox(x):
    """
    Evaluates the tension/compression spring design problem.
    Returns: [objective_value, [c1, c2, c3, c4]]
    """
    x1, x2, x3 = float(x[0]), float(x[1]), float(x[2])
    
    # Objective function f(x): weight of the spring
    f = (x3 + 2.0) * x2 * (x1**2)
    
    # Constraint functions c_j(x) <= 0
    c1 = 1.0 - (x2**3 * x3) / (71785.0 * x1**4)
    
    term2_num = 4.0 * x2**2 - x1 * x2
    term2_den = 12566.0 * (x2 * x1**3 - x1**4)
    c2 = (term2_num / term2_den) + (1.0 / (5108.0 * x1**2)) - 1.0
    
    c3 = 1.0 - (140.45 * x1) / (x2**2 * x3)
    c4 = ((x1 + x2) / 1.5) - 1.0
    
    return [f, [c1, c2, c3, c4]]

# =====================================================================
# 2. FIXED SOLVER EXPERIMENT RUNNER WITH DIRECT DICT KEY EXTRACTION
# =====================================================================
def run_solver_experiment(start_pt, barrier, name):
    eval_hist = []
    
    # Wrapper function logs every internal evaluation point cleanly
    def tracked_bb(x):
        f_val, constrs = spring_blackbox(x)
        # Calculate penalty violation function h(x) = sum( max(c_j(x), 0) )^2
        h_val = sum(max(c, 0.0)**2 for c in constrs)
        eval_hist.append({"x": np.array(x), "f": f_val, "c": constrs, "h": h_val})
        return [f_val, constrs]

    config = {
        "evaluator": {
            "blackbox": tracked_bb
        },
        "param": {
            "name": f"spring_{name}_{barrier.lower()}",
            "baseline": list(start_pt),
            "lb": [0.05, 0.25, 2.0],
            "ub": [2.0, 1.3, 15.0],
            "constraints_type": [barrier] * 4, # Sets barrier type properly per constraint
            "post_dir": "./post"
        },
        "options": {
            "seed": 12345, # Mandated explicit random tracking seed
            "budget": 2000, # Specified maximum evaluation budget
            "tol": 1e-12,
            "store_cache": True,
            "check_cache": True,
            "display": False
        },
        "search": {
            "type": "sampling",
            "s_method": "ACTIVE",
            "ns": 10
        }
    }
    
    # Execute the raw OMADS solver main method
    out, poll_step, search_step = mads.main(config)
    return out, eval_hist

# =====================================================================
# 3. RUN THE EXPERIMENT DATA CAPTURE
# =====================================================================
actual_xC = [0.06, 0.5, 10.0]

print("Running baseline optimization experiment for point xC (PB)...")
out_xC, hist_xC = run_solver_experiment(actual_xC, "PB", "xC")
print("Optimization finished successfully!\n")

# =====================================================================
# 4. TABULATE PERFORMANCE & RELATIVE GAPS
# =====================================================================
f_star = 0.012665

# Consolidating your verified run profiles with the new run for xC
data_table = [
    {"Run": "xA_EB", "f": 0.016607, "x1": 0.06696, "x2": 0.84728, "x3": 2.37205, "Calls": 2000, "Infeasible": 1690},
    {"Run": "xA_PB", "f": 0.016903, "x1": 0.06748, "x2": 0.86821, "x3": 2.27557, "Calls": 2000, "Infeasible": 1654},
    {"Run": "xB_EB", "f": 0.013547, "x1": 0.05890, "x2": 0.55635, "x3": 5.01823, "Calls": 2000, "Infeasible": 1801},
    {"Run": "xB_PB", "f": 0.013547, "x1": 0.05890, "x2": 0.55635, "x3": 5.01823, "Calls": 2000, "Infeasible": 1801},
    {"Run": "xC_PB", "f": out_xC["fmin"][0], 
     "x1": round(out_xC["xmin"][0], 5), 
     "x2": round(out_xC["xmin"][1], 5), 
     "x3": round(out_xC["xmin"][2], 5), 
     "Calls": len(hist_xC), 
     "Infeasible": sum(1 for d in hist_xC if d["h"] > 0)}
]

df_results = pd.DataFrame(data_table)

# Compute the mandatory relative gap percentage formula ((f - f*) / f*)
df_results["Relative Gap"] = ((df_results["f"] - f_star) / f_star) * 100

print("="*95)
print(f"{'RUN PROFILE':<15} | {'f_min':<8} | {'x1':<7} | {'x2':<7} | {'x3 (coils)':<10} | {'EVALS':<5} | {'GAP (%)':<8}")
print("="*95)
for _, r in df_results.iterrows():
    print(f"{r['Run']:<15} | {r['f']:0.6f} | {r['x1']:0.5f} | {r['x2']:0.5f} | {r['x3']:<10.5f} | {r['Calls']:<5} | {r['Relative Gap']:0.2f}%")
print("="*95)
