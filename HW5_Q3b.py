import contextlib
import io
import tempfile
import time
import numpy as np
import pandas as pd
from OMADS import poll, search, mads

# ==========================================
# 1. CORE SETUP FROM THE L10 NOTEBOOK
# ==========================================
BASE_OPTIONS = {
    "seed": 12345, 
    "budget": 1000, 
    "tol": 1e-12, 
    "display": False, 
    "save_results": False,
    "check_cache": True, 
    "store_cache": True, 
    "rich_direction": True,
    "opportunistic": False, 
    "is_verbose": False
}
BASE_SEARCH = {"type": "sampling", "s_method": "ACTIVE", "ns": 10, "visualize": False}
SOLVERS = {"poll": poll, "search": search, "mads": mads}
POST_DIR = tempfile.mkdtemp(prefix="omads_post_")

def solve(blackbox, param, solver="mads", options=None, search_options=None):
    """Run an OMADS solver quietly; return (summary dict, history DataFrame, wall time)."""
    calls = []

    def logged_blackbox(x):                    
        out = blackbox(x)
        calls.append((list(x), out))
        return out

    data = {
        "evaluator": {"blackbox": logged_blackbox},
        "param": {"post_dir": POST_DIR, **param},
        "options": {**BASE_OPTIONS, **(options or {})},
        "search": {**BASE_SEARCH, **(search_options or {})}
    }
    tic = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        summary = SOLVERS[solver].main(data)[0]
    wall = time.perf_counter() - tic

    rows = []
    for x, (f, c) in calls:
        f = float(np.ravel(f)[0])
        # Constraint violation h(x) formula from assignment description
        h = float(np.sum(np.maximum(np.asarray(c, dtype=float), 0.0) ** 2))   
        rows.append({"x": x, "f": f, "h": h, "feasible": h == 0.0 and np.isfinite(f)})
    hist = pd.DataFrame(rows)
    hist["best feasible f"] = hist["f"].where(hist["feasible"]).cummin()
    return summary, hist, wall

def report(name, summary, hist, wall):
    return {
        "run": name, 
        "fmin": summary["fmin"][0], 
        "hmin": summary["hmin"],
        "xmin": np.round(summary["xmin"], 5).tolist(), 
        "blackbox calls": len(hist), 
        "time [s]": round(wall, 1)
    }

# ==========================================
# 2. DEFINING THE PROBLEM (SPRING DESIGN)
# ==========================================
def spring_blackbox(x):
    x1, x2, x3 = float(x[0]), float(x[1]), float(x[2])
    
    # Objective function: minimize weight
    f = (x3 + 2.0) * x2 * (x1**2)
    
    # Constraints c_j(x) <= 0
    c1 = 1.0 - (x2**3 * x3) / (71785.0 * (x1**4))
    
    c2_num = 4.0 * (x2**2) - (x1 * x2)
    c2_den = 12566.0 * (x2 * (x1**3) - (x1**4))
    c2 = (c2_num / c2_den) + (1.0 / (5108.0 * (x1**2))) - 1.0
    
    c3 = 1.0 - (140.45 * x1) / ((x2**2) * x3)
    c4 = ((x1 + x2) / 1.5) - 1.0
    
    return [f, [c1, c2, c3, c4]]

# Problem Boundaries and Configurations
lb_spring = [0.05, 0.25, 2.0]
ub_spring = [2.0, 1.3, 15.0]
var_names = ["x1", "x2", "x3"]

# Starting vectors
xA = [0.1, 1.0, 10.0]
xB = [0.05, 1.0, 2.0]

# Execution Loop for the 4 Runs
experimental_runs = [
    ("xA_EB", xA, "EB"),
    ("xA_PB", xA, "PB"),
    ("xB_EB", xB, "EB"),
    ("xB_PB", xB, "PB")
]

results = []
for run_id, x0, barrier_type in experimental_runs:
    param = {
        "name": f"spring_{run_id}",
        "baseline": x0,
        "lb": lb_spring,
        "ub": ub_spring,
        "var_names": var_names,
        "scaling": 10.0,
        "constraints_type": [barrier_type] * 4 # Sets barrier type per constraint
    }
    
    # Configured with assignment's specific parameters
    options = {"seed": 12345, "budget": 2000}
    
    try:
        summary, hist, wall = solve(spring_blackbox, param, solver="mads", options=options)
        
        # Calculate evaluation categories
        total_evals = len(hist)
        infeasible_evals = int((hist["h"] > 0).sum())
        
        print(f"--- Run Profile: {run_id} ---")
        print(f"Best Feasible f: {summary['fmin'][0]}")
        print(f"Best Point x:   {summary['xmin']}")
        print(f"Total Blackbox Evaluations: {total_evals}")
        print(f"Infeasible Evaluations:     {infeasible_evals}")
        
        # Re-evaluate best point to see the constraint values
        _, final_constraints = spring_blackbox(summary["xmin"])
        print(f"Final Constraints values: {np.round(final_constraints, 5)}\n")
        
    except Exception as e:
        print(f"--- Run Profile: {run_id} failed ---")
        print(f"Reason: {e}\n")
