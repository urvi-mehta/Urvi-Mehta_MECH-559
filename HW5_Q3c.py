import contextlib
import io
import tempfile
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from OMADS import mads

# Create post-processing storage directory safely
if not os.path.exists("./post"):
    os.makedirs("./post")

# =====================================================================
# 1. CORE COMPONENT IMPLEMENTATION FROM THE L10 NOTEBOOK HELPERS
# =====================================================================
BASE_OPTIONS = {
    "seed": 12345, "budget": 1000, "tol": 1e-12, "display": False, "save_results": False,
    "check_cache": True, "store_cache": True, "rich_direction": True,
    "opportunistic": False, "is_verbose": False
}
BASE_SEARCH = {"type": "sampling", "s_method": "ACTIVE", "ns": 10, "visualize": False}
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
        summary, _, _ = mads.main(data)
    wall = time.perf_counter() - tic

    rows = []
    for x, (f, c) in calls:
        f = float(np.ravel(f)[0])
        h = float(np.sum(np.maximum(np.asarray(c, dtype=float), 0.0) ** 2))   # constraint violation h(x)
        rows.append({"x": x, "f": f, "h": h, "feasible": h == 0.0 and np.isfinite(f)})
    hist = pd.DataFrame(rows)
    hist["best feasible f"] = hist["f"].where(hist["feasible"]).cummin()
    return summary, hist, wall

# =====================================================================
# 2. DEFINE SPRING PHYSICS PROBLEM FORMULAS
# =====================================================================
def spring_blackbox(x):
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
# 3. CONFIGURE AND RUN EXPERIMENT FROM POINT xB TO CRERATE "results"
# =====================================================================
xB = [0.05, 1.0, 2.0]
results = {}

print("Running optimization from point xB using Progressive Barrier...")
param_xB_pb = {
    "name": "spring_xB_pb",
    "baseline": xB,
    "lb": [0.05, 0.25, 2.0],
    "ub": [2.0, 1.3, 15.0],
    "var_names": ["x1", "x2", "x3"],
    "scaling": 10.0,
    "constraints_type": ["PB"] * 4
}
options = {"seed": 12345, "budget": 2000}

# Populate the results memory cache
summary_xB_pb, hist_xB_pb, wall_xB_pb = solve(spring_blackbox, param_xB_pb, solver="mads", options=options)
results["xB_PB"] = (summary_xB_pb, hist_xB_pb, wall_xB_pb)
print("Optimization finished successfully!\n")

# =====================================================================
# 4. PLOT GENERATION USING CORRECT PANDAS CHROMATIC MATRICES
# =====================================================================
print("Generating convergence plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Left Plot: History Trace ---
# Uses the pre-computed column "best feasible f" from the dataframe
ax1.plot(hist_xB_pb.index + 1, hist_xB_pb["best feasible f"], color='blue', lw=2)
ax1.set_xlabel("Number of Evaluations")
ax1.set_ylabel("Objective $f(x)$")
ax1.set_title("Convergence History from $x_B^0$ (PB)")
ax1.grid(True, alpha=0.25)

# --- Right Plot: (h, f) space trace ---
# Maps constraint violation "h" vs objective value "f" over time sequence
sc = ax2.scatter(
    hist_xB_pb["h"], 
    hist_xB_pb["f"], 
    c=hist_xB_pb.index + 1, 
    cmap='viridis', 
    alpha=0.6, 
    s=15
)
ax2.set_xlabel("Constraint Violation $h(x)$")
ax2.set_ylabel("Objective Value $f(x)$")
ax2.set_title("Evaluated Points in the $(h, f)$ Plane")
ax2.grid(True, alpha=0.25)

# Add a colorbar tracker indicating the loop index progress
plt.colorbar(sc, ax=ax2, label='Evaluation Count')

plt.tight_layout()

# Save the figure locally to clear backend display lockouts
plt.savefig("question_3c_plots.png", dpi=300)
print("Plot successfully saved to disk as 'question_3c_plots.png'.")

plt.show()
