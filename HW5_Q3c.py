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
# 3. CONFIGURE AND RUN BOTH EB AND PB FROM POINT xB
# =====================================================================
xB = [0.05, 1.0, 2.0]
options = {"seed": 12345, "budget": 2000}

param_xB_eb = {
    "name": "spring_xB_eb",
    "baseline": xB,
    "lb": [0.05, 0.25, 2.0],
    "ub": [2.0, 1.3, 15.0],
    "var_names": ["x1", "x2", "x3"],
    "scaling": 10.0,
    "constraints_type": ["EB"] * 4
}
summary_xB_eb, hist_xB_eb, wall_xB_eb = solve(spring_blackbox, param_xB_eb, solver="mads", options=options)

param_xB_pb = {
    "name": "spring_xB_pb",
    "baseline": xB,
    "lb": [0.05, 0.25, 2.0],
    "ub": [2.0, 1.3, 15.0],
    "var_names": ["x1", "x2", "x3"],
    "scaling": 10.0,
    "constraints_type": ["PB"] * 4
}
summary_xB_pb, hist_xB_pb, wall_xB_pb = solve(spring_blackbox, param_xB_pb, solver="mads", options=options)

# =====================================================================
# 4. PLOT BOTH RUNS FROM xB
# =====================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))

# Left Plot: Best Feasible f vs. Evaluations (Both EB and PB)
ax1.plot(hist_xB_eb.index + 1, hist_xB_eb["best feasible f"], color='blue', lw=2.2, label='Extreme Barrier (EB)')
ax1.plot(hist_xB_pb.index + 1, hist_xB_pb["best feasible f"], color='red', lw=1.8, linestyle='--', label='Progressive Barrier (PB)')
ax1.set_xlabel("Number of Evaluations")
ax1.set_ylabel("Best Feasible Objective $f(x)$")
ax1.set_title("Convergence History from $x_B^0$")
ax1.grid(True, alpha=0.25)
ax1.legend()

# Middle Plot: (h, f) Plane for EB
sc1 = ax2.scatter(hist_xB_eb["h"], hist_xB_eb["f"], c=hist_xB_eb.index + 1, cmap='viridis', alpha=0.6, s=15)
ax2.axvline(0, color='red', linestyle='--', lw=1)
ax2.set_xlabel("Constraint Violation $h(x)$")
ax2.set_ylabel("Objective Value $f(x)$")
ax2.set_title("Evaluated Points in $(h, f)$ Plane — EB")
ax2.grid(True, alpha=0.25)

# Right Plot: (h, f) Plane for PB
sc2 = ax3.scatter(hist_xB_pb["h"], hist_xB_pb["f"], c=hist_xB_pb.index + 1, cmap='viridis', alpha=0.6, s=15)
ax3.axvline(0, color='red', linestyle='--', lw=1)
ax3.set_xlabel("Constraint Violation $h(x)$")
ax3.set_ylabel("Objective Value $f(x)$")
ax3.set_title("Evaluated Points in $(h, f)$ Plane — PB")
ax3.grid(True, alpha=0.25)

plt.colorbar(sc2, ax=[ax2, ax3], label='Evaluation Count', pad=0.02)
plt.savefig("question_3c_plots.png", dpi=300, bbox_inches="tight")
plt.show()