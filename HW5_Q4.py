import contextlib
import io
import tempfile
import time
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from OMADS import poll, mads
import matplotlib.pyplot as plt

# Ensure the temporary post directory exists to prevent folder errors
if not os.path.exists("./post"):
    os.makedirs("./post")

# =====================================================================
# 1. NOTEBOOK SOLVER COMPONENT HELPERS (From Dr. Ahmed Bayoumy's L10)
# =====================================================================
BASE_OPTIONS = {
    "seed": 12345, "budget": 1000, "tol": 1e-12, "display": False, "save_results": False,
    "check_cache": True, "store_cache": True, "rich_direction": True,
    "opportunistic": False, "is_verbose": False
}
BASE_SEARCH = {"type": "sampling", "s_method": "ACTIVE", "ns": 10, "visualize": False}
SOLVERS = {"poll": poll, "mads": mads}
POST_DIR = "./post"

def compute_h(c):
    """Computes the constraint violation function h(x) (A&H, Section 10.3)."""
    return float(sum(max(v, 0.0)**2 for v in c))

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
        # OMADS main returns a tuple containing the summary dict at index 0
        summary_out = SOLVERS[solver].main(data)
        summary = summary_out[0] if isinstance(summary_out, tuple) else summary_out
    wall = time.perf_counter() - tic

    rows = []
    for x, (f, c) in calls:
        f = float(np.ravel(f)[0])
        h = compute_h(c)
        rows.append({"x": x, "f": f, "h": h, "feasible": h == 0.0 and np.isfinite(f)})
    hist = pd.DataFrame(rows)
    hist["best feasible f"] = hist["f"].where(hist["feasible"]).cummin()
    return summary, hist, wall

def report(name, summary, hist, wall):
    return {
        "run": name, 
        "fmin": summary["fmin"][0] if isinstance(summary["fmin"], (list, np.ndarray)) else summary["fmin"], 
        "hmin": summary["hmin"],
        "xmin": np.round(summary["xmin"], 5).tolist(), 
        "blackbox calls": len(hist), 
        "time [s]": round(wall, 1)
    }

# =====================================================================
# 2. QUESTION 4: WELDED BEAM MATHEMATICAL MODELS
# =====================================================================
def welded_beam_blackbox(x):
    x1, x2, x3, x4 = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    
    # Fabrication Cost Objective function
    f = 1.10471 * (x1**2) * x2 + 0.04811 * x3 * x4 * (14.0 + x2)
    
    # Engineering Constants
    P = 6000.0
    L = 14.0
    tau_max = 13600.0
    sigma_max = 30000.0
    delta_max = 0.25
    E = 30e6
    G = 12e6
    
    # Stresses and Deflections
    tau_prime = P / (np.sqrt(2.0) * x1 * x2)
    M = P * (L + x2 / 2.0)
    R = np.sqrt((x2**2) / 4.0 + ((x1 + x3) / 2.0)**2)
    J = 2.0 * (np.sqrt(2.0) * x1 * x2 * ((x2**2) / 12.0 + ((x1 + x3) / 2.0)**2))
    tau_double = (M * R) / J
    tau = np.sqrt(tau_prime**2 + 2.0 * tau_prime * tau_double * x2 / (2.0 * R) + tau_double**2)
    
    sigma = (6.0 * P * L) / (x4 * (x3**2))
    delta = (4.0 * P * (L**3)) / (E * x4 * (x3**3))
    P_c = (4.013 * E * np.sqrt((x3**2 * x4**6) / 36.0) / (L**2)) * (1.0 - (x3 / (2.0 * L)) * np.sqrt(E / (4.0 * G)))
    
    # Normalized Constraints c_j(x) <= 0
    c1 = (tau / tau_max) - 1.0
    c2 = (sigma / sigma_max) - 1.0
    c3 = x1 - x4
    c4 = (0.10471 * (x1**2) + 0.04811 * x3 * x4 * (14.0 + x2)) / 5.0 - 1.0
    c5 = 0.125 - x1
    c6 = (delta / delta_max) - 1.0
    c7 = 1.0 - (P_c / P)
    
    # MUST BE EXPLICITLY RETURNED AS A LIST TO AVOID TYPEERRORS
    return [f, [c1, c2, c3, c4, c5, c6, c7]]

# Problem Constraints Setup
x0 = [1.0, 5.0, 5.0, 1.0]
lb = [0.1, 0.1, 0.1, 0.1]
ub = [2.0, 10.0, 10.0, 2.0]
var_names = ["x1", "x2", "x3", "x4"]

# =====================================================================
# 3. RUN OMADS SOLVERS (PART B)
# =====================================================================
param_base = {
    "baseline": x0, "lb": lb, "ub": ub, "var_names": var_names, 
    "scaling": 10.0, "constraints_type": ["PB"] * 7
}

print("Running OMADS Poll solver...")
sum_poll, hist_poll, wall_poll = solve(welded_beam_blackbox, {**param_base, "name": "beam_poll"}, solver="poll", options={"budget": 3000})

print("Running OMADS MADS solver...")
sum_mads, hist_mads, wall_mads = solve(welded_beam_blackbox, {**param_base, "name": "beam_mads"}, solver="mads", options={"budget": 3000})

# =====================================================================
# 4. RUN SCIPY NELDER-MEAD SOLVERS (PART C)
# =====================================================================
lb_arr = np.array(lb, dtype=float)
ub_arr = np.array(ub, dtype=float)

def eval_with_bounds(x):
    x = np.asarray(x, dtype=float)
    f, c = welded_beam_blackbox(x)
    bound_c = np.concatenate([lb_arr - x, x - ub_arr])
    h_total = compute_h(list(c) + list(bound_c))
    return f, c, h_total

def nm_extreme_barrier(x):
    f, _, h = eval_with_bounds(x)
    return f if h == 0.0 else np.inf

def nm_quadratic_penalty(x):
    f, _, h = eval_with_bounds(x)
    return f + 1e3 * h

res_nm_eb = minimize(nm_extreme_barrier, x0, method='Nelder-Mead', options={'maxfev': 3000})
res_nm_pen = minimize(nm_quadratic_penalty, x0, method='Nelder-Mead', options={'maxfev': 3000})

# =====================================================================
# 5. PRINT ALL ACCURATE OUTPUT METRICS
# =====================================================================
print("\n" + "="*80)
print("                   QUESTION 4 FINAL RESULTS SUMMARY")
print("="*80)
print(pd.DataFrame([
    report("OMADS Poll", sum_poll, hist_poll, wall_poll),
    report("OMADS MADS", sum_mads, hist_mads, wall_mads)
]).to_string(index=False))

# Check active constraints (c_j >= -1e-3) for OMADS runs
for label, summary in [("OMADS Poll", sum_poll), ("OMADS MADS", sum_mads)]:
    _, c_opt = welded_beam_blackbox(summary["xmin"])
    active = [f"c{j+1} ({val:.2e})" for j, val in enumerate(c_opt) if val >= -1e-3]
    print(f"{label} constraints: {np.round(c_opt, 5).tolist()} | Active: {active}")

# Nelder-Mead Results (including bound violations in h)
f_eb, c_eb, h_eb = eval_with_bounds(res_nm_eb.x)
f_pen, c_pen, h_pen = eval_with_bounds(res_nm_pen.x)

print("-"*80)
print(f"{'Nelder-Mead (EB)':<24} | fmin: {f_eb:<8.5f} | hmin: {h_eb:<8.5e} | fev: {res_nm_eb.nfev:<4d} | xmin: {np.round(res_nm_eb.x, 5).tolist()}")
print(f"{'Nelder-Mead (Penalty)':<24} | fmin: {f_pen:<8.5f} | hmin: {h_pen:<8.5e} | fev: {res_nm_pen.nfev:<4d} | xmin: {np.round(res_nm_pen.x, 5).tolist()}")
print("="*80 + "\n")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(np.arange(1, len(hist_poll) + 1), hist_poll['best feasible f'], label='OMADS Poll (No Search)', color='blue', lw=2)
ax.plot(np.arange(1, len(hist_mads) + 1), hist_mads['best feasible f'], label='OMADS MADS (Search+Poll)', color='red', lw=2)
ax.set_yscale('log')
ax.set_xlabel('Number of Evaluations')
ax.set_ylabel('Best Feasible Fabrication Cost $f(x)$')
ax.set_title('Welded Beam: OMADS Performance Comparison')
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.show()
