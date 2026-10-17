import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Problem Objective & Constraints Definitions ---
def f(x):
    return (x[0] - x[1])**2 + (x[1] - x[2])**2 + (x[2] - x[3])**2 + (x[3] - x[4])**2

def hvec(x):
    return np.array([
        x[0] + x[1]**2 + x[2]**3 - 3.0,
        x[1] - x[2]**2 + x[3] - 1.0,
        x[0]*x[4] - 1.0
    ])

def jac_h_partitions(x):
    # State: (x1, x4, x5), Decision: (x2, x3)
    grad_xs_h = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [x[4], 0.0, x[0]]
    ])
    grad_xd_h = np.array([
        [2.0*x[1], 3.0*x[2]**2],
        [1.0, -2.0*x[2]],
        [0.0, 0.0]
    ])
    return grad_xs_h, grad_xd_h

def grad_f_partitions(x):
    g1 = 2.0*(x[0] - x[1])
    g2 = 2.0*(x[1] - x[0]) + 2.0*(x[1] - x[2])
    g3 = 2.0*(x[2] - x[1]) + 2.0*(x[2] - x[3])
    g4 = 2.0*(x[3] - x[2]) + 2.0*(x[3] - x[4])
    g5 = 2.0*(x[4] - x[3])
    
    grad_xs_f = np.array([g1, g4, g5])
    grad_xd_f = np.array([g2, g3])
    return grad_xs_f, grad_xd_f

def restore_state(xd, xs_guess, max_inner=100, tol=1e-12):
    xs = xs_guess.copy()
    x = np.zeros(5)
    for _ in range(max_inner):
        x[0], x[3], x[4] = xs[0], xs[1], xs[2]
        x[1], x[2] = xd[0], xd[1]
        
        h = hvec(x)
        if np.linalg.norm(h) < tol:
            return xs, True
            
        grad_xs_h, _ = jac_h_partitions(x)
        try:
            dxs = np.linalg.solve(grad_xs_h, -h)
        except np.linalg.LinAlgError:
            return xs, False
        xs += dxs
    return xs, False

def run_grg(x0, kmax=5000, tol=1e-6, beta=0.5, epsilon=1e-4):
    x = np.array(x0, dtype=float)
    xd = np.array([x[1], x[2]])  
    xs = np.array([x[0], x[3], x[4]])  
    
    path_xd = [xd.copy()]
    grad_w_norms = []
    
    print(f"{'Iter k':<6} {'f(x_k)':<10} {'Max ||h||':<10} {'||grad_w||':<12} {'Step Alpha':<10}")
    print("-" * 55)
    
    for k in range(kmax):
        x[0], x[3], x[4] = xs[0], xs[1], xs[2]
        x[1], x[2] = xd[0], xd[1]
        
        grad_xs_h, grad_xd_h = jac_h_partitions(x)
        grad_xs_f, grad_xd_f = grad_f_partitions(x)
        
        inv_jac_prod = np.linalg.solve(grad_xs_h, grad_xd_h)
        grad_w = grad_xd_f - grad_xs_f @ inv_jac_prod
        w_norm = np.linalg.norm(grad_w)
        grad_w_norms.append(w_norm)
        
        # Print progress row FOR EACH iteration
        max_h = np.linalg.norm(hvec(x))
        print(f"{k:<6} {f(x):<10.6f} {max_h:<10.2e} {w_norm:<12.4e} {0.0 if k==0 else alpha:<10.4f}")
        
        if w_norm < tol:
            break
            
        alpha = 1.0
        f_curr = f(x)
        step_accepted = False
        
        while alpha > 1e-15:
            xd_trial = xd - alpha * grad_w
            xs_pred = xs - alpha * (inv_jac_prod @ grad_w)
            
            xs_restored, success = restore_state(xd_trial, xs_pred)
            if success:
                x_trial = np.zeros(5)
                x_trial[0], x_trial[3], x_trial[4] = xs_restored[0], xs_restored[1], xs_restored[2]
                x_trial[1], x_trial[2] = xd_trial[0], xd_trial[1]
                
                if f(x_trial) <= f_curr - epsilon * alpha * w_norm**2:
                    xd = xd_trial
                    xs = xs_restored
                    path_xd.append(xd.copy())
                    step_accepted = True
                    break
            alpha *= beta
            
        if not step_accepted:
            print(f"Line search stalled at step {k}")
            break
            
    # FIXED: Added the sensitivity multiplier calculation equation for part (d)
    # Solve: grad_xs_h.T @ lambda = -grad_xs_f
    lambda_vals = np.linalg.solve(grad_xs_h.T, -grad_xs_f)
            
    return x, np.array(path_xd), grad_w_norms, k, lambda_vals

# --- Execute Optimization ---
x0 = np.array([2.0, np.sqrt(2.0), -1.0, 2.0 - np.sqrt(2.0), 0.5])
# FIXED: Modified unpacking variable list to catch the new multiplier output
x_opt, xd_history, w_norms, iterations, final_lambdas = run_grg(x0)

# --- part (d) SciPy SLSQP Cross-Check Baseline ---
cons_dict = [{"type": "eq", "fun": hvec}]
scipy_res = minimize(f, x0, method="SLSQP", constraints=cons_dict, tol=1e-8)

print("\n=================== FINAL SUMMARY ===================")
print(f"GRG Iterations       : {iterations}")
print(f"Final Solution x     : {np.round(x_opt, 6)}")
print(f"Objective f(x*)      : {f(x_opt):.6f}")
print(f"Final ||grad_w||     : {w_norms[-1]:.2e}")
print(f"Lagrange Multipliers : {np.round(final_lambdas, 5)}") # FIXED: Prints multipliers now
print("\n--- SciPy SLSQP Cross-Check Baseline ---")
print(f"SLSQP Iterations     : {scipy_res.nit}")
print(f"SLSQP Solution x     : {np.round(scipy_res.x, 6)}")
print(f"SLSQP Objective f(x) : {scipy_res.fun:.6f}")
print("======================================================")

# --- Generate Fixed Plots ---
x2_vec = np.linspace(0.4, 1.8, 400)
x3_vec = np.linspace(-1.2, 1.2, 400)
X2, X3 = np.meshgrid(x2_vec, x3_vec)

# Explicit elimination equations
X1 = 3.0 - X2**2 - X3**3
X4 = 1.0 - X2 + X3**2

# FIX 1: Mask singular points where X1 <= 0
X1_safe = np.where(X1 <= 1e-5, np.nan, X1)
X5 = 1.0 / X1_safe
W_mesh = (X1_safe - X2)**2 + (X2 - X3)**2 + (X3 - X4)**2 + (X4 - X5)**2

# FIX 2: Clip the mesh array to cap infinity outliers so contour colors scale properly
W_mesh = np.clip(W_mesh, 0.0, 50.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# FIX 3: Manually assign close contour lines near the lower bounds
levels = np.linspace(0.0, 20.0, 40)
cp = ax1.contour(X2, X3, W_mesh, levels=levels, cmap="viridis", alpha=0.8)
ax1.clabel(cp, inline=True, fontsize=8, fmt="%.1f")

ax1.plot(xd_history[:, 0], xd_history[:, 1], "-or", ms=4, lw=1.2, label="GRG Iteration Steps")
ax1.plot(np.sqrt(2.0), -1.0, "Xg", ms=11, label="Initial Guess $x_0$")
# FIX 4: Extract correct 2D index slices for decision optimum plotting
ax1.plot(x_opt[1], x_opt[2], "*b", ms=14, label="Final Optimum $x^*$")

ax1.set_xlabel("$x_2$ (Decision)", fontsize=11)
ax1.set_ylabel("$x_3$ (Decision)", fontsize=11)
ax1.set_title("GRG Path over Reduced Objective Contours", fontsize=12, fontweight="bold")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.25)

# Plot 2: Semi-Log Convergence Profile
ax2.semilogy(range(len(w_norms)), w_norms, "-o", color="#1D5FA8", ms=4, lw=1.2)
ax2.set_xlabel("Outer Iteration Step $k$", fontsize=11)
ax2.set_ylabel("$\\|\\nabla w_k\\|$ Norm (Log Scale)", fontsize=11)
ax2.set_title("Reduced Gradient Norm Convergence History", fontsize=12, fontweight="bold")
ax2.grid(True, which="both", alpha=0.25)

plt.tight_layout()
plt.show()
