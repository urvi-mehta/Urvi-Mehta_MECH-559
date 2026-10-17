import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Problem Formulation ---
def f(x):
    return x[0]**2 + (x[1] - 3.0)**2

def grad_f(x):
    return np.array([2.0 * x[0], 2.0 * (x[1] - 3.0)])

def g(x):
    return x[1]**2 - 2.0*x[0]

def grad_g(x):
    return np.array([-2.0, 2.0 * x[1]])

# Analytical optimal point x* from solving the cubic equation x2^3 + 2*x2 - 6 = 0
roots = np.roots([1.0, 0.0, 2.0, -6.0])
x2_star = roots[np.isreal(roots)].real[0]
x1_star = (x2_star**2) / 2.0
x_star = np.array([x1_star, x2_star])
mu_star = x1_star

print("--- (a) Analytical Exact Optimum ---")
print(f"x* = [{x_star[0]:.5f}, {x_star[1]:.5f}]")
print(f"f(x*) = {f(x_star):.5f}")
print(f"mu* = {mu_star:.5f}\n")


# --- (b) Penalty Method Implementation ---
def T_penalty(x, r):
    return f(x) + (1.0 / r) * max(0.0, g(x))**2

def jac_penalty(x, r):
    val_g = g(x)
    if val_g <= 0:
        return grad_f(x)
    else:
        return grad_f(x) + (2.0 / r) * val_g * grad_g(x)

def hessian_penalty(x, r):
    val_g = g(x)
    if val_g <= 0:
        return np.array([[2.0, 0.0], [0.0, 2.0]])
    else:
        H11 = 2.0 + 8.0 / r
        H12 = -8.0 * x[1] / r
        H22 = 2.0 + 4.0 * val_g / r + 8.0 * x[1]**2 / r
        return np.array([[H11, H12], [H12, H22]])

x0 = np.array([0.0, 3.0])
penalty_path = [x0.copy()]
penalty_errors = []

print("--- (b) Penalty Method Sequence Log ---")
print(f"{'k':<4}{'r_k':<10}{'x_k^T':<24}{'f(x_k)':<10}{'g(x_k)':<10}{'mu_k':<10}{'||x_k-x*||':<12}{'Cond No.':<10}")
print("-" * 96)

x_curr = x0.copy()
for k in range(7):
    r_k = 0.1**k
    res = minimize(T_penalty, x_curr, args=(r_k,), jac=jac_penalty, method='BFGS', tol=1e-12)
    x_curr = res.x
    penalty_path.append(x_curr.copy())
    
    val_g = g(x_curr)
    mu_k = (2.0 / r_k) * max(0.0, val_g)
    err = np.linalg.norm(x_curr - x_star)
    penalty_errors.append(err)
    
    H = hessian_penalty(x_curr, r_k)
    cond_num = np.linalg.cond(H)
    
    print(f"{k:<4}{r_k:<10.1e}[{x_curr[0]:.5f}, {x_curr[1]:.5f}]   {f(x_curr):<10.5f}{val_g:<10.5f}{mu_k:<10.5f}{err:<12.2e}{cond_num:<10.2f}")


# --- (c) Augmented Lagrangian Method Implementation ---
def T_augmented(x, r, mu):
    val_g = g(x)
    return f(x) + mu * val_g + (1.0 / r) * max(0.0, val_g)**2

def jac_augmented(x, r, mu):
    val_g = g(x)
    if val_g <= 0:
        return grad_f(x) + mu * grad_g(x)
    else:
        return grad_f(x) + mu * grad_g(x) + (2.0 / r) * val_g * grad_g(x)

al_path = [x0.copy()]
al_errors = []
r_fixed = 1.0
mu_curr = 0.0
x_curr_al = x0.copy()

print("\n--- (c) Augmented Lagrangian Sequence Log (r = 1.0) ---")
print(f"{'k':<4}{'mu_k':<10}{'x_k^T':<24}{'g(x_k)':<10}{'||x_k-x*||':<12}")
print("-" * 64)

for k in range(1, 9):
    res = minimize(T_augmented, x_curr_al, args=(r_fixed, mu_curr), jac=jac_augmented, method='BFGS', tol=1e-12)
    x_curr_al = res.x
    al_path.append(x_curr_al.copy())
    
    val_g = g(x_curr_al)
    err = np.linalg.norm(x_curr_al - x_star)
    al_errors.append(err)
    
    print(f"{k-1:<4}{mu_curr:<10.5f}[{x_curr_al[0]:.5f}, {x_curr_al[1]:.5f}]   {val_g:<10.5f}{err:<12.2e}")
    
    mu_curr = mu_curr + (2.0 / r_fixed) * val_g

penalty_path = np.array(penalty_path)
al_path = np.array(al_path)


# --- (d) Visualization Code ---
plt.rcParams.update({"font.size": 11, "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Subplot 1: Contour Plot of Iteration Paths
x1_grid, x2_grid = np.meshgrid(np.linspace(-0.5, 2.5, 400), np.linspace(0.0, 4.0, 400))
F_grid = x1_grid**2 + (x2_grid - 3.0)**2
boundary_x2 = np.linspace(0.0, 4.0, 400)
boundary_x1 = (boundary_x2**2) / 2.0

axes[0].contour(x1_grid, x2_grid, F_grid, levels=30, cmap="Greys", alpha=0.5)
axes[0].plot(boundary_x1, boundary_x2, color="black", lw=1.5, label=r"$g(x)=0$")
axes[0].fill_betweenx(boundary_x2, boundary_x1, 2.5, color="#A8DADC", alpha=0.25, label="Feasible Region")

axes[0].plot(penalty_path[:, 0], penalty_path[:, 1], "-s", color="#D1495B", ms=5, label="Penalty Path")
axes[0].plot(al_path[:, 0], al_path[:, 1], "-o", color="#1D5FA8", ms=5, label="Augmented Lagrangian Path")
axes[0].plot(x_star[0], x_star[1], "*", color="#2A9D5C", ms=18, label="Exact Optimum")

axes[0].set_xlabel("$x_1$")
axes[0].set_ylabel("$x_2$")
axes[0].set_xlim(-0.5, 2.5)
axes[0].set_ylim(0.0, 4.0)
axes[0].set_title("Optimization Iterate Trajectories")
axes[0].legend(loc="upper left", fontsize=9)

# Subplot 2: Semi-log Convergence Rates Plot
axes[1].semilogy(range(len(penalty_errors)), penalty_errors, "-s", color="#D1495B", label="Penalty Method ($r_k=0.1^k$)")
axes[1].semilogy(range(len(al_errors)), al_errors, "-o", color="#1D5FA8", label="Augmented Lagrangian ($r=1.0$)")
axes[1].set_xlabel("Outer Iteration Step $k$")
axes[1].set_ylabel(r"Error Distance $\|x_k - x^*\|$")
axes[1].set_title("Convergence Distance Rate to Optimum")
axes[1].legend()

plt.tight_layout()
plt.show()
