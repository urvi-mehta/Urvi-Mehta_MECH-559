import numpy as np
from scipy.optimize import minimize

# --- Problem Formulation Functions ---
def f_func(x):
    return x[0]**2 + x[1]**2 + 2*x[2]**2 + x[3]**2 - 5*x[0] - 5*x[1] - 21*x[2] + 7*x[3]

def gvec(x):
    g1 = x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[0] - x[1] + x[2] - x[3] - 8
    g2 = x[0]**2 + 2*x[1]**2 + x[2]**2 + 2*x[3]**2 - x[0] - x[3] - 10
    g3 = 2*x[0]**2 + x[1]**2 + x[2]**2 + 2*x[0] - x[1] - x[3] - 5
    return np.array([g1, g2, g3])

def grad_f(x):
    return np.array([2*x[0] - 5, 2*x[1] - 5, 4*x[2] - 21, 2*x[3] + 7])

def jac_g(x):
    dg1 = [2*x[0] + 1, 2*x[1] - 1, 2*x[2] + 1, 2*x[3] - 1]
    dg2 = [2*x[0] - 1, 4*x[1],     2*x[2],     4*x[3] - 1]
    dg3 = [4*x[0] + 2, 2*x[1] - 1, 2*x[2],     -1]
    return np.array([dg1, dg2, dg3])

H_f = np.diag([2.0, 2.0, 4.0, 2.0])
H_g = [
    np.diag([2.0, 2.0, 2.0, 2.0]),
    np.diag([2.0, 4.0, 2.0, 4.0]),
    np.diag([4.0, 2.0, 2.0, 0.0])
]

# --- Active-Set QP Solver ---
def solve_equality_kkt(H, c, A, b, working_set):
    n = H.shape[0]
    idx = list(working_set)
    if not idx:
        return np.linalg.solve(H, -c), {}
    Aw, bw = A[idx], b[idx]
    m = len(idx)
    KKT = np.block([[H, Aw.T], [Aw, np.zeros((m, m))]])
    rhs = np.concatenate([-c, bw])
    sol = np.linalg.solve(KKT, rhs)
    return sol[:n], dict(zip(idx, sol[n:]))

def solve_qp_active_set(H, c, A, b, working_set0, max_iter=50):
    m = A.shape[0]
    working = set(working_set0)
    for _ in range(max_iter):
        dx, mu = solve_equality_kkt(H, c, A, b, working)
        residual = A @ dx - b
        candidates = [j for j in range(m) if j not in working]
        violated = [j for j in candidates if residual[j] > 1e-10]
        negative_mu = [i for i in working if mu.get(i, 0.0) < -1e-10]
        if not violated and not negative_mu:
            return dx, mu
        if violated:
            working.add(violated[np.argmax(residual[violated])])
        elif negative_mu:
            working.remove(min(negative_mu, key=lambda i: mu.get(i, 0.0)))
    raise RuntimeError("inner QP active-set method did not converge")

# --- SQP Execution Loop (kmax = 200) ---
def run_sqp(x0, max_iter=200, tol=1e-8, merit_weight=25.0):
    x = np.array(x0, dtype=float)
    mu_k = np.zeros(3)
    
    print(f"{'Iter':<5} {'f(x_k)':<10} {'Max Viol':<10} {'||grad_L||':<12} {'Step Size':<10}")
    print("-" * 55)
    
    for k in range(max_iter):
        gk, Ak, gf = gvec(x), jac_g(x), grad_f(x)
        Wk = H_f.copy()
        for j in range(3):
            Wk += mu_k[j] * H_g[j]
            
        grad_L = gf + Ak.T @ mu_k
        norm_grad_L = np.linalg.norm(grad_L)
        max_viol = max(0.0, np.max(gk))
        
        active_guess = {i for i in range(3) if gk[i] > -1e-6}
        dx, mu_dict = solve_qp_active_set(Wk, gf, Ak, -gk, active_guess)
        
        def merit(xx):
            return f_func(xx) + merit_weight * np.sum(np.maximum(0.0, gvec(xx)))
            
        t, m0 = 1.0, merit(x)
        for _ in range(30):
            if merit(x + t * dx) < m0:
                break
            t *= 0.5
            
        print(f"{k:<5} {f_func(x):<10.4f} {max_viol:<10.4f} {norm_grad_L:<12.4e} {t:<10.4f}")
        
        if np.linalg.norm(dx) < tol:
            break
        x += t * dx
        mu_k = np.zeros(3)
        for i, val in mu_dict.items():
            mu_k[i] = val
            
    active_set = [i+1 for i in range(3) if gvec(x)[i] > -1e-5]
    return x, mu_k, active_set, k

# Run Custom SQP Engine
x0 = np.zeros(4)
x_custom, mu_custom, active_custom, total_iters = run_sqp(x0)

print("\n=================== FINAL CUSTOM SQP RESULTS ===================")
print(f"Total Iterations       : {total_iters}")
print(f"Final Solution x       : {np.round(x_custom, 6)}")
print(f"Final Multipliers (mu) : {np.round(mu_custom, 4)}")
print(f"Active Constraints Set : {active_custom}")
print("=================================================================")

# --- (c) SciPy SLSQP Cross-Check Reference ---
cons_scipy = [{'type': 'ineq', 'fun': lambda x, i=i: -gvec(x)[i]} for i in range(3)]
res_scipy = minimize(f_func, x0, method='SLSQP', constraints=cons_scipy, tol=1e-8)

print("\n--- (c) SciPy SLSQP Comparison Results ---")
print(f"SciPy SLSQP Iterations  : {res_scipy.nit}")
print(f"SciPy SLSQP Solution x  : {np.round(res_scipy.x, 6)}")
print(f"SciPy SLSQP f(x*) value : {res_scipy.fun:.4f}")
