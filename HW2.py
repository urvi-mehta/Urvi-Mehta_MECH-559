import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

n, p = 6, 500

# TODO: compute the full-factorial grid size needed to match p's 1-D resolution
grid_full = p**n

# TODO: compute how many levels/dimension a budget of p runs can afford
levels_afford = int(np.floor(p ** (1 / n)))

# TODO: print both results and compare grid_full to p
print(f"Full-factorial grid size needed: {grid_full:.4e} ({grid_full} runs)")
print(f"Levels per dimension affordable with p={p} runs: {levels_afford}")
print(f"Comparison: grid_full is {grid_full / p:.2e} times larger than p ({p}).")
# ==========================================
# METRIC HELPER FUNCTIONS
# ==========================================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# ==========================================
# QUESTION 1: THE CURSE OF DIMENSIONALITY
# ==========================================
print("=== QUESTION 1: THE CURSE OF DIMENSIONALITY ===")
n, p = 6, 500

grid_full = p**n
levels_afford = int(np.floor(p ** (1 / n)))

print(f"Full-factorial grid size needed: {grid_full:.4e} ({grid_full} runs)")
print(f"Levels per dimension affordable with p={p} runs: {levels_afford}")
print(f"Comparison: grid_full is {grid_full / p:.2e} times larger than p ({p}).\n")


# ==========================================
# DATALOADING & TRANSFORMATION (Shared)
# ==========================================
csv_path = 'Urvi-Mehta_MECH-559\aero_naca4_airfoil_panel.csv'
df = pd.read_csv(csv_path)
df["log10_reynolds"] = np.log10(df["reynolds"])

feature_cols = [
    "max_camber_m",
    "camber_pos_p",
    "thickness_t",
    "alpha_deg",
    "mach",
    "log10_reynolds",
]
target_col = "CL"

train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

Xtr = train_df[feature_cols].values
ytr = train_df[target_col].values
Xte = test_df[feature_cols].values
yte = test_df[target_col].values

# Min-Max Scaling bounded strictly by training bounds
lo = Xtr.min(axis=0)
hi = Xtr.max(axis=0)


def scale(X):
    return (X - lo) / (hi - lo)


Xtr_s, Xte_s = scale(Xtr), scale(Xte)

# Define shared 5-fold cross-validation engine
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# ==========================================
# QUESTION 2: BASELINE OLS LINEAR REGRESSION
# ==========================================
print("=== QUESTION 2: LINEAR REGRESSION BASELINE ===")

Ztr_linear = np.column_stack([np.ones(len(Xtr)), Xtr])
Zte_linear = np.column_stack([np.ones(len(Xte)), Xte])

w_linear = np.linalg.solve(Ztr_linear.T @ Ztr_linear, Ztr_linear.T @ ytr)
yhat_te_linear = Zte_linear @ w_linear

print("--- Direct Normal Equations Solution ---")
print(f"Intercept (w0): {w_linear[0]:.6f}")
print(f"Coefficients (w1 to w6): \n{w_linear[1:]}")
print(f"Test RMSE: {rmse(yte, yhat_te_linear):.6f}")
print(f"Test R^2: {r2(yte, yhat_te_linear):.6f}\n")

lr = LinearRegression(fit_intercept=True).fit(Xtr, ytr)
print("--- Scikit-Learn Verification ---")
print(f"scikit-learn intercept: {lr.intercept_:.6f}")
print(f"scikit-learn coefficients: \n{lr.coef_}")

w_sklearn = np.concatenate(([lr.intercept_], lr.coef_))
if np.allclose(w_linear, w_sklearn, atol=1e-8):
    print(
        "[SUCCESS] Your custom first-principles solution matches scikit-learn exactly!\n"
    )
else:
    print(
        "[WARNING] Discrepancy detected between custom weights and scikit-learn.\n"
    )


# ==========================================
# QUESTION 3: SCALED QUADRATIC RIDGE REGRESSION
# ==========================================
print("=== QUESTION 3: SCALED QUADRATIC RIDGE REGRESSION ===")


def make_quad(Xraw):
    return np.column_stack([Xraw, Xraw**2])


Ztr_quad_raw = np.column_stack([np.ones(len(Xtr)), make_quad(Xtr)])
kappa_unscaled = np.linalg.cond(Ztr_quad_raw.T @ Ztr_quad_raw)
print(f"Condition number on UNSCALED features: {kappa_unscaled:.5e}")

Ztr_quad_s = np.column_stack([np.ones(len(Xtr_s)), make_quad(Xtr_s)])
Zte_quad_s = np.column_stack([np.ones(len(Xte_s)), make_quad(Xte_s)])

kappa_scaled = np.linalg.cond(Ztr_quad_s.T @ Ztr_quad_s)
print(f"Condition number on SCALED features:   {kappa_scaled:.5e}\n")

lambdas = [0, 1e-4, 1e-2, 1, 100]
q3_cv_rmses = []

for lam in lambdas:
    fold_rmses = []
    for tr_idx, va_idx in kf.split(Ztr_quad_s):
        Z_f, y_f = Ztr_quad_s[tr_idx], ytr[tr_idx]
        Z_val, y_val = Ztr_quad_s[va_idx], ytr[va_idx]

        w_ridge = np.linalg.solve(
            Z_f.T @ Z_f + lam * np.eye(Z_f.shape[1]), Z_f.T @ y_f
        )
        yhat_val = Z_val @ w_ridge
        fold_rmses.append(rmse(y_val, yhat_val))

    q3_cv_rmses.append(np.mean(fold_rmses))

best_lambda = lambdas[np.argmin(q3_cv_rmses)]
print(f"Identified Best Lambda via Cross-Validation: {best_lambda}")

results_quad = {}
for lam in [0, best_lambda]:
    lhs = Ztr_quad_s.T @ Ztr_quad_s + lam * np.eye(Ztr_quad_s.shape[1])
    rhs = Ztr_quad_s.T @ ytr
    w_full = np.linalg.solve(lhs, rhs)

    yhat_te_quad = Zte_quad_s @ w_full
    results_quad[lam] = {
        "RMSE": rmse(yte, yhat_te_quad),
        "R2": r2(yte, yhat_te_quad),
        "yhat": yhat_te_quad,
    }

print("\n--- Final Test Set Comparison (Quadratic) ---")
print(
    f"Pure Quadratic (λ = 0):       Test RMSE = {results_quad[0]['RMSE']:.5f}, R² = {results_quad[0]['R2']:.5f}"
)
print(
    f"Regularized (λ = {best_lambda}): Test RMSE = {results_quad[best_lambda]['RMSE']:.5f}, R² = {results_quad[best_lambda]['R2']:.5f}\n"
)


# ==========================================
# QUESTION 4: RADIAL BASIS FUNCTION (RBF) INTERPOLATOR
# ==========================================
print("=== QUESTION 4: RADIAL BASIS FUNCTION (RBF) INTERPOLATOR ===")
spreads = [1e-3, 1e-2, 1e-1, 1, 10, 100]
q4_cv_rmses = []

for lam in spreads:
    fold_rmses = []
    eps = np.sqrt(lam)

    for tr_idx, va_idx in kf.split(Xtr_s):
        Xtr_f, ytr_f = Xtr_s[tr_idx], ytr[tr_idx]
        Xva_f, yva_f = Xtr_s[va_idx], ytr[va_idx]

        rbfi = RBFInterpolator(Xtr_f, ytr_f, kernel="gaussian", epsilon=eps)
        fold_rmse = rmse(yva_f, rbfi(Xva_f))
        fold_rmses.append(fold_rmse)

    q4_cv_rmses.append(np.mean(fold_rmses))

best_spread = spreads[np.argmin(q4_cv_rmses)]
print(f"Identified Best Spread Parameter (λ): {best_spread}")

best_eps = np.sqrt(best_spread)
rbfi_best = RBFInterpolator(Xtr_s, ytr, kernel="gaussian", epsilon=best_eps)
yhat_te_rbf = rbfi_best(Xte_s)

print("\n--- Final RBF Test Set Results ---")
print(f"Test RMSE: {rmse(yte, yhat_te_rbf):.5f}")
print(f"Test R²:   {r2(yte, yhat_te_rbf):.5f}\n")


# ==========================================
# QUESTION 5: MULTI-LAYER PERCEPTRON (MLP) REGRESSOR
# ==========================================
print("=== QUESTION 5: NEURAL NETWORK REGRESSOR ===")
widths = [2, 8, 32, 64]
q5_cv_rmses = []

for width in widths:
    fold_rmses = []

    for tr_idx, va_idx in kf.split(Xtr_s):
        Xtr_f, ytr_f = Xtr_s[tr_idx], ytr[tr_idx]
        Xva_f, yva_f = Xtr_s[va_idx], ytr[va_idx]

        mlp = MLPRegressor(
            hidden_layer_sizes=(width,),
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
        )
        mlp.fit(Xtr_f, ytr_f)
        fold_rmse = rmse(yva_f, mlp.predict(Xva_f))
        fold_rmses.append(fold_rmse)

    q5_cv_rmses.append(np.mean(fold_rmses))

best_width = widths[np.argmin(q5_cv_rmses)]
print(f"Optimal Hidden Layer Width: {best_width} neurons")

best_mlp = MLPRegressor(
    hidden_layer_sizes=(best_width,),
    solver="lbfgs",
    max_iter=5000,
    random_state=42,
)
best_mlp.fit(Xtr_s, ytr)
yhat_te_nn = best_mlp.predict(Xte_s)

print("\n--- Final Neural Network Test Set Results ---")
print(f"Test RMSE: {rmse(yte, yhat_te_nn):.5f}")
print(f"Test R²:   {r2(yte, yhat_te_nn):.5f}\n")


# =======================================================================
# QUESTION 6: KRIGING / GAUSSIAN PROCESS REGRESSOR
# =======================================================================
print("=== QUESTION 6: GAUSSIAN PROCESS REGRESSOR ===")


def corr_matrix(X1, X2, theta):
    d2 = ((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1)
    return np.exp(-theta * d2)


# Part A: Small Subsample Condition Diagnostics
rng_sub = np.random.default_rng(559)
sub_idx = rng_sub.choice(len(Xtr_s), size=25, replace=False)
Xsub = Xtr_s[sub_idx]
ysub = ytr[sub_idx]

thetas = [1e-3, 1.0, 100.0]
print("--- Part A Diagnostics ---")
for theta in thetas:
    R_pure = corr_matrix(Xsub, Xsub, theta)
    kappa_pure = np.linalg.cond(R_pure)
    print(f"Theta = {theta:7.3f} -> Matrix Condition Number κ(R): {kappa_pure:.5e}")

worst_theta = 1e-3
R_worst = corr_matrix(Xsub, Xsub, worst_theta)
kappa_before = np.linalg.cond(R_worst)

nugget_val = 1e-6
R_fixed = R_worst + nugget_val * np.eye(len(Xsub))
kappa_after = np.linalg.cond(R_fixed)

print("\n--- Nugget Regularization Effect ---")
print(f"Condition number BEFORE nugget addition (theta={worst_theta}): {kappa_before:.5e}")
print(f"Condition number AFTER nugget addition  (theta={worst_theta}): {kappa_after:.5e}")

# Part B: Full Scale Fit
kernel = RBF(length_scale=np.ones(6), length_scale_bounds=(1e-2, 1e3))
gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    alpha=1e-10,
    n_restarts_optimizer=5,
    random_state=559,
)
gp.fit(Xtr_s, ytr)

yhat_gp, std_gp = gp.predict(Xte_s, return_std=True)
test_rmse_gp = rmse(yte, yhat_gp)
test_r2_gp = r2_score(yte, yhat_gp)

print("\n--- Part B (Full-Scale GP Results) ---")
print(f"Test RMSE: {test_rmse_gp:.5f}")
print(f"Test R^2:  {test_r2_gp:.5f}")

D = cdist(Xte_s, Xtr_s)
min_dist = D.min(axis=1)

i_near = np.argmin(min_dist)
i_far = np.argmax(min_dist)

print(f"\nNearest test point index: {i_near:3d} | Distance: {min_dist[i_near]:.4f} | Predicted Std Dev: {std_gp[i_near]:.6f}")
print(f"Farthest test point index: {i_far:3d} | Distance: {min_dist[i_far]:.4f} | Predicted Std Dev: {std_gp[i_far]:.6f}\n")


# ==========================================
# SURROGATE MODEL SUMMARY REPORT
#==========================================
rows = [
("Polynomial (quad+ridge)", f"lambda={best_lambda}", results_quad[best_lambda]["RMSE"], results_quad[best_lambda]["R2"], "No"),
("RBF", f"spread={best_spread}", rmse(yte, yhat_te_rbf), r2(yte, yhat_te_rbf), "No"),
("ANN", f"width={best_width}", rmse(yte, yhat_te_nn), r2(yte, yhat_te_nn), "No"),
("Kriging/GP", "theta (MLE)", test_rmse_gp, test_r2_gp, "Yes (std)"),
]
df_summary = pd.DataFrame(rows, columns=["Family", "Knob", "Test RMSE", "Test R2", "Native UQ"])
print("--- Surrogate Model Comparison Summary ---")
print(df_summary.to_string(index=False))
#==========================================
#INDEPENDENT SEPARATE PLOTS PIPELINE
#==========================================
#--- PLOT 1: Question 2 Scatter Performance Evaluation ---
plt.figure(figsize=(6.5, 5))
plt.scatter(yte, yhat_te_linear, alpha=0.6, color="#2B3A67", label="Linear OLS Baseline")
plt.scatter(yte, yhat_gp, alpha=0.4, color="#E65F2B", label="Kriging/GP Model Predictions")
ideal_line = [yte.min(), yte.max()]
plt.plot(ideal_line, ideal_line, color="black", linestyle="--", label="Ideal Prediction")
plt.xlabel("True Lift Coefficient ($C_L$)", fontsize=11)
plt.ylabel("Predicted Lift Coefficient ($\hat{C}_L$)", fontsize=11)
plt.title("Q2: Baseline vs. Non-Linear Model Predictions", fontsize=12, fontweight="bold")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
#--- PLOT 2: Question 3 Ridge Hyperparameter Path ---
plt.figure(figsize=(6.5, 5))
plt.plot(lambdas, q3_cv_rmses, marker="o", color="#D1495B", lw=2)
plt.xscale("symlog", linthresh=1e-4)
plt.xlabel(r"Ridge Penalty $\lambda$", fontsize=11)
plt.ylabel("5-Fold CV Mean RMSE", fontsize=11)
plt.title("Q3: Ridge Polynomial CV Path", fontsize=12, fontweight="bold")
plt.grid(True, alpha=0.3)
#--- PLOT 3: Question 4 RBF Hyperparameter Curve ---
plt.figure(figsize=(6.5, 5))
plt.loglog(spreads, q4_cv_rmses, marker="o", linestyle="-", color="#2A9D8F", lw=2)
plt.xlabel(r"RBF Spread Parameter $\lambda$ (log scale)", fontsize=11)
plt.ylabel("5-Fold CV Mean RMSE (log scale)", fontsize=11)
plt.title("Q4: RBF Gaussian Kernel Spread Path", fontsize=12, fontweight="bold")
plt.grid(True, which="both", ls="--", alpha=0.3)
#--- PLOT 4: Question 5 Neural Network Hidden Layer Grid Search ---
plt.figure(figsize=(6.5, 5))
plt.plot(widths, q5_cv_rmses, marker="o", linestyle="-", color="#3F88C5", lw=2)
plt.xscale("log", base=2)
plt.xticks(widths)
plt.gca().set_xticklabels([str(w) for w in widths])
plt.xlabel("Hidden Layer Width (Neurons - Log2 Scale)", fontsize=11)
plt.ylabel("5-Fold CV Mean RMSE", fontsize=11)
plt.title("Q5: MLP Hidden Layer Scaling Path", fontsize=12, fontweight="bold")
plt.grid(True, which="both", ls="--", alpha=0.3)
#--- PLOT 5: Question 6 Kriging Uncertainty Verification ---
plt.figure(figsize=(6.5, 5))
plt.scatter(min_dist, std_gp, color="#00798C", alpha=0.6, edgecolors="w", linewidths=0.5)
plt.axvline(min_dist[i_near], color="#2A9D8F", linestyle="--", label="Nearest Test Point")
plt.axvline(min_dist[i_far], color="#D1495B", linestyle="--", label="Farthest Test Point")
plt.xlabel("Distance to Nearest Training Point (6-D Scaled Space)", fontsize=11)
plt.ylabel("Predicted Kriging Standard Deviation ($\sigma_{pred}$)", fontsize=11)
plt.title("Q6: Kriging Uncertainty Quantification vs. Distance", fontsize=12, fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
#Render all standalone windows cleanly
plt.show()