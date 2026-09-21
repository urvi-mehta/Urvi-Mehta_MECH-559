import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Fixed System Parameters ---
rho_mat = 2700.0   # Density of Aluminum 6061-T6 (kg/m^3)
E = 70e9           # Young's Modulus (Pa)
sigma_y = 200e6    # Yield Strength with safety factor (Pa)
m_0 = 1.2          # Base drone body weight (kg)
r = 0.006          # Arm radius (m)
rho = 1.225        # Air density at sea level (kg/m^3)
eta = 0.60         # Power efficiency
g = 9.81           # Acceleration due to gravity (m/s^2)
delta_max = 0.01   # Max deflection (m)

def solve_drone_optimization(P_0, k):
    """Solves for optimized dimensions and mass given a specific power and thrust margin."""
    def objective(x):
        return 4 * rho_mat * np.pi * (r**2) * x[1]  # Minimize total arm mass W(L)

    def T(R):
        return (eta * P_0 * np.sqrt(2 * rho * np.pi) * R)**(2/3)

    # Scipy Constraints: function(x) >= 0 represents physical inequalities
    constraints = [
        {'type': 'ineq', 'fun': lambda x: sigma_y - (4 * T(x[0]) * x[1]) / (np.pi * (r**3))},          # Stress
        {'type': 'ineq', 'fun': lambda x: delta_max - (4 * T(x[0]) * (x[1]**3)) / (3 * np.pi * E * (r**4))}, # Deflection
        {'type': 'ineq', 'fun': lambda x: 4 * T(x[0]) - k * (m_0 + 4 * rho_mat * np.pi * (r**2) * x[1]) * g}, # Thrust Margin
        {'type': 'ineq', 'fun': lambda x: x[1] / np.sqrt(2) - x[0]}                                    # Clearance
    ]
    
    bounds = [(0.01, 1.0), (0.01, 1.0)]
    res = minimize(objective, [0.15, 0.3], bounds=bounds, constraints=constraints, method='SLSQP')
    return res.x[0], res.x[1], res.fun

# --- Sweeping Parameters for Parts d & e ---
P0_values = [100.0, 150.0, 200.0]
k_values = [1.5, 1.8, 2.1, 2.4, 2.7]

# Initialize 1-row, 2-column figure layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Loop through parameters to compute values and draw curves
for idx, P_0 in enumerate(P0_values):
    mass_results = []
    R_results = []
    L_results = []
    
    for k in k_values:
        R_opt, L_opt, W_opt = solve_drone_optimization(P_0, k)
        mass_results.append(W_opt)
        R_results.append(R_opt)
        L_results.append(L_opt)
        
    # Plot 1: Optimal Arm Mass vs. k
    ax1.plot(k_values, mass_results, marker='o', linewidth=2, color=colors[idx], label=f'$P_0 = {int(P_0)}\\text{{ W}}$')
    
    # Plot 2: Individual dimensions R* and L* vs. k
    ax2.plot(k_values, R_results, marker='s', linestyle='--', linewidth=1.5, color=colors[idx], label=f'$R^*\\,(P_0={int(P_0)}\\text{{W}})$')
    ax2.plot(k_values, L_results, marker='^', linestyle='-', linewidth=1.5, color=colors[idx], label=f'$L^*\\,(P_0={int(P_0)}\\text{{W}})$')

ax1.set_xlabel('Thrust Margin Factor ($k$)', fontsize=11)
ax1.set_ylabel('Optimal Arm Mass $W^*$ (kg)', fontsize=11)
ax1.set_title('Optimal Arm Mass vs. Thrust Margin', fontsize=12, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(title="Motor Rated Power")

ax2.set_xlabel('Thrust Margin Factor ($k$)', fontsize=11)
ax2.set_ylabel('Optimal Dimension Value (m)', fontsize=11)
ax2.set_title('Optimal Geometry ($R^*$ & $L^*$) vs. Thrust Margin', fontsize=12, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(title="Design Elements", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()

# Save complete file to your computer directory
plt.savefig('drone_dimension_trends.png', dpi=300, bbox_inches='tight')
