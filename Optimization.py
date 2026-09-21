import numpy as np
from scipy.optimize import minimize

rho_mat = 2700.0   # Density of Aluminum 6061-T6 (kg/m^3)
E = 70e9           # Young's Modulus (Pa)
sigma_y = 200e6     # Yield Strength with safety factor (Pa)
m_0 = 1.2          # Fixed central mass (kg)
r = 0.006          # Arm cross-section radius (m)
rho = 1.225        # Air density at sea level (kg/m^3)
P_0 = 150.0        # Max electrical power per motor (W)
n = 0.60         # Power efficiency
g = 9.81           # Acceleration due to gravity (m/s^2)
k = 2.0            # Required thrust margin factor
delta_max = 0.01   # Max allowable arm deflection (m)

#OPTIMIZATION FUNCTIONS

def objective(x):
    """
    x[0] = R (Rotor radius)
    x[1] = L (Arm length)
    """
    R, L = x[0], x[1]
    return 4 * rho_mat * np.pi * (r**2) * L

def T(R):

    return (n * P_0 * np.sqrt(2 * rho * np.pi) * R)**(2/3)

def constraint_stress(x):
    """g1: Max allowable stress constraint (sigma_y - sigma >= 0)"""
    R, L = x[0], x[1]
    sigma = (4 * T(R) * L) / (np.pi * (r**3))
    return sigma_y - sigma

def constraint_deflection(x):
    """g2: Max allowable deflection constraint (delta_max - delta >= 0)"""
    R, L = x[0], x[1]
    delta = (4 * T(R) * (L**3)) / (3 * np.pi * E * (r**4))
    return delta_max - delta

def constraint_thrust_margin(x):
    """g3: Thrust margin requirement (4T - k*m_tot*g >= 0)"""
    R, L = x[0], x[1]
    m_tot = m_0 + 4 * rho_mat * np.pi * (r**2) * L
    return 4 * T(R) - k * m_tot * g

def constraint_clearance(x):
    """g4: Rotor-clearance condition (L/sqrt(2) - R >= 0)"""
    R, L = x[0], x[1]
    return L / np.sqrt(2) - R

# Boundary Conditions: [R_min, R_max], [L_min, L_max]

bounds = [(0.02, 1), (0.02, 1)]

# Format constraints into scipy dictionary format
constraints = [
    {'type': 'ineq', 'fun': constraint_stress},
    {'type': 'ineq', 'fun': constraint_deflection},
    {'type': 'ineq', 'fun': constraint_thrust_margin},
    {'type': 'ineq', 'fun': constraint_clearance}
]

# Initial guess for [R, L]
x_initial = [0.15, 0.3]

# Run Sequential Least Squares Programming (SLSQP) optimization
result = minimize(
    objective, 
    x_initial, 
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints
)

if result.success:
    R_opt, L_opt = result.x[0], result.x[1]
    W_opt = result.fun
    total_mass = m_0 + W_opt
    
    print("--- Optimization Successful ---")
    print(f"Optimal Rotor Radius (R*): {R_opt:.4f} m")
    print(f"Optimal Arm Length   (L*): {L_opt:.4f} m")
    print(f"Minimum Total Airframe Mass   (W*): {W_opt:.4f} kg")
    print(f"Minimum Total Mass   (W*): {total_mass:.4f} kg")
    print(f"Stress Margin:     {constraint_stress(result.x):.2e} Pa")
    print(f"Deflection Margin: {constraint_deflection(result.x):.2e} m")
    print(f"Thrust Margin:     {constraint_thrust_margin(result.x):.2f} N")
    print(f"Clearance Margin:  {constraint_clearance(result.x):.4f} m")
else:
    print("Optimization failed:", result.message)