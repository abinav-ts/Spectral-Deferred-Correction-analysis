# Exp 3.2: Stabilty Gain per unit Computational Cost Increase

import numpy as np
import matplotlib.pyplot as plt

from pySDC.core.collocation import CollBase
def get_collocation(num_nodes):
    return CollBase(num_nodes, tleft=0.0, tright=1.0, node_type='LEGENDRE', quad_type='RADAU-RIGHT')

# Vectorized Amplification Factor calculation

def get_amplification_factor(z_array, sweeps, num_nodes):
    coll = get_collocation(num_nodes)
    Q = coll.Qmat 
    tau = np.insert(coll.nodes, 0, 0.0)
    delta_tau = np.diff(tau)

    total_iterations = sweeps + 1 
    u = np.ones((num_nodes + 1, len(z_array)), dtype=complex) 
    
    for k in range(total_iterations):
        u_new = np.ones_like(u) 
        for m in range(num_nodes):
            Q_diff = Q[m+1, :] - Q[m, :]
            S_m_k = z_array * np.tensordot(Q_diff, u, axes=([0], [0]))
            
            f_u_new_prev = z_array * u_new[m]
            f_u_k_prev   = z_array * u[m]
            
            u_new[m+1] = u_new[m] + delta_tau[m] * (f_u_new_prev - f_u_k_prev) + S_m_k
        u = u_new
        
    return u[-1] 

def main():
    # Range of sweeps
    sweep_counts = np.arange(2, 42, 2)
    
    # Negative real axis 1D grid
    z_real = np.linspace(0, -60, 5000) 
    
    max_stable_z = []
    efficiency = []

    for sweeps in sweep_counts:
        num_nodes = sweeps+1
        
        # Calculate R(z) for all points on the real axis
        R = get_amplification_factor(z_real, sweeps, num_nodes)
        
        # Find the most negative z where |R| <= 1
        stable_mask = np.abs(R) <= 1.0 + 1e-12
        
        if np.any(stable_mask):
            # The boundary is the minimum real value that remains stable
            boundary = np.min(z_real[stable_mask])
        else:
            boundary = 0.0
            
        max_stable_z.append(np.abs(boundary))
        # Efficiency = Max Stable Limit / Total Iterations (Cost)
        efficiency.append(np.abs(boundary) / (sweeps + 1)) 
        
        print(f"Sweeps: {sweeps:2d} | Nodes: {num_nodes:2d} | Max Re(z): {boundary:.2f} | Efficiency: {efficiency[-1]:.3f}")

    # Double y-axis Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:red'
    ax1.set_xlabel('Number of SDC Sweeps (K)', fontsize=12)
    ax1.set_ylabel('Absolute Stability Limit |Re(z)|', color=color1, fontsize=12, fontweight='bold')
    ax1.plot(sweep_counts, max_stable_z, 'o-', color=color1, linewidth=2.5, label="Absolute Limit")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    ax2.set_ylabel('Efficiency (Stability Limit / Computations)', color=color2, fontsize=12, fontweight='bold')  
    ax2.plot(sweep_counts, efficiency, 's--', color=color2, linewidth=2.5, label="Efficiency")
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.suptitle("Explicit SDC: Absolute Stability Growth vs. Efficiency Plateau", fontsize=14, fontweight='bold')
    fig.tight_layout() 
    plt.show()
    # plt.savefig('plots/12_ode_stability_limit.png', bbox_inches='tight', dpi=600)
    

if __name__ == "__main__":
    main()
