# Exp 3.1: Stable Amplification Factor Region Plot for SDC

import numpy as np
import matplotlib.pyplot as plt

from pySDC.core.collocation import CollBase
def get_collocation(num_nodes):
    return CollBase(num_nodes, tleft=0.0, tright=1.0, node_type='LEGENDRE', quad_type='RADAU-RIGHT')

def main():
    # MASSIVE grid to ensure the bubble never goes out of bounds
    x = np.linspace(-40, 5, 800)
    y = np.linspace(-40, 40, 800)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y  # Z = 800x800 matrix

    # Chosen sweeps for evaluation
    sweep_counts = [0, 2, 5, 10, 20, 30]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Explicit SDC Stability: Scaling Nodes with Sweeps (M = K/2)", fontsize=18, fontweight='bold')
    
    for idx, sweeps in enumerate(sweep_counts):
        ax = axes.flat[idx]
        
        # Dynamic Node Calculation: 2k sweeps -> k+1 nodes (2k+1 quadrature order)
        num_nodes = max(1, (sweeps // 2) + 1) 
        
        coll = get_collocation(num_nodes)
        Q = coll.Qmat 
        tau = np.insert(coll.nodes, 0, 0.0)
        delta_tau = np.diff(tau)

        total_iterations = sweeps + 1 
        
        # Vectorized sweeps
        u = np.ones((num_nodes + 1,) + Z.shape, dtype=complex) 
        
        for k in range(total_iterations):
            u_new = np.ones_like(u) 
            
            for m in range(num_nodes):
                Q_diff = Q[m+1, :] - Q[m, :]
                S_m_k = Z * np.tensordot(Q_diff, u, axes=([0], [0]))
                
                f_u_new_prev = Z * u_new[m]
                f_u_k_prev   = Z * u[m]
                
                u_new[m+1] = u_new[m] + delta_tau[m] * (f_u_new_prev - f_u_k_prev) + S_m_k
                
            u = u_new
            
        # Amplification factor = final node layer
        R = u[-1] 
        
        # Contour Plot
        ax.contour(X, Y, np.abs(R), levels=[1.0], colors=['#2b83ba'], linewidths=2.5)
        
        # Axis Scaling
        stable_mask = np.abs(R) <= 1.0
        
        if np.any(stable_mask):
            xmin, xmax = np.min(X[stable_mask]), np.max(X[stable_mask])
            ymin, ymax = np.min(Y[stable_mask]), np.max(Y[stable_mask])
            
            x_margin = max(1.0, (xmax - xmin) * 0.1)
            y_margin = max(1.0, (ymax - ymin) * 0.1)
            
            ax_xmin = min(-5, xmin - x_margin)
            ax_xmax = max(2, xmax + x_margin)
            ax_ymin = min(-5, ymin - y_margin)
            ax_ymax = max(5, ymax + y_margin)
            
            ax.set_xlim([ax_xmin, ax_xmax])
            ax.set_ylim([ax_ymin, ax_ymax])
            
            # Unstable right-half plane
            ax.fill_betweenx([ax_ymin, ax_ymax], 0, ax_xmax, color='gray', alpha=0.1) 
        
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(f"Base + {sweeps} Sweeps ({num_nodes} Nodes)", fontsize=14, fontweight='bold')
        
        if idx >= 3:
            ax.set_xlabel("Re(z)", fontsize=12)
        if idx % 3 == 0:
            ax.set_ylabel("Im(z)", fontsize=12)

    plt.tight_layout()
    plt.show()
    # plt.savefig('plots/11_ode_stability_2.png', dpi=600)

if __name__ == "__main__":
    main()
