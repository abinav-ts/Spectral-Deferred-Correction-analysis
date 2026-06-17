# Exp 1.3: Implicit SDC solution for van der Pol's equation

"""van der Pol - x'' = (mu (1-x^2) x') - x"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
from pySDC.implementations.hooks.log_solution import LogSolution

def run_vanderpol_sdc(mu, dt, Tend, num_sweeps):
    """Solves van der Pol equation with a single Implicit SDC run"""
    level_params = {'dt': dt, 'restol': -1} 
    step_params = {'maxiter': num_sweeps + 1} # +1 for the base integrator step (predictor)
    # Newton-Raphson method to handle non-linearity
    problem_params = {'u0': (2.0, 0.0), 'mu': mu, 'newton_tol': 1e-9, 'newton_maxiter': 200}
    # Implicit Euler base integrator with 3 quadrature nodes
    sweeper_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'IE'}

    description = {
        'problem_class': vanderpol,
        'problem_params': problem_params,
        'sweeper_class': generic_implicit,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params
    }

    controller = controller_nonMPI(
        num_procs=1, 
        controller_params={'logger_level': 30, 'hook_class': LogSolution}, 
        description=description
    )
    
    u0_array = controller.MS[0].levels[0].prob.u_exact(0.0)
    uend, stats = controller.run(u0=u0_array, t0=0.0, Tend=Tend)
    
    t_vals = [0.0]
    u1_vals = [2.0] 
    
    for key, val in stats.items():
        if key.type == 'u':
            t_vals.append(key.time)
            u1_vals.append(val[0])
            
    return np.array(t_vals), np.array(u1_vals)

def get_scipy_reference(mu, Tend):
    """Generates a highly accurate reference solution to van der Pol's oscillator using SciPy."""
    def vdp_deriv(t, y):
        return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]
    
    sol = solve_ivp(vdp_deriv, [0, Tend], [2.0, 0.0], method='Radau', rtol=1e-10, atol=1e-10, dense_output=True)
    return sol

# Comparison betweeen Implicit Euler and Implicit SDC (4th order)

def plot_comparison():
    mu = 20.0
    dt = 0.04
    Tend = 100.0
    
    t_euler, y_euler = run_vanderpol_sdc(mu, dt, Tend, num_sweeps=0)
    t_sdc, y_sdc = run_vanderpol_sdc(mu, dt, Tend, num_sweeps=3)
    
    sol_ref = get_scipy_reference(mu, Tend)
    
    zoom_start = 0.0
    zoom_end = 100.0
    
    t_fine = np.linspace(zoom_start, zoom_end, 1000)
    y_ref_fine = sol_ref.sol(t_fine)[0]
    
    # Plot - Solution Comparison
  
    plt.figure(figsize=(16, 6))
    
    plt.plot(t_fine, y_ref_fine, 'k-', linewidth=3, label='Reference (SciPy Radau, tol=1e-10)')
    plt.plot(t_euler, y_euler, 'r--', markersize=4, label='Implicit Euler (Base Integrator)')
    plt.plot(t_sdc, y_sdc, 'g--', markersize=4, label='Implicit SDC (Base + 3 Sweeps)')
    
    plt.xlim(zoom_start, zoom_end)
    plt.ylim(-3, 2.5)
    
    plt.xlabel('Time (t)')
    plt.ylabel('Position ($y_1$)')
    plt.title(f'Implicit SDC vs Euler - Comparison - Van der Pol, $\mu={mu}$, $\Delta t={dt}$')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    # plt.savefig('plots/03_vanderpol_comparison.png', bbox_inches='tight', dpi=600)
    

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    plot_comparison()
