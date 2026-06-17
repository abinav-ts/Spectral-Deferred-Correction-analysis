# Exp 1.1: Explicit SDC simulation on Dahlquist equation

import numpy as np
import matplotlib.pyplot as plt
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.explicit import explicit
from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d

# SDC implementation

def run_sdc_for_dt(dt, num_nodes, num_sweeps):
    """Runs a single Explicit SDC simulation on Dahlquist equation for a specific step size."""
    
    level_params = {'dt': dt, 'restol': -1} # No tolerance limit set, runs for num_sweeps
    step_params = {'maxiter': num_sweeps + 1} # +1 for the base integrator step (predictor)
    problem_params = {'u0': 1.0, 'lambdas': np.array([-1.0])} # Non-stiff lambda - for explicit SDC
    
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT', # Radau-Right quadrature
        'num_nodes': num_nodes,
        'QE': 'EE' # Explicit Euler base integrator
    }

    description = {
        'problem_class': testequation0d, # Dahlquist: u' = -u, u(0) = 1
        'problem_params': problem_params,
        'sweeper_class': explicit, 
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params
    }

    # Single core run
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    
    Tend = 1.0
    
    u0_array = controller.MS[0].levels[0].prob.u_exact(0.0)
    uend, stats = controller.run(u0=u0_array, t0=0.0, Tend=Tend)
    
    exact_solution = np.exp(-Tend)
    error = abs(exact_solution - uend[0])
    
    return error

# Order of Accuracy Analysis

def plot_convergence():
    dt_values = [0.5, 0.25, 0.125, 0.0625, 0.03125]
    num_nodes = 4
    num_sweeps = 5 # 1 (Base Euler) + 5 Sweeps = 6th Order
    
    errors = []
    for dt in dt_values:
        err = run_sdc_for_dt(dt, num_nodes, num_sweeps)
        errors.append(err)
        print(f"dt: {dt:.5f} | Error: {err:.2e}")

    plt.figure(figsize=(9, 6))
    plt.loglog(dt_values, errors, marker='o', linestyle='-', linewidth=2, label='Explicit SDC (4 nodes, 5 sweeps)')
    
    # 6th-order reference line
    ref_line = [errors[0] * (dt / dt_values[0])**6 for dt in dt_values]
    plt.loglog(dt_values, ref_line, linestyle='--', color='gray', label='Reference Slope - 6th order')

    plt.xlabel(r'Time Step ($\Delta t$)')
    plt.ylabel(r'Absolute Error at $t=1$')
    plt.title('Explicit SDC - Basic Accuracy - Dahlquist')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    # plt.savefig('plots/01_explicit_ode_convergence.png', bbox_inches='tight', dpi=600)

if __name__ == "__main__":
    plot_convergence()
