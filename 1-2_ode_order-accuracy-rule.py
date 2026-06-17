# Exp 1.2: Verification of Order of Accuracy rule

# Order of accuracy = min(order of quadrature, order of base integrator + no. of sweeps)

import numpy as np
import matplotlib.pyplot as plt
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.explicit import explicit
from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d

# Variable no. of sweeps - SDC implementation

def run_sdc_sweeps(num_sweeps):
    """Runs a fixed dt SDC simulation, varying only the number of sweeps."""
    dt = 0.2  # Large dt to see the error drop
    num_nodes = 3 # Fixed at 3 --> Order of quadrature = 5
    
    level_params = {'dt': dt, 'restol': -1} 
    step_params = {'maxiter': num_sweeps + 1} # +1 for the base integrator step (predictor)
    problem_params = {'u0': 1.0, 'lambdas': np.array([-1.0])}
    
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT', 
        'num_nodes': num_nodes,
        'QE': 'EE' # Explicit Euler
    }

    description = {
        'problem_class': testequation0d,
        'problem_params': problem_params,
        'sweeper_class': explicit,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params
    }

    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    u0_array = controller.MS[0].levels[0].prob.u_exact(0.0)
    uend, stats = controller.run(u0=u0_array, t0=0.0, Tend=1.0)
    
    exact_solution = np.exp(-1.0)
    error = abs(exact_solution - uend[0])
    return error

# Order of Accuracy rule plotting

def plot_accuracy_limit():
    sweeps_range = list(range(0, 9)) # Testing 0 to 8 sweeps
    errors = []
    
    for k in sweeps_range:
        err = run_sdc_sweeps(num_sweeps=k)
        errors.append(err)
        print(f"Sweeps (K): {k} | Total pySDC Iters: {k+1} | Error: {err:.2e}")

    plt.figure(figsize=(9, 6))
    plt.semilogy(sweeps_range, errors, 'b-o', linewidth=2.5, markersize=8)
    
    # Annotations: 
    plt.axvline(x=4, color='r', linestyle='--', alpha=0.7)
    plt.text(5.2, 1e-4, 'Error Flatlines\n(Quadrature Bottleneck)', color='red', fontsize=11)
    plt.text(1.5, 1e-7, 'Error Drops Exponentially\n(Sweep Dominant)', color='blue', fontsize=11)

    plt.xlabel('No. of Correction Sweeps (K)')
    plt.ylabel(r'Absolute Error at $t=1$ (log Scale)')
    plt.title('SDC - Order of Accuracy Verification - Dahlquist')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    # plt.legend()
    
    plt.tight_layout()
    plt.show()
    # plt.savefig('plots/02_accuracy_bottleneck.png', bbox_inches='tight', dpi=600)

if __name__ == "__main__":
    plot_accuracy_limit()
