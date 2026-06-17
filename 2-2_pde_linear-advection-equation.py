# Exp 2.2: Explicit SDC + Method of Lines simulation on Linear Advection equation

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.explicit import explicit

# Custom PDE class for spatial discretization

class AdvectionEquation1D_FD(Problem):
    """
    Custom pySDC Problem Class for the Linear Advection Equation
    Equation: u_t + u_x = 0
    Boundary Conditions: Periodic
    """
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=100, c=1.0, L=1.0):
        init = (nvars, None, np.dtype('float64'))
        super().__init__(init=init)
        self._makeAttributeAndRegister('nvars', 'c', 'L', localVars=locals(), readOnly=True)
        
        self.dx = self.L / self.nvars
        self.xvalues = np.array([i * self.dx for i in range(self.nvars)])
        
        # 3rd-Order Upwind-Biased Spatial Matrix

        I = np.eye(self.nvars)
        u_plus_1 = np.roll(I, -1, axis=0)
        u_minus_1 = np.roll(I, 1, axis=0)
        u_minus_2 = np.roll(I, 2, axis=0)
        
        D = (2.0 * u_plus_1 + 3.0 * I - 6.0 * u_minus_1 + 1.0 * u_minus_2)
        A_dense = D * (-self.c / (6.0 * self.dx))
        self.A = sp.csr_matrix(A_dense)

    def eval_f(self, u, t):
        f = self.dtype_f(self.init)
        f[:] = self.A.dot(u)
        return f

    def u_exact(self, t):
        shift = (self.xvalues - self.c * t) % self.L
        u = self.dtype_u(self.init)
        u[:] = np.exp(-100.0 * (shift - 0.5)**2)
        return u

def run_fair_comparison():
    nvars = 100  
    c = 1.0     
    L = 1.0     
    Tend = 1.0 
    
    dt_sdc = 0.008   # Large step for SDC
    dt_euler = 0.001 # Tiny step required to keep Euler stable with 3rd-order space
    
    
    # Explicit SDC solver for lumped system (large dt)

    level_params = {'dt': dt_sdc, 'restol': -1} 
    step_params = {'maxiter': 4} 
    problem_params = {'nvars': nvars, 'c': c, 'L': L}
    sweeper_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QE': 'EE'}

    description = {
        'problem_class': AdvectionEquation1D_FD,
        'problem_params': problem_params,
        'sweeper_class': explicit,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params
    }

    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    u0_array = prob.u_exact(0.0)
    
    uend_sdc, stats = controller.run(u0=u0_array, t0=0.0, Tend=Tend)
    
    # SDC Cost: (1 predictor + 3 sweeps) * 3 nodes * 125 steps = ~1500 evaluations
    sdc_evals = 4 * 3 * int(Tend / dt_sdc)

    # Standard Explicit Euler (Small dt)

    u_current_fd = u0_array[:]
    
    current_time = 0.0
    euler_steps = 0
    while current_time < Tend - 1e-10:
        current_time += dt_euler
        u_current_fd = u_current_fd + dt_euler * prob.A.dot(u_current_fd)
        euler_steps += 1

    # Euler Cost: 1 evaluation per step * 1000 steps = 1000 evaluations
    euler_evals = euler_steps

    # Calculate Errors
    u_exact_final = prob.u_exact(Tend)[:]
    error_sdc = np.linalg.norm(u_exact_final - uend_sdc[:], np.inf)
    error_euler = np.linalg.norm(u_exact_final - u_current_fd, np.inf)

    print("\n--- RESULTS ---")
    print(f"Explicit Euler Evaluations: {euler_evals} | Final Error: {error_euler:.4e}")
    print(f"Explicit SDC Evaluations:   {sdc_evals} | Final Error: {error_sdc:.4e}")

    # Plot - Error Comparison

    x_vals = prob.xvalues
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, u_exact_final, 'k-', linewidth=3, label='Exact Analytical Solution')
    plt.plot(x_vals, u_current_fd, 'r--', linewidth=2, label=f'Explicit Euler (dt={dt_euler})')
    plt.plot(x_vals, uend_sdc[:], 'b-o', markersize=4, label=f'Explicit SDC (dt={dt_sdc})')
    
    plt.xlabel(r'Spatial Domain ($x$)')
    plt.ylabel(r'Amplitude ($u$)')
    plt.title('Stable Euler vs. Large-Step SDC - Linear Advection Equation')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show
    # plt.savefig('plots/08_pde_advection.png', bbox_inches='tight', dpi=600)

if __name__ == "__main__":
    run_fair_comparison()
