# Exp 2.1: Implicit SDC + Method of Lines simulation on Linear Heat equation

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.hooks.log_solution import LogSolution

# Custom PDE class for spatial discretization

class HeatEquation1D_FD(Problem):
    """
    Custom pySDC Problem Class for the Porous Medium Equation (m=1)
    Equation: u_t = 0.1(u^2)_xx
    Boundary Conditions: Periodic
    """
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=63, nu=0.1, L=1.0, freq=1):
        init = (nvars, None, np.dtype('float64'))
        super().__init__(init=init)
        self._makeAttributeAndRegister('nvars', 'nu', 'L', 'freq', localVars=locals(), readOnly=True)
        
        self.dx = self.L / (self.nvars + 1)
        self.xvalues = np.array([(i + 1) * self.dx for i in range(self.nvars)])
        
        # 2rd-Order Central Difference Spatial Matrix

        diagonals = [np.ones(self.nvars - 1), 
                    -2.0 * np.ones(self.nvars), 
                     np.ones(self.nvars - 1)]
        
        A = sp.diags(diagonals, [-1, 0, 1], format='csc')
        self.A = A * (self.nu / (self.dx ** 2))

    def eval_f(self, u, t):
        f = self.dtype_f(self.init)
        f[:] = self.A.dot(u)
        return f

    def solve_system(self, rhs, factor, u0, t):
        Id = sp.eye(self.nvars, format='csc')
        M = Id - factor * self.A
        u = self.dtype_u(self.init)
        u[:] = la.spsolve(M, rhs)
        return u

    def u_exact(self, t):
        u = self.dtype_u(self.init)
        u[:] = np.sin(np.pi * self.freq * self.xvalues / self.L) * \
               np.exp(-t * self.nu * (np.pi * self.freq / self.L)**2)
        return u

# Comparison of Implicit SDC and standard BTCS solution

def run_heat_equation():
    nvars = 63  
    nu = 0.1    
    L = 1.0     
    dt = 0.01 
    Tend = 0.5
    
    # Implicit SDC solver for lumped system

    level_params = {'dt': dt, 'restol': 1e-8} 
    step_params = {'maxiter': 5} 
    problem_params = {'nvars': nvars, 'nu': nu, 'L': L, 'freq': 1}
    sweeper_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'IE'}

    description = {
        'problem_class': HeatEquation1D_FD,
        'problem_params': problem_params,
        'sweeper_class': generic_implicit,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params
    }

    controller_params = {'logger_level': 30, 'hook_class': LogSolution}
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    
    prob = controller.MS[0].levels[0].prob
    u0_array = prob.u_exact(0.0)
    
    uend_sdc, stats = controller.run(u0=u0_array, t0=0.0, Tend=Tend)
    
    t_vals_sdc = [0.0]
    u_vals_sdc = [u0_array[:]]
    error_sdc = [0.0]
    
    for key, val in stats.items():
        if key.type == 'u':
            t_vals_sdc.append(key.time)
            u_vals_sdc.append(val[:])
            exact_val = prob.u_exact(key.time)[:]
            error_sdc.append(np.linalg.norm(exact_val - val[:], np.inf))

    # Standard BTCS Finite Difference Solver
    
    Id = sp.eye(nvars, format='csc')
    M_btcs = Id - dt * prob.A
    
    t_vals_fd = [0.0]
    error_fd = [0.0]
    u_current_fd = u0_array[:]
    
    current_time = 0.0
    while current_time < Tend - 1e-10:
        current_time += dt
        u_current_fd = la.spsolve(M_btcs, u_current_fd)
        
        t_vals_fd.append(current_time)
        exact_val = prob.u_exact(current_time)[:]
        error_fd.append(np.linalg.norm(exact_val - u_current_fd, np.inf))

    # Table of Errors

    avg_error_sdc = np.mean(error_sdc[1:]) 
    final_error_sdc = error_sdc[-1]

    avg_error_fd = np.mean(error_fd[1:])
    final_error_fd = error_fd[-1]

    print("\n" + "="*50)
    print("QUANTITATIVE ERROR ANALYSIS (dt = 0.01)")
    print("="*50)
    print("Standard BTCS (Implicit Euler Finite Difference):")
    print(f"  -> Average Error Over Time: {avg_error_fd:.2e}")
    print(f"  -> Final Error at T={Tend}:    {final_error_fd:.2e}")
    print("\nImplicit SDC (3 Nodes, 4 Sweeps):")
    print(f"  -> Average Error Over Time: {avg_error_sdc:.2e}")
    print(f"  -> Final Error at T={Tend}:    {final_error_sdc:.2e}")
    print("="*50 + "\n")

    # Plot - Error Comparison

    fig2 = plt.figure(figsize=(8, 6))
    plt.semilogy(t_vals_fd, error_fd, 'r--', linewidth=2, label='Standard BTCS (Implicit Euler)')
    plt.semilogy(t_vals_sdc, error_sdc, 'b-', linewidth=2.5, label='Implicit SDC (3 Nodes, 4 Sweeps)')
    
    plt.xlabel('Time (t)')
    plt.ylabel('Max Absolute Error (log Scale)')
    plt.title(f'Implicit SDC vs. BTCS - Heat Equation, dt = {dt}')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # fig2.savefig('plots/05_pde_heat_error.png', bbox_inches='tight', dpi=600)
    

if __name__ == "__main__":
    run_heat_equation()
