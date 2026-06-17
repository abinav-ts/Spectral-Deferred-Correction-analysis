# Exp 2.3: Implicit SDC + Method of Lines simulation on Non-Linear Porous Diffusion equation

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

class NonlinearDiffusion1D(Problem):
    """
    Custom pySDC Problem Class for the Porous Medium Equation (m=1)
    Equation: u_t = (nu/2) * (u^2)_xx
    Boundary Conditions: Periodic
    """
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=100, nu=0.5, L=1.0):
        init = (nvars, None, np.dtype('float64'))
        super().__init__(init=init)
        self._makeAttributeAndRegister('nvars', 'nu', 'L', localVars=locals(), readOnly=True)
        
        self.dx = self.L / self.nvars
        self.xvalues = np.array([i * self.dx for i in range(self.nvars)])
        
        # Standard 2nd Derivative Central Matrix (Periodic)
        diagonals = [np.ones(self.nvars - 1), -2.0 * np.ones(self.nvars), np.ones(self.nvars - 1)]
        A = sp.diags(diagonals, offsets=[-1, 0, 1], shape=(self.nvars, self.nvars), format='lil')
        
        A[0, -1] = 1.0  
        A[-1, 0] = 1.0
        
        self.A = A.tocsc() / (self.dx ** 2)

    def eval_f(self, u, t):
        """Evaluates the Non-Linear RHS: f(u) = (nu/2) * A * u^2"""
        f = self.dtype_f(self.init)
        f[:] = 0.5 * self.nu * self.A.dot(u**2)
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Embedded Newton-Raphson Solver for the Implicit SDC Step.
        Solves: u - factor * f(u) - rhs = 0
        """
        u = self.dtype_u(self.init)
        u[:] = u0[:] # Initial guess for Newton
        
        tol = 1e-9
        max_iters = 50
        
        for i in range(max_iters):
            # 1. Calculate the Residual: G(u) = u - dt * (nu/2)*A*(u^2) - rhs
            F = u[:] - factor * 0.5 * self.nu * self.A.dot(u[:]**2) - rhs[:]
            
            # 2. Check for convergence
            if np.linalg.norm(F, np.inf) < tol:
                break
                
            # 3. Calculate the Jacobian: J = I - dt * nu * A * diag(u)
            # We use sp.diags to multiply the u array against the columns of A
            J = sp.eye(self.nvars, format='csc') - factor * self.nu * self.A.dot(sp.diags(u[:]))
            
            # 4. Solve the linear system for the Newton step and update
            delta_u = la.spsolve(J, F)
            u[:] = u[:] - delta_u
            
        return u

    def u_exact(self, t):
        """Initial Condition: A dry environment with a wet Gaussian pulse."""
        u = self.dtype_u(self.init)
        # We add a tiny offset (0.1) to prevent the Jacobian from becoming 
        # infinitely singular in the completely "dry" regions (u=0)
        u[:] = 0.1 + np.exp(-100.0 * (self.xvalues - 0.5)**2)
        return u

# Implicit SDC implementation

def run_nonlinear_experiment():
    nvars = 100
    nu = 0.5
    L = 1.0
    
    dt = 0.01  
    Tend = 0.02
    
    level_params = {'dt': dt, 'restol': -1}
    problem_params = {'nvars': nvars, 'nu': nu, 'L': L}
    
    # Base Implicit Euler (1 Sweep)
    desc_euler = {
        'problem_class': NonlinearDiffusion1D,
        'problem_params': problem_params,
        'sweeper_class': generic_implicit,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'IE'},
        'level_params': level_params,
        'step_params': {'maxiter': 1} 
    }
    
    # Implicit SDC (4 Sweeps)
    desc_sdc = desc_euler.copy()
    desc_sdc['step_params'] = {'maxiter': 4}

    # 3. High-Resolution Reference (Ground Truth)
    desc_ref = desc_sdc.copy()
    # Use a time step 10x smaller and double the sweeps for maximum precision
    desc_ref['level_params'] = {'dt': dt / 10.0, 'restol': -1} 
    desc_ref['step_params'] = {'maxiter': 8}

    ctrl_euler = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=desc_euler)
    u0_arr = ctrl_euler.MS[0].levels[0].prob.u_exact(0.0)
    uend_euler, _ = ctrl_euler.run(u0=u0_arr, t0=0.0, Tend=Tend)
    
    ctrl_sdc = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=desc_sdc)
    uend_sdc, _ = ctrl_sdc.run(u0=u0_arr, t0=0.0, Tend=Tend)

    ctrl_ref = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=desc_ref)
    uend_ref, _ = ctrl_ref.run(u0=u0_arr, t0=0.0, Tend=Tend)

    # Plot - Solution Comparison

    x_vals = ctrl_sdc.MS[0].levels[0].prob.xvalues
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, u0_arr[:], 'k:', linewidth=2, label='Initial Condition (T=0)')
    plt.plot(x_vals, uend_ref[:], 'k-', linewidth=2, alpha=0.7, label='Reference Solution (High-Res)')
    
    plt.plot(x_vals, uend_euler[:], 'r--', linewidth=2, label=f'Implicit Euler (dt={dt})')
    plt.plot(x_vals, uend_sdc[:], 'b-o', markersize=4, label=f'SDC 4-Sweeps (dt={dt})')
    
    plt.xlabel(r'Spatial Domain ($x$)')
    plt.ylabel(r'Amplitude ($u$)')
    plt.title(f'Implicit SDC vs Euler - Non-Linear Diffusion (T={Tend})')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    # plt.savefig('plots/09_pde_nonlinear.png', bbox_inches='tight', dpi=600)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run_nonlinear_experiment()
