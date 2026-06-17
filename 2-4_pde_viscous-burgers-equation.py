# Exp 2.4: IMEX SDC + Method of Lines simulation on Viscous Burgers' Equation

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

# Custom PDE class for spatial discretization (Implicit and Explicit done separately)

class ViscousBurgers1D(Problem):
    """
    Custom pySDC Problem Class for the 1D Viscous Burgers' Equation
    Equation: u_t + uu_x = nu * u_xx
    IMEX Split: f_expl = -uu_x, f_impl = nu * u_xx
    Boundary Conditions: Periodic
    """
    dtype_u = mesh
    dtype_f = imex_mesh 

    def __init__(self, nvars=100, nu=0.02, L=1.0):
        init = (nvars, None, np.dtype('float64'))
        super().__init__(init=init)
        self._makeAttributeAndRegister('nvars', 'nu', 'L', localVars=locals(), readOnly=True)
        
        self.dx = self.L / self.nvars
        self.xvalues = np.array([i * self.dx for i in range(self.nvars)])
        
        # 1. Implicit Diffusion Matrix D_xx (Central Difference)
        diag_xx = [np.ones(self.nvars - 1), -2.0 * np.ones(self.nvars), np.ones(self.nvars - 1)]
        D_xx = sp.diags(diag_xx, offsets=[-1, 0, 1], shape=(self.nvars, self.nvars), format='lil')
        D_xx[0, -1] = 1.0  
        D_xx[-1, 0] = 1.0
        self.D_xx = D_xx.tocsc() / (self.dx ** 2)

        # 2. Explicit Advection Matrix D_x (Central Difference)
        diag_x = [-0.5 * np.ones(self.nvars - 1), 0.5 * np.ones(self.nvars - 1)]
        D_x = sp.diags(diag_x, offsets=[-1, 1], shape=(self.nvars, self.nvars), format='lil')
        D_x[0, -1] = -0.5 
        D_x[-1, 0] = 0.5
        self.D_x = D_x.tocsc() / self.dx

    def eval_f(self, u, t):
        f = self.dtype_f(self.init)
        f.impl[:] = self.nu * self.D_xx.dot(u[:])
        f.expl[:] = -u[:] * self.D_x.dot(u[:])
        return f

    def solve_system(self, rhs, factor, u0, t):
        J = sp.eye(self.nvars, format='csc') - factor * self.nu * self.D_xx
        u = self.dtype_u(self.init)
        u[:] = la.spsolve(J, rhs)
        return u

    def u_exact(self, t):
        u = self.dtype_u(self.init)
        u[:] = np.sin(2.0 * np.pi * self.xvalues / self.L)
        return u

# Comparison between IMEX SDC and BTCS

def run_burgers_comparison():
    nvars = 100
    nu = 0.02
    L = 1.0
    dt = 0.015
    Tend = 0.25
    
    level_params = {'dt': dt, 'restol': -1}
    problem_params = {'nvars': nvars, 'nu': nu, 'L': L}
    
    # IMEX SDC
    description = {
        'problem_class': ViscousBurgers1D,
        'problem_params': problem_params,
        'sweeper_class': imex_1st_order, 
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'IE'},
        'level_params': level_params,
        'step_params': {'maxiter': 4}
    }

    ctrl_sdc = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = ctrl_sdc.MS[0].levels[0].prob
    u0_arr = prob.u_exact(0.0)
    uend_sdc, _ = ctrl_sdc.run(u0=u0_arr, t0=0.0, Tend=Tend)

    # High-Resolution Reference (Ground Truth)

    desc_ref = description.copy()
    desc_ref['level_params'] = {'dt': dt / 10.0, 'restol': -1} 
    desc_ref['step_params'] = {'maxiter': 8}
    ctrl_ref = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=desc_ref)
    uend_ref, _ = ctrl_ref.run(u0=u0_arr, t0=0.0, Tend=Tend)

    # Fully Implicit BTCS via Newton-Raphson
    
    u_btcs = u0_arr[:]
    num_steps = int(Tend / dt)
    
    for step in range(num_steps):
        u_old = u_btcs[:]
        u_new = u_old[:]

        # Newton Iteration for the non-linear implicit step
        for i in range(50):
            # Residual: F = u_new - u_old - dt*nu*D_xx*u_new + dt*u_new*(D_x*u_new)
            F = u_new - u_old - dt * nu * prob.D_xx.dot(u_new) + dt * u_new * prob.D_x.dot(u_new)
            
            if np.linalg.norm(F, np.inf) < 1e-9:
                break
                
            # Jacobian: J = I - dt*nu*D_xx + dt*diag(D_x*u_new) + dt*diag(u_new)*D_x
            J = sp.eye(nvars, format='csc') - dt * nu * prob.D_xx \
                + dt * sp.diags(prob.D_x.dot(u_new)) \
                + dt * sp.diags(u_new).dot(prob.D_x)
                
            delta_u = la.spsolve(J, F)
            u_new = u_new - delta_u
            
        u_btcs = u_new[:]

    # Plot - Solution Comparison

    x_vals = prob.xvalues
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, uend_ref[:], 'k-', linewidth=2, alpha=0.7, label='Reference Solution (High-Res)')
    plt.plot(x_vals, u_btcs, 'r--', linewidth=2, label=f'Fully Implicit BTCS (dt={dt})')
    plt.plot(x_vals, uend_sdc[:], 'b-o', markersize=4, label=f'IMEX SDC 4-Iters (dt={dt})')
    
    plt.xlabel(r'Spatial Domain ($x$)')
    plt.ylabel(r'Amplitude ($u$)')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    # plt.savefig('plots/10_pde_viscous_1.png', bbox_inches='tight', dpi=600)
    

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run_burgers_comparison()
