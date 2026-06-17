# Advanced Time-Stepping Methods: Spectral Deferred Correction (SDC) Scheme

**Course:** DS289 Coursework Project 

### Team Members
* **Abinav Thangaraju Sethupathy** – BTech Mathematics and Computing
* **Vinothini S** – PhD Computational and Data Sciences

---

## Project Overview
This repository contains the codebase for experiments conducted during the DS289 coursework project. The project explores the theoretical boundaries, computational efficiency, and practical PDE applications of the Spectral Deferred Correction (SDC) time-stepping scheme. 

## Experiments
The repository is structured around a series of sequential experiments, ranging from foundational ODE proofs to advanced PDE applications using the Method of Lines. 

### Part 1: ODE Fundamentals & Accuracy 
* **Experiment 1.1: 6th-Order Accuracy Verification** A foundational test verifying that the SDC scheme successfully achieves 6th-order temporal accuracy on a standard ordinary differential equation.
* **Experiment 1.2: Order of Accuracy Rule Verification** Empirically proves the mathematical rule that each successive SDC sweep increases the formal order of accuracy by one, up to the underlying quadrature limit.
* **Experiment 1.3: Implicit SDC on Stiff ODEs (Van der Pol)** Compares the performance of explicit versus implicit SDC sweeps, demonstrating the absolute necessity of A-stable implicit formulations for highly stiff systems.

### Part 2: PDE Applications (Method of Lines)
* **Experiment 2.1: Linear Heat Equation** Applies implicit SDC to a parabolic diffusion PDE, showcasing absolute stability and error reduction at macroscopic time steps.
* **Experiment 2.2: Linear Advection Equation** Implements SDC for a hyperbolic upwind-biased system to analyze wave fidelity, phase lag, and Courant-Friedrichs-Lewy (CFL) limits.
* **Experiment 2.3: Non-Linear Porous Medium Diffusion** Evaluates a state-dependent stiff PDE by embedding a custom Newton-Raphson root-finding solver directly inside the SDC correction loop.
* **Experiment 2.4: Viscous Burgers' Equation (IMEX Splitting)** Utilizes an Implicit-Explicit (IMEX) sweeper to simultaneously handle stiff linear diffusion implicitly and non-linear advection explicitly, completely circumventing expensive Jacobian inversions.

### Part 3: Stability & Efficiency Analysis
* **Experiment 3.1: Amplification Factor Stability Plot** Systematically maps the complex-plane stability contours of the SDC scheme to identify absolute convergence boundaries and Picard iteration limits.
* **Experiment 3.2: Stability Computational Efficiency Analysis** Analyzes the real-axis stability limits to identify the explicit "efficiency plateau," mathematically proving where adding more explicit sweeps yields diminishing computational returns.

---

## Requirements & Installation

Each `.py` file in this repository is designed as a standalone script and can be executed directly. 

To run the experiments, ensure you have Python installed along with the following standard libraries:
```bash
pip install numpy matplotlib scipy pandas
```

### Installing pySDC
Most importantly, these experiments rely heavily on the `pySDC` framework. You will need to clone and install the library directly from GitHub. Ensure you are using **version 2.5.x or above**.

Run the following commands in your terminal to install it:
```bash
git clone https://github.com/Parallel-in-Time/pySDC.git
cd pySDC
pip install .
```

## Happy Journey!
