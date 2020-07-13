Name: Dustin Mcafee
Finite difference method to approximate solution of Poisson Problem using Iterative Methods

To execute Iteration Methods:
./one.py N

Where (N-1)x(N-1) is the number of dimensions of the stiffness matrix A in Ax=b,
A and b and determined from the Finite Difference Method for approximating the Poisson Distribution Problem

The output is number of iterations until convergence (with tolerance 1e-5) for each method: Jacobi, Gauss-Seidel, SOR, Conjugate Gradient, and Steepest Descent.

The output files are:
output/Jacobi_Error_N.csv
output/Gauss_Seidel_Error_N.csv
output/SOR_Error_N.csv
output/Conjugate_Gradient_Error_N.csv
output/Steepest_Descent_Error_N.csv
