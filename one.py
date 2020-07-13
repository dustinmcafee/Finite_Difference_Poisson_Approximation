#!/usr/bin/python3
"""
Name: Dustin Mcafee
Finite difference method to approximate solution of Poisson Problem using Iterative Methods
"""
import numpy as np
import os
import sys
import operator
import math
import operator

def cmp(x, y):
	for j in range(len(x)):
		if (not (x[j] == y[j])):
			return -1
	return 0

def jacobi(b,N,x,iter,tol,actual):
	# While not Converged
	xs = np.array(x)
	x_last = x
	k = 1
	change = np.inf
	while(k <= iter and change > tol):
		res = []
		for i in range(N):
			if(i == 0):
				res.append(float(0.5 * (b[i] + x_last[i+1])))
			elif(not(i == N-1)):
				res.append(float(0.5 * (b[i] + x_last[i-1] + x_last[i+1])))
			else:
				res.append(float(0.5 * (b[i] + x_last[i-1])))
		change = np.linalg.norm((np.array(actual) - np.array(res)), ord=np.inf)
		#change = np.abs(np.array(x_last) - np.array(res)).max()
		if(change <= tol):
			print("Jacobi Converged in", k, "Iterations!")
			break
		x_last = res
		xs = np.hstack((xs, np.array(res).reshape(N, 1)))
		k += 1
	return res, xs, k

def gauss_seidel(b,N,x,iter,tol, actual):
	# While not Converged
	xs = np.array(x)
	x_last = x
	k = 1
	change = np.inf
	while(k <= iter and change > tol):
		res = []
		for i in range(N):
			if(i == 0):
				res.append(float(0.5 * (b[i] + x_last[i+1])))
			elif(not(i == N-1)):
				res.append(float(0.5 * (b[i] + res[i-1] + x_last[i+1])))
			else:
				res.append(float(0.5 * (b[i] + res[i-1])))
		change = np.linalg.norm((np.array(actual) - np.array(res)), ord=np.inf)
		#change = np.abs(np.array(x_last) - np.array(res)).max()
		if(change <= tol):
			print("Gauss-Seidel Converged in", k, "Iterations!")
			break
		x_last = res
		xs = np.hstack((xs, np.array(res).reshape(N, 1)))
		k += 1
	return res, xs, k

def SOR(w,b,N,x,iter,tol, actual):
	# While not Converged
	xs = np.array(x)
	x_last = x
	k = 1
	change = np.inf
	while(k <= iter and change > tol):
		res = []
		for i in range(N):
			if(i == 0):
				res.append(float(((1 - w) * x_last[i]) + (0.5 * w * (b[i] + x_last[i+1]))))
			elif(not(i == N-1)):
				res.append(float(((1 - w) * x_last[i]) + (0.5 * w * (b[i] + res[i-1] + x_last[i+1]))))
			else:
				res.append(float(((1 - w) * x_last[i]) + (0.5 * w * (b[i] + res[i-1]))))
		change = np.linalg.norm((np.array(actual) - np.array(res)), ord=np.inf)
		if(change <= tol):
			print("SOR Converged in", k, "Iterations!")
			break
		x_last = res
		xs = np.hstack((xs, np.array(res).reshape(N, 1)))
		k += 1
	return res, xs, k

def A_x(x):
	Ax = []
	N = np.size(x)
	for i in range(N):
		if(i == 0):
			Ax.append(float((2*x[i]) - x[i+1]))
		elif(i == N - 1):
			Ax.append(float(-x[i-1] + (2*x[i])))
		else:
			Ax.append(float(-x[i-1] + (2*x[i]) - x[i+1]))
	return Ax

def conjugate_gradient(b,N,x,iter,tol, actual):
	# While not Converged
	xs = np.array(x)
	k = 1
	res_x = x.T[0].copy()
	change = np.inf
	# Create First Residual
	r = b - np.array(A_x(x))
	s = r.copy()
	while((k-1) < N and change > tol):
		# Create alpha
		u = np.array(A_x(s))
		alpha = np.dot(s, r)/np.dot(s, u)

		# Update X
		x_last = res_x.copy()
		res_x += (alpha*s)
		xs = np.hstack((xs, np.array(res_x).reshape(N, 1)))

		# Update Residual
		r = b - np.array(A_x(res_x))

		beta = -np.dot(r, u)/np.dot(s, u)
		s = r + (beta*s)

		k += 1
		# Check for Convergence
		change = np.linalg.norm((np.array(actual) - np.array(res_x)), ord=np.inf)
		#change = np.abs(np.array(x_last) - np.array(res_x)).max()
		if(change <= tol or k >= N):
			print("Conjugate Gradient Converged in", k, "Iterations!")
			break
	return res_x, xs, k

def steepest_descent(b,N,x,iter,tol, actual):
	# While not Converged
	xs = np.array(x)
	k = 1
	res_x = x.T[0].copy()
	change = np.inf
	# Create First Residual
	r = b - np.array(A_x(x))
	s = r.copy()
	#while((k-1) < N and change > tol):
	while(change > tol):
		# Create alpha
		u = np.array(A_x(r))
		alpha = np.dot(r, r)/np.dot(r, u)

		# Update X
		x_last = res_x.copy()
		res_x += (alpha*r)
		xs = np.hstack((xs, np.array(res_x).reshape(N, 1)))

		# Update Residual
		r = b - np.array(A_x(res_x))

		k += 1
		# Check for Convergence
		change = np.linalg.norm((np.array(actual) - np.array(res_x)), ord=np.inf)
		#change = np.abs(np.array(x_last) - np.array(res_x)).max()
		#if(change <= tol or k >= N):
		if(change <= tol):
			print("Steepest Descent Converged in", k, "Iterations!")
			break
	return res_x, xs, k

#Generate N * N Stiffness Matrix
def create_stiff(n):
	A = np.zeros((n, n))
	for i in range(n):
		A[i,i] = 2
		if(i > 0):
			A[i,i-1] = -1
			A[i-1,i] = -1
	return A

def main():
	n = int(sys.argv[1])
	h = 1/n
	tol = 0.00001

	#Generate N * N Stiffness Matrix
	A = create_stiff(n-1)

	#Generate Correct Answer
	w = []
	for i in range(n-1):
		w.append(np.exp(np.sin(np.pi * (i+1) * h)) - 1)
	f = (1/(h**2)) * np.matmul(A, w)

	#Ax = b variables to iterate:
	x = np.zeros((n-1, 1))
	b = (h**2) * f

	# Jacobi
	j_res, j_xs, j_k = jacobi(b,n-1,x,10000,tol,w)
	j_error = []
	for col in j_xs.T:
		j_error.append(np.abs(col - w))
	j_error = np.array(j_error).T

	# Gauss-Seidel
	gs_res, gs_xs, gs_k = gauss_seidel(b,n-1,x,10000,tol,w)
	gs_error = []
	for col in gs_xs.T:
		gs_error.append(np.abs(col - w))
	gs_error = np.array(gs_error).T

	# SOR

	# Eigenvalues of Jacobi and Gauss Seidel Error Transfer matrices:
	g = []
	j = []
	a = []
	for i in range(len(x)):
		a.append(2 - (2 * math.cos(math.pi * (i+1) * h)))	# Eigenvalues of A
		j.append(1 - (0.5 * a[i]))				# Eigenvalues of Jacobi Error Transfer Matrix
	g_max = max(j)**2						# Maximum Eigenvalue of Gauss-Seidel Error Transfer Matrix

	# Find optimal value and perform SOR
	w_opt = 2 / (1 + math.sqrt(1 - g_max))
	sor_res, sor_xs, j_k = SOR(w_opt, b, n-1, x, 10000, tol, w)
	sor_error = []
	for col in sor_xs.T:
		sor_error.append(np.abs(col - w))
	sor_error = np.array(sor_error).T

	# Conjugate Gradient Method
	cj_res, cj_xs, cj_k = conjugate_gradient(b,n-1,x,10000,tol,w)
	cj_error = []
	for col in cj_xs.T:
		cj_error.append(np.abs(col - w))
	cj_error = np.array(cj_error).T

	# Steepest Descent Method
	sd_res, sd_xs, sd_k = steepest_descent(b,n-1,x,10000,tol,w)
	sd_error = []
	for col in sd_xs.T:
		sd_error.append(np.abs(col - w))
	sd_error = np.array(sd_error).T


	j_err = []
	gs_err = []
	sor_err = []
	cj_err = []
	sd_err = []
	for i in range(len(j_error.T)):
		j_err.append(np.linalg.norm(np.array(j_error.T[i]), ord=np.inf))
	for i in range(len(gs_error.T)):
		gs_err.append(np.linalg.norm(np.array(gs_error.T[i]), ord=np.inf))
	for i in range(len(sor_error.T)):
		sor_err.append(np.linalg.norm(np.array(sor_error.T[i]), ord=np.inf))
	for i in range(len(cj_error.T)):
		cj_err.append(np.linalg.norm(np.array(cj_error.T[i]), ord=np.inf))
	for i in range(len(sd_error.T)):
		sd_err.append(np.linalg.norm(np.array(sd_error.T[i]), ord=np.inf))

	# Print Output
	#print("Jacobi: ", np.array(j_res).reshape(len(j_res), 1))
	#print("Gauss-Seidel: ", np.array(gs_res).reshape(len(gs_res), 1))
	#print("SOR: ", np.array(sor_res).reshape(len(sor_res), 1))
	#print("Conjugate Gradient: ", np.array(cj_res).reshape(len(cj_res), 1))
	#print("Steepenst Descent: ", np.array(sd_res).reshape(len(cj_res), 1))

	# Save output
	filename = "output/Jacobi_Error_" + str(n) + ".csv"
	np.savetxt(filename, j_err, delimiter=',', fmt='%3.6f')
	filename = "output/Gauss_Seidel_Error_" + str(n) + ".csv"
	np.savetxt(filename, gs_err, delimiter=',', fmt='%3.6f')
	filename = "output/SOR_Error_" + str(n) + ".csv"
	np.savetxt(filename, sor_err, delimiter=',', fmt='%3.6f')
	filename = "output/Conjugate_Gradient_Error_" + str(n) + ".csv"
	np.savetxt(filename, cj_err, delimiter=',', fmt='%3.6f')
	filename = "output/Steepest_Descent_Error_" + str(n) + ".csv"
	np.savetxt(filename, sd_err, delimiter=',', fmt='%3.6f')

main()
