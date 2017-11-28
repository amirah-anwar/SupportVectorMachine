# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import numpy as np
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt
# ==============Support Vector Machine for Non-linearly Separable==================================

def main():

	# load linearly separable txt file
	f = open("linsep.txt", 'r')
	result_matrix = []
	labels = []

	for line in f.readlines():
		values_as_strings = line.split(',')
		arr = np.array(map(float, values_as_strings))
		result_matrix.append(arr)

	X = np.array(result_matrix)
	data = np.array(X[:,0:2]) #(100,2)
	labels = np.array(X[:,2]) #(1,100)

	#(1) Solve with QP Q(yi x yj x K(xi, xj))
	yi_yj = np.outer(labels,labels) #(100,100)
	xi_xj = np.dot(data,data.T) #(100,100)
	Q = cvxopt.matrix(yi_yj*xi_xj) #(100,100)
	q = cvxopt.matrix(np.ones(100) * -1) #(100,)
	A = cvxopt.matrix(labels,(1,100))
	b = cvxopt.matrix(0.0)
	G = cvxopt.matrix(np.diag(np.ones(100) * -1))
	h = cvxopt.matrix(np.zeros(100))

	solution = cvxopt.solvers.qp(Q, q, G, h, A, b)
	a = np.ravel(solution['x'])
	sv = a > 1e-6

	indices = np.arange(len(a))[sv]
	alphas = a[sv] #alphas that are above threshold
	sv_data = data[sv] #data points corresponding to that alphas
	sv_label = labels[sv] #labels corresponding to that alphas

	#(2) Calculate b
	b = np.sum(sv_label - ( sv_label * xi_xj[indices,sv] * alphas))
	b = b/len(alphas)
	print "Intercept", b

	#(3) w.T x + b
	weights = np.zeros(2)
	for n in range(len(alphas)):
		weights += alphas[n] * sv_label[n] * sv_data[n]
	print "Coefficients", weights

if __name__ == "__main__":
	main()