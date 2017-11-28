# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import numpy as np
import cvxopt
import cvxopt.solvers
# ==============Support Vector Machine for Non-linearly Separable==================================

def main():

	# load non-linearly separable txt file
	f = open("nonlinsep.txt", 'r')
	result_matrix = []
	labels = []

	for line in f.readlines():
		values_as_strings = line.split(',')
		arr = np.array(map(float, values_as_strings))
		result_matrix.append(arr)

	X = np.array(result_matrix)
	data = np.array(X[:,0:2]) #(100,2)
	labels = np.array(X[:,2]) #(1,100)

	print "data", data.shape
	print "labels", labels.shape

	#(1) Solve with QP Q(yi x yj x K(xi, xj))
	yi_yj = labels * labels #(100,100)
	print "yi_yj", yi_yj.shape
	kernel_xi_xj = polynomial_kernel(data) #(100,100)
	# kernel_xi_xj = np.zeros((100, 100))
	# for i in range(100):
	# 	for j in range(100):
	# 		kernel_xi_xj[i,j] = polynomial_kernel(data[i], data[j])
	print "kernel_xi_xj", kernel_xi_xj.shape
	Q = cvxopt.matrix(yi_yj*kernel_xi_xj) #(100,100)

	# print "Q", Q.shape
	q = cvxopt.matrix(np.ones(100) * -1) #(100,)
	# print "q", q.shape
	A = cvxopt.matrix(labels,(1,100))
	b = cvxopt.matrix(0.0)
	G = cvxopt.matrix(np.diag(np.ones(100) * -1))
	h = cvxopt.matrix(np.zeros(100))

	# G = -np.ones((100,1)) #(100,1)
	# print "Q", Q

	solution = cvxopt.solvers.qp(Q, q, G, h, A, b)
	a = np.ravel(solution['x'])
	print "a before", a
	print "a", a.shape
	sv = a > 1e-5
	print "sv", sv

	indices = np.arange(len(a))[sv]
	print "len(a)", len(a)
	print "indices", indices
	alphas = a[sv]
	sv_data = data[sv]
	sv_label = labels[sv]
	print "alphas after", alphas
	print "sv_data", sv_data
	print "sv_label", sv_label
	print("%d support vectors out of %d points" % (len(alphas), 100))

	print "kernel_xi_xj", kernel_xi_xj

	# alpha = quadprog_solve_qp(Q,q,G.T,0.0,A,0.0) #(100,)
	# print "alpha", alpha
	print "kernel_xi_xj[indices,sv]", kernel_xi_xj[indices,sv]
	# #(2) Calculate b
	b = sv_label - ( sv_label * kernel_xi_xj[indices,sv] * alphas)
	print "b shape", b

	#(3) w.T x + b
	weights = (sv_label * kernel_xi_xj[indices,sv]*alphas) + b #(100,100)
	print "weights shape", weights

	print "Kernel function: Polynomial kernel = (1 + x.T * x`)^2"
	print "data[indices,sv]", data[indices]
 	print "Equation of line of separation:", equation(weights, data[indices], b)

def equation(w, x, b):
	w = np.array([w]).T
	print "w.shaoe",w.shape
	print "w", w
	print "x", x.shape
	return (np.dot(x.T, w) + b)

#(1 + x.T * x`)^2
def polynomial_kernel(data):
	return np.power((1+np.dot(data,data.T)),2)

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = P
    print "qp_G", qp_G
    qp_a = -q
    if A is not None:
    	print "A.shape", A.shape
    	print "G.shape", G.shape
    	print "A", A
    	print "G", G

    	qp_C = -np.vstack((A, G)).T
    	print "qp_C", qp_C
    	print "qp_C.shape", qp_C.shape
    	qp_b = -np.hstack([b, h])
    	meq = A.shape[0]
    else:  # no equality constraint
      qp_C = -G.T
      qp_b = -h
      meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

if __name__ == "__main__":
	main()