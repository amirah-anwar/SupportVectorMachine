# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import numpy as np
import quadprog
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
	labels = np.array([X[:,2]]) #(1,100)

	print "data", data.shape
	print "labels", labels.shape

	#(1) Solve with QP Q(yi x yj x K(xi, xj))
	yi_yj = labels * labels #(1,100)
	print "yi_yj", yi_yj.shape
	kernel_xi_xj = polynomial_kernel(data) #(100,100)
	print "kernel_xi_xj", kernel_xi_xj.shape
	Q = yi_yj*kernel_xi_xj #(100,100)

	print "Q", Q.shape
	q = np.ones(100) #(100,)
	print "q", q.shape
	G = -np.ones((100,1)) #(100,1)
	print "G", G.shape

	alpha = quadprog_solve_qp(Q,q,G.T,0.0,labels,0.0) #(100,)
	print "alpha", alpha

	#(2) w.T x + b
	weights = alpha * labels * kernel_xi_xj #(100,100)

	#(3) Calculate b
	b = labels - (alpha * labels * kernel_xi_xj)

	print "Kernel function: Polynomial kernel = (1 + x.T * x`)^2"
 	print "Equation of line of separation:", equation(weights, data, b)

def equation(w, x, b):
	return np.dot(w, x) + b

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