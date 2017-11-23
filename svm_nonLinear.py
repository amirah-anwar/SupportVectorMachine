# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import numpy as np
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
	data = np.array(X[:,0:2])
	labels = np.array(X[:,2])

	print "data", data
	print "labels", labels

if __name__ == "__main__":
	main()