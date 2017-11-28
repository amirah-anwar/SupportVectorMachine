# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
#from sklearn.svm import LinearSVC
from sklearn import svm
 
# load txt file
f = open("linsep.txt", 'r')
result_matrix = []
for line in f.readlines():
    values_as_strings = line.split(',')
    arr = np.array(map(float, values_as_strings))
    result_matrix.append(arr)
X = np.array(result_matrix)

coordinates = np.array(X[:,0:2])
labels = np.array(X[:,2])

# Linear Classifier
clf = svm.SVC(kernel = 'linear', C = 1.0)
#clf = LinearSVC()
clf.fit(coordinates,labels)
 
# Print the results
#print "Prediction " + str(clf.predict(coordinates))
#print "Actual     " + str(labels)
#print "Accuracy   " + str(net.score(coordinates, labels)*100) + "%"

# Output the values
print "Coefficients " + str(clf.coef_)
print "Intercept " + str(clf.intercept_)
#print "Support Vectors " + str(clf.support_vectors_)

# Plot the data points
#plt.scatter(X[:,0],X[:,1])
#plt.show()

#Visualizing the data
w = clf.coef_[0]
print "weights", w

a = -w[0] / w[1]

xx = np.linspace(0,1)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-')

plt.scatter(X[:, 0], X[:, 1], c = labels)
plt.legend()
plt.show()
 
