# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
 
# load txt file
f = open("nonlinsep.txt", 'r')
result_matrix = []
for line in f.readlines():
    values_as_strings = line.split(',')
    arr = np.array(map(float, values_as_strings))
    result_matrix.append(arr)
X = np.array(result_matrix)

coordinates = np.array(X[:,0:2])
labels = np.array(X[:,2])

# fit the model
clf = svm.NuSVC()
clf.fit(coordinates,labels)

# Print the results
#print "Prediction " + str(clf.predict(coordinates))
#print "Actual     " + str(labels)
#print "Accuracy   " + str(net.score(coordinates, labels)*100) + "%"

# Output the values
#print "Coefficients " + str(clf.coef_)
#print "Intercept " + str(clf.intercept_)

# Plot the data points
#plt.scatter(X[:,0],X[:,1])
#plt.show()

# plot the decision function for each datapoint on the grid
xx, yy = np.meshgrid(np.linspace(-26, 26), np.linspace(-26, 26))

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#h0 = plt.plot(xx, yy, 'k-')
#plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
#plt.imshow(Z, interpolation='none', extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linetypes='--')
plt.scatter(X[:, 0], X[:, 1], s=30, c=labels, cmap=plt.cm.Paired, edgecolors='none')

plt.xticks(())
plt.yticks(())
plt.axis([-26, 26, -26, 26])
plt.show() 
