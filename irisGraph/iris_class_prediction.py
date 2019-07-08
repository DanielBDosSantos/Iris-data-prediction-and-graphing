#This program predics what type of flower it is based on the inputs and graphs the points.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris = load_iris()
#print(iris.keys())
#print(iris['data'])
#print(iris['DESCR'])
#print(iris['target_names'])

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = np.array([[1,2,1,2]])

prediction = knn.predict(X_new)

print(iris['target_names'][prediction])

print(knn.score(X_test, y_test))

plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.show()
