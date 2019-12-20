import pandas as pd
#Loading the data

df=pd.read_csv('F:/PDF\Machine Learning/Videos/Machine Learning -Andrew Ng/Assignments ML/machine-learning-ex1/ex1/ex1data1.txt', sep=",", header=None, names=["pop", "prof"])

#Splitting coloumn in X and Y

X=df["pop"]
y=df["prof"]

#plotting the data

import matplotlib.pyplot as plt
plt.plot(X,y, 'rX')
#plt.show()

#Implementing Gradient Descent

m=y.size        #number of training examples
import numpy as np
X=np.stack([np.ones(m), X], axis=1)         #adding coloumn of 1's to X

        #cost function
def computeCost(X, y, theta):
    m=y.size
    J=0
    h=np.dot(X,theta)
    J=(1/(2*m))*np.sum(np.square(h-y))
    return J

#Computing cost twice
J = computeCost(X, y, theta=np.array([0.0, 0.0])) #Initializing theta to 0,0
J = computeCost(X, y, theta=np.array([-1, 2])) #Initializing theta to -1,2

        #Gradient Descent
def gradientDescent(X, y, theta, aplha, iterations):
    m=y.size
    theta=theta.copy()      #Copying the original theta as to avioid cahnging the real values as numpy arrays are always passed through reference
    J_history = []          #Using python list to save all the computed J
    for i in range(iterations):
        theta = theta - (alpha/m) * (np.dot(X, theta) - y).dot(X) #Computing theta in each iteration
        J_history.append(computeCost(X, y, theta))
    return theta, J_history

theta = np.zeros(2)
iterations = 1500
alpha = 0.01
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

#plotting Linear Regression

#plotData(X[:, 1], y)
plt.plot(X[:, 1], np.dot(X, theta), '-')
#plt.show()

#TESTING
predict1 = np.dot([1, 3.5], theta)
predict2 = np.dot([1, 7], theta)

print(predict1)
print(predict2)


