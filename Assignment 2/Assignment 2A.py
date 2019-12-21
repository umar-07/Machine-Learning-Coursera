import numpy as np
import pandas as pd
from matplotlib import pyplot

df=pd.read_csv('F:/PDF/Machine Learning/Videos/Machine Learning -Andrew Ng/Assignments ML/machine-learning-ex2/ex2/ex2data1.txt', sep=",", header=None)
X=df.iloc[: ,0:2]
y=df.iloc[:, -1]

#Plotting
#pos = y==1
#neg = y==0

def plotData(X, y):
    fig=plt.figure()
    pos = y == 1       #All the examples where output is 1
    neg = y == 0       #All the examples where output is 0
    pyplot.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)                    #plotting the positive data, and choosing on x-axis as X[0] and y-axis as X[1]
    pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', mec='k', ms=8, mew=1)  #plotting the negative data, and choosing on x-axis as X[0] and y-axis as X[1]
    plt.show()
    # linewidth(lw), markersize(ms), markeredgecolor(mec), markerfacecolor(mfc), markeredgewidth(mew)

#plotData(X,y)
pyplot.xlabel('Exam 1 score')
pyplot.ylabel('Exam 2 score')
pyplot.legend(['Admitted', 'Not admitted'])

#Sigmoid Function

def sigmoid(z):
    z=np.array(z)   #Convert input to numpy array
    g=np.zeros(z.shape)
    g= 1/(1+ np.exp(-z))
    return g

#Cost Function

m,n = X.shape   #m=number of training examples, n=number of features
X=np.concatenate([np.ones((m,1)), X], axis=1)

def costFunction(theta, X, y):
    m=y.size
    J=0
    grad=np.zeros(theta.shape)                                              #As in logistic regression we compute gradient of cost function
    h=sigmoid(np.dot(X,theta.T))    #Or h=sigmiod(X.dot(theta.T))           #previously only, as it will help in calculting Theta later....
    J = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h)))    #Gradient is the partial derivate of the cost function
    grad = (1 / m) * (h - y).dot(X)
    return J, grad

#Testing

#initial_theta = np.zeros(n+1)
#cost, grad = costFunction(initial_theta, X, y)
#print(cost)

#Method 2: Advance Optimization Technique Using SciPy

from scipy import optimize
initial_theta = np.zeros(n+1)
options= {'maxiter': 400}
res = optimize.minimize(costFunction,initial_theta,(X, y),jac=True,method='TNC',options=options)
cost=res.fun
theta=res.x
#print(cost)

#Prediction

def predict(theta, X):
    m=X.shape[0]
    p=np.zeros(m)
    p=np.round(sigmoid(X.dot(theta.T)))   #Using threshold as 0.5
    return p

#Testing
prob = sigmoid(np.dot([1, 45, 85], theta))
print(prob)
