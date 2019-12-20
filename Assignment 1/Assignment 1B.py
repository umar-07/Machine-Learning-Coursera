import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('F:/PDF/Machine Learning/Videos/Machine Learning -Andrew Ng/Assignments ML/machine-learning-ex1/ex1/ex1data2.txt', sep=",", header=None, names=["area", "bed", "price"])
X=df.iloc[: , [0,1]]                                                  #X=[['area', 'bed']] can be used           #X= df[['area', 'bed']]     Can not be used as two keys in one array is giving error
y= df.iloc[: , -1]        #y=df.iloc[:,2] can also be used            #y=[['price']] can be used                 #y=df["price"]              Can be done as single key is being used

#Noramlization

def featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])     #Creating a n*1 vector of mu (n=number of features)
    sigma = np.zeros(X.shape[1]) #Creating a n*1 vector of sigma
    mu=np.mean(X, axis=0)        #Taking out mean of X of the two coloumns (axis=0 means coloumn, axis=1 means rows)
    sigma=np.std(X, axis=0)
    X_norm = (X-mu)/sigma        #Normalizing
    return X_norm, mu, sigma

m=y.shape[0]
X_norm, mu, sigma = featureNormalize(X)
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)   #or X=np.stack([np.ones(m), X], axis=1)

#Cost Function

def costMulti(X, y, theta):
    m = y.shape[0]
    J=0
    h = np.dot(X, theta)
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    return J
#Gradient Descent

def gradientDescent(X, y, theta, alpha, iterations):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    for i in range(iterations):
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        J_history.append(costMulti(X, y, theta))
    return theta, J_history

#Noraml Equation (Method 2)

def normalEqn(X, y):
    theta = np.zeros(X.shape[1])
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    return theta

#Calculating through Normal Equation
theta = normalEqn(X, y)
X_array = [1, 1650, 3]
X_array = [1, 1650, 3]
X_array[1:3] = (X_array[1:3] - mu) / sigma          #We are normalizing input again, we will have to do this
price = np.dot(X_array, theta)                      #whenever we take input as the input is not in normalized range
