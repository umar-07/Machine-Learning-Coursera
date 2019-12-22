import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils          #Util stores small functions created by us or Python defined
                      #Here Util was originally used to use function map feature which was defined by user

#Reading the data

df=pd.read_csv('F:/PDF/Machine Learning/Videos/Machine Learning -Andrew Ng/Assignments ML/machine-learning-ex2/ex2/ex2data2.txt', sep=",", header=None)
X=df.iloc[:, :2]        #This is a dataframe as more than one column
y=df.iloc[:, 2]         #This is series as one column (still in dataframe format)

#Plotting the Data

def plotData(X,y):
    fig=plt.figure()
    pos = y==1
    neg = y==0
    plt.plot(X.loc[pos, 0], X.loc[pos, 1], 'k*', lw=2, ms=10)        #Use of loc is necessary and not iloc as X, y are dataframe
    plt.plot(X.loc[neg, 0], X.loc[neg, 1], 'ko', mfc='y', mec='k', ms=8)

    #ALTERNATE :

    #Preiously only convert dataframe X, y into numpy as
    #X=df.iloc[:, :2].to_numpy()
    #y=df.iloc[:, 2].to_numpy()
    #and then just use
    #plt.plot(X[pos, 0], X[pos, 1], 'k*')
    #plt.plot(X[neg, 0], X[neg, 1], 'ko')

    plt.show()

#plotData(X,y)


#Sigmoid Function

def sigmoid(z):
    z=np.array(z)   #Convert input to numpy array
    g=np.zeros(z.shape)
    g= 1/(1+ np.exp(-z))
    return g

#Feauture Mapping


def mapFeature(X1, X2, degree=6):
    if X1.ndim > 0:
        out = [np.ones(X1.shape[0])]
    else:
        out = [np.ones(1)]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))     #X1 ** (i-j) == X1 raise to power (i-j)

    if X1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.array(out)


X = mapFeature(X.iloc[:, 0], X.iloc[:, 1])          #Using iloc is necessary or previously should have converted into numpy array 
                                                    #X has been converted into m*n array_like(m=training examples, n=feature(after feature mapping i.e. 28))

#Cost Function

def costReg(theta, X, y, lambda_):
    m=y.size
    J=0;
    grad=np.zeros(theta.shape)
    h=sigmoid(X.dot(theta.T))
    temp=theta
    temp[0]=0   #As for regularizzed term theta[0]=0
    J = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h))) + (lambda_ / (2 * m)) * np.sum(np.square(temp))
    grad = (1 / m) * (h - y).dot(X) #first gradient for normal term
    grad = grad + (lambda_ / m) * temp #then gradient for regularized term
    return J, grad

#Tesing of Cost and Gradient
initial_theta = np.zeros(X.shape[1])
lambda_ = 1
cost, grad = costReg(initial_theta, X, y, lambda_)
#print(cost)
#print(grad)
