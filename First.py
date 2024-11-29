import numpy as np
import matplotlib.pyplot as plt

def computeGradient(x:np.ndarray,y:np.ndarray ,w ,b):
    devW=devB=0
    m = x.shape[0]
    for i in range(m):
        devW += (w*x[i] + b - y[i])* x[i]
        devB += (w*x[i] + b - y[i])
    return devW/m,devB/m

def gradientDecent(x:np.ndarray,y:np.ndarray , alpha : float , iter : int):
    
    w=b=8
    for i in range(iter):
        devW , devB = computeGradient(x,y, w,b)
        w = w - alpha * devW
        b = b - alpha * devB
    return w,b

def LinerReggressionModel(x : np.ndarray,w,b):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i]= w*x[i] + b
    return y    
x_train = np.array(range(0,20,2))
y_train = np.random.randint(0,60,x_train.shape[0])

w,d = gradientDecent(x_train,y_train,0.01,10000)

plt.scatter(x_train,y_train,marker='x' , c= 'r')
plt.plot(x_train , LinerReggressionModel(x_train,w,d))
plt.show()
