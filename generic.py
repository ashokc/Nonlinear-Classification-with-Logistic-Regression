import sys
import json
from itertools import product

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import random as rn
import numpy as np
import matplotlib.pyplot as plt
rn.seed(1)
np.random.seed(1)

xmin = -1.77
xmax = 1.77
ymin = 0.0
ymax = 1.0

#
# f(x,y) = c0 + c1 * sin(c2 * x^2) + c3*y
# df/dc0 = 1
# df/dc1 = sin (c2 * x^2)
# df/dc2 = c1 * x^2 * cos (c2 * x^2)
# df/dc3 = y
#
#   prp ./generic.py 1000 1.0 -1.0 1.0 -1.0
#

args = sys.argv
true_w = np.zeros(4)
nPoints = int(args[1])
true_w[0] = float(args[2])
true_w[1] = float(args[3])
true_w[2] = float(args[4])
true_w[3] = float(args[5])

def getF(w, data):
    return w[0] + w[1] * np.sin(w[2] * np.power(data[:,0], 2)) + w[3] * data[:,1]
def getFprime(w, data):
    size0 = data[:,0].size
    dfbydw0 = np.ones(size0)
    dfbydw1 = np.sin(w[2] * np.power(data[:,0], 2))
    dfbydw2 = w[1] * np.power(data[:,0],2) * np.cos(w[2] * np.power(data[:,0], 2))
    dfbydw3 = data[:,1]
    dfbydw = np.stack ((dfbydw0, dfbydw1, dfbydw2, dfbydw3), axis=-1)
    return dfbydw
def generateData():
    x = (xmax - xmin) * np.random.random_sample(nPoints*25) + xmin
    y = (ymax - ymin) * np.random.random_sample(nPoints*25) + ymin
    allData = np.stack((x, y),axis=-1)
    fvals = getF (true_w, allData)
    data, labels = [], []
    zeroCount, oneCount = 0, 0
    for i,f in enumerate(fvals):
        if ( (zeroCount < nPoints) and (f > 1.0e-2) ):
            labels.append(0.0)
            data.append(allData[i])
            zeroCount = zeroCount + 1
        if ( (oneCount < nPoints) and (f < -1.0e-2) ):
            labels.append(1.0)
            data.append(allData[i])
            oneCount = oneCount + 1
        if (zeroCount == nPoints) and (oneCount == nPoints):
            break;
    data=np.array([np.array(xi) for xi in data])
    labels = np.array(labels)
    return data, labels

data, labels = generateData()
zeroPoints = np.where(labels == 0.0)
data0 = data[zeroPoints[0]]
onePoints = np.where(labels == 1.0)
data1 = data[onePoints[0]]
print ('n_zero, n_one:', len(data0), len(data1))

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(data, labels)
train_indices, test_indices = next(sss)
data_train = data[train_indices]
labels_train = labels[train_indices]

lr = LogisticRegression(max_iter=10000, verbose=0, tol=1.0e-8, solver='lbfgs' )
lr.fit(data_train, labels_train)
linear_predicted_labels = lr.predict(data[test_indices])
linear_w = np.array([lr.intercept_[0], lr.coef_[0][0], lr.coef_[0][1]]) / lr.intercept_[0]

from scipy import optimize

def getLLforMin(w):
    f = getF(w, data_train)
    LL = np.dot(labels_train.T, f) - np.sum (np.log(1.0 + np.exp(f)))
    return -LL
def getdLLbydwforMin(w):
    f = getF(w, data_train)
    dfbydw = getFprime(w, data_train)
    dLLbydW = np.dot(dfbydw.T, labels_train - 1.0/(1+np.exp(-f)))
    return -dLLbydW

w0 = true_w * 1.0e-3
print ('\nMinimize with Numerical Jacobian\n')
root = optimize.minimize(getLLforMin, w0)
print (root)
minimize_numerical_jac_w = root['x']/root['x'][0]
minimize_numerical_jac_w[2] = root['x'][2]

print ('\nMinimize with Analytic Jacobian\n')
root = optimize.minimize(getLLforMin, w0,jac=getdLLbydwforMin)
print (root)
minimize_analytic_jac_w = root['x']/root['x'][0]
minimize_analytic_jac_w[2] = root['x'][2]

print ('True Weights:',true_w)
print ('Linear SK Weights:',linear_w)
print ('Minimization with Numerical Jacobian:',minimize_numerical_jac_w)
print ('Minimization with Analytic Jacobian:',minimize_analytic_jac_w)

#
# Plot decisoon boundaries
#

xvals = np.linspace(xmin, xmax, 100)
def getBoundary (w):
    y = (-w[0] - w[1] * np.sin(w[2] * xvals**2.0) ) / w[3]
    return np.stack((xvals,y),axis=-1)

linear_boundary_data = np.stack((xvals, ( - linear_w[0] - linear_w[1] * xvals ) / linear_w[2]),axis=-1)
true_boundary_data = getBoundary (true_w)
min_nj_boundary_data = getBoundary (minimize_numerical_jac_w)
min_aj_boundary_data = getBoundary (minimize_analytic_jac_w)

fig = plt.figure(figsize=(8,8),dpi=720)
filename = 'predicted-boundaries'
plt.xlim(xmin-1.0e-3, xmax+1.0e-3)
plt.ylim(ymin-1.0e-3, ymax+1.0e-3)
#plt.plot(true_boundary_data[:, 0], true_boundary_data[:, 1], color='g')
plt.plot(min_nj_boundary_data[:, 0], min_nj_boundary_data[:, 1], color='g')
#plt.plot(min_aj_boundary_data[:, 0], min_aj_boundary_data[:, 1], color='pink', linestyle='', markersize=2,marker='>')
plt.plot(linear_boundary_data[:, 0], linear_boundary_data[:, 1], color='r')
plt.plot(data0[:, 0], data0[:, 1], color='k', linestyle='', markersize=1,marker='.')
plt.plot(data1[:, 0], data1[:, 1], color='c', linestyle='', markersize=1,marker='+')
fig.savefig(filename + '.png', format='png', dpi=720)
plt.close()

target_names = ['zero', 'one']
def getPredictions (w, key):
    if (key == 'default-linear' ):
        pred_labels = linear_predicted_labels
    else:
        pred_fvals = getF(w, data[test_indices])
        pred_labels = np.zeros(pred_fvals.size)
        onePoints = np.where(pred_fvals < 0.0)
        pred_labels[onePoints] = 1.0
    print (key)
    print (confusion_matrix(labels[test_indices], pred_labels))
    print (classification_report(labels[test_indices], pred_labels, digits=4, target_names=target_names))

getPredictions(linear_w, 'default-linear')
getPredictions(minimize_numerical_jac_w, 'numerical-jac')
getPredictions(minimize_analytic_jac_w, 'analytic-jac')

