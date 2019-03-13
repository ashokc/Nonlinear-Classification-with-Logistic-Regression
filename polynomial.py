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

lmin = 1.0e-3
lmax = 1-1.0e-3

#
#   f(x,y) = c0 * 1 + c1 * x + c2 * y + c3 * x^2 + c4 * x * y + c5 * y^2
#
#   2.25 -3.0*x -18.0*y + 35.0 * x * x - 64.0 * x * y + 50.0 * y * y
#   prp ./polynomial.py 1000 2.25 -3.0 -18.0 35.0 -64.0 50.0

args = sys.argv
true_w = np.zeros(6)
nPoints = int(args[1])
true_w[0] = float(args[2])
true_w[1] = float(args[3])
true_w[2] = float(args[4])
true_w[3] = float(args[5])
true_w[4] = float(args[6])
true_w[5] = float(args[7])

if (true_w[0] == 0):
    print ('For easy scaling & comparison, we use a non-zero intercept here... exiting')
    sys.exit(0)

def getF(w, data):
    return np.dot(data, w)
def generateData():
    ones = np.ones(nPoints*15)
    x = (lmax - lmin) * np.random.random_sample(nPoints*15) + lmin
    y = (lmax - lmin) * np.random.random_sample(nPoints*15) + lmin
    allData = np.stack((ones, x, y, x*x, x*y, y*y),axis=-1)
    fvals = getF(true_w, allData)
    data, labels = [], []
    zeroCount, oneCount = 0, 0
    for i,f in enumerate(fvals):
        if ( (zeroCount < nPoints) and (f > 1.0e-1) ):
            labels.append(0.0)
            data.append(allData[i])
            zeroCount = zeroCount + 1
        if ( (oneCount < nPoints) and (f < -1.0e-1) ):
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

# True boundary
true_w_scaled = true_w/true_w[0]
# SK. Just the linear data
lr = LogisticRegression(max_iter=10000, verbose=0, tol=1.0e-8, solver='lbfgs' )
lr.fit(data[train_indices][:,1:3], labels[train_indices])
linear_predicted_labels = lr.predict(data[test_indices][:,1:3])
w = [lr.intercept_[0], lr.coef_[0][0], lr.coef_[0][1]]
linear_w_scaled = w/w[0]
# SK. Augmented with non linear data
lr2 = LogisticRegression(max_iter=10000, verbose=0, tol=1.0e-8, solver='lbfgs' )
lr2.fit(data[train_indices][:,1:6], labels[train_indices])
nonlinear_predicted_labels = lr2.predict(data[test_indices][:,1:6])
w = [lr2.intercept_[0], lr2.coef_[0][0], lr2.coef_[0][1], lr2.coef_[0][2], lr2.coef_[0][3], lr2.coef_[0][4]]
nonlinear_w_scaled = w/w[0]

from scipy import optimize

data_train = data[train_indices]
labels_train = labels[train_indices]
def getLLforMin(w):
    f = getF(w, data_train)
    LL = np.dot(labels_train.T, f) - np.sum (np.log(1.0 + np.exp(f)))
    return -LL
def getdLLbydwforMin(w):
    f = getF(w, data_train)
    dLLbydW = np.dot(data_train.T, labels_train - 1.0/(1+np.exp(-f)))
    return -dLLbydW

w0 = true_w * 1.0e-9
print ('\nsolve df/dw = 0 \n')
root = optimize.fsolve(getdLLbydwforMin, w0)
solve_dLLbydw_0_w_scaled = root / root[0]
print ('True Weights:',true_w_scaled)
print ('Linear scikit-learn Weights:',linear_w_scaled)
print ('Augmented/Nonlinear scikit-learn Weights:',nonlinear_w_scaled)
print ('Solving dLL/dw = 0:',solve_dLLbydw_0_w_scaled)
#
# Plot decisoon boundaries
#
def solveQuad (w, z):
    a = w[5]
    b = w[4] * z + w[2]
    c = w[0] + w[1] * z + w[3] * z * z
    disc = b*b - 4 * a * c
    y1, y2 = -99, -99
    if (disc > 0):
        y1 = (-b + (b*b - 4 * a * c)**0.5 ) / (2*a)
        y2 = (-b - (b*b - 4 * a * c)**0.5 ) / (2*a)
    return y1, y2

def getBoundary (w):
    low = (lmax-lmin)/100
    randX = np.random.uniform(low, high=low*2)
    xvals = np.linspace(lmin+randX, lmax-randX, 100)
    boundary_data = []
    for i, x in enumerate(xvals):
        y_1, y_2 = solveQuad (w, x)
        if (y_1 != -99):
            boundary_data.append ([x, y_1])
            boundary_data.append ([x, y_2])
    boundary_data=np.array([np.array(xi) for xi in boundary_data])
    return boundary_data

xvals = np.linspace(lmin, lmax, 100)
linear_boundary_data = np.stack((xvals, ( - linear_w_scaled[0] - linear_w_scaled[1] * xvals ) / linear_w_scaled[2]),axis=-1)
nonlinear_boundary_data = getBoundary (nonlinear_w_scaled)
true_boundary_data = getBoundary (true_w_scaled)
fsolve_boundary_data = getBoundary (solve_dLLbydw_0_w_scaled)

fig = plt.figure(figsize=(8,8),dpi=720)
filename = 'predicted-boundaries'
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
#plt.plot(true_boundary_data[:, 0], true_boundary_data[:, 1], color='g', linestyle='', markersize=3,marker='*')
plt.plot(nonlinear_boundary_data[:, 0], nonlinear_boundary_data[:, 1], color='b', linestyle='', markersize=3,marker='<')
plt.plot(fsolve_boundary_data[:, 0], fsolve_boundary_data[:, 1], color='g', linestyle='', markersize=3,marker='>')
plt.plot(linear_boundary_data[:, 0], linear_boundary_data[:, 1], color='r')
plt.plot(data0[:, 1], data0[:, 2], color='k', linestyle='', markersize=1,marker='.')
plt.plot(data1[:, 1], data1[:, 2], color='c', linestyle='', markersize=1,marker='+')
fig.savefig(filename + '.png', format='png', dpi=720)
plt.close()

# fsolve predictions
fsolve_predicted_fvals = getF(solve_dLLbydw_0_w_scaled, data[test_indices])
fsolve_predicted_labels = np.zeros(fsolve_predicted_fvals.size)
onePoints = np.where(fsolve_predicted_fvals < 0.0)
fsolve_predicted_labels[onePoints] = 1.0

target_names = ['zero', 'one']
print ('LR linear:')
print (confusion_matrix(labels[test_indices], linear_predicted_labels))
print (classification_report(labels[test_indices], linear_predicted_labels, digits=4, target_names=target_names))

print ('LR nonlinear:')
print (confusion_matrix(labels[test_indices], nonlinear_predicted_labels))
print (classification_report(labels[test_indices], nonlinear_predicted_labels, digits=4, target_names=target_names))

print ('fsolve dLLbydw=0:')
print (confusion_matrix(labels[test_indices], fsolve_predicted_labels))
print (classification_report(labels[test_indices], fsolve_predicted_labels, digits=4, target_names=target_names))

