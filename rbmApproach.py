import numpy as np
from os.path import join
from os import system,getpid
import matplotlib.pyplot as plt
from time import time
import cPickle
import csv
from IO import load

system("taskset -p 0xff %d" % getpid())

def energy(v,h,w):
	cbs = np.dstack(np.meshgrid(v,h)).reshape(-1, 2)
	return -(np.prod(cbs,axis=1)*w.flatten()).sum()

''' Stochastic activation of hidden layer given input and weights '''
def actH(v,h,w,stochastic=True):
	z = (v*w).sum(axis=1)
	prob = 1/(1+np.exp(-z))
	if stochastic:
		h = np.array((prob > np.random.random(h.size)),dtype=int)
	else:
		h = prob
	return h

''' Stochasitic activation of visible layer given hidden and weights '''
def actV(v,h,w,stochastic=True):
	z = (h*w.transpose()).sum(axis=1)
	prob = 1/(1+np.exp(-z))
	if stochastic:
		v = np.array((prob > np.random.random(v.size)),dtype=int)
	else:
		v = prob
	return v

''' Quick single pass contrastive divergence to update weights '''
def updateWeights(v,h,w,lr,decay,stochastic=False):
	h = actH(v,h,w)
	start = np.prod(np.dstack(np.meshgrid(v,h)).reshape(-1, 2),axis=1)
	v = actV(v,h,w,stochastic)
	h = actH(v,h,w,stochastic)
	end = np.prod(np.dstack(np.meshgrid(v,h)).reshape(-1, 2),axis=1)
	dw = lr*(start-end)
	w = w*(1-decay)
	w += dw.reshape((h.size,v.size))
	return w

def oneHot(n):
	vect = np.zeros(7)
	vect[n] = 1
	return vect

def sample(X,Y=None,i=None):
	if i is None:
		index = np.random.randint(0,X.shape[0])
	else:
		index = i
	if Y is None:
		x = X[index]
		return np.hstack((x,np.zeros(7)))
	else:
		x = X[index]
		return np.hstack((x,oneHot(Y[index])))

def test(v,h,w,nTestSamps):
	accuracy = []
	for i in xrange(nTestSamps):
		if i % 250 == 0: 
			print 'testing',round(i/float(nTestSamps)*100,3),'%'
		v = sample(testX,i=i)
		h = actH(v,h,w,False)
		v = actV(v,h,w,False)
		x = int(np.argmax(v[48*48:]))
		y = int(testY[i])
		if x == y:
			accuracy.append(1)
		else:
			accuracy.append(0)
	return np.sum(accuracy)/float(len(accuracy))

nTestSamps = 3589
nTrainSamps = 25000
t0 = time()
trainX,trainY,testX,testY = load(nTrainSamps,nTestSamps)
print 'loaded and formatted data in',time()-t0

lr = 0.001
decay = 0
v = sample(trainX,trainY)
h = np.zeros(1000)
w = np.random.normal(loc=0,scale=0.01,size=(h.size,v.size))
trainItrs = 250000
testItrs = 3589
samps = 20
start = time()
testBreaks = list(np.unique(np.array(np.logspace(3,np.log10(trainItrs),samps),dtype=int)))
# testBreaks = [1000,10000,20000,30000,40000,50000,60000,70000,80000,90000,99999]
errs = []
print testBreaks
for i in xrange(trainItrs):
	if i+1 in testBreaks:
		print i,'tests','training',round(i/float(trainItrs)*100,2),'%'
		err = test(v,h,w,nTestSamps)
		errs.append(err)
		print err
	v = sample(trainX,trainY,i = i % nTrainSamps)
	w = updateWeights(v,h,w,lr,decay,False)
print time()-start
tData = [testBreaks,errs]
cPickle.dump(tData,open('tDataLR0001N1000D0250K.p','wb'))
plt.plot(testBreaks,errs)
plt.show()
# for i in xrange(trainItrs):
# 	v = sample(trainX,trainY,i = i % nTrainSamps)
# 	w = updateWeights(v,h,w,lr,decay,False)
# print test(v,h,w,nTestSamps)