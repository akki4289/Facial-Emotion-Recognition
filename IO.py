import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from time import time
import cPickle
import csv

def norm(a):
	return a/float(a.max())

def processCSV(arr):
	X = np.array([np.fromstring(str(arr[i,1]),sep=' ') for i in xrange(len(arr))])
	X = norm(X)
	Y = arr[:,0].astype(float)
	return X,Y

def load(nTrain,nTest):
	trainX = []
	trainY = []
	f = open('train.csv')
	reader = csv.reader(f,delimiter=',', skipinitialspace=True)
	data = np.array(list(reader)[1:])
	trainData = data[:nTrain]
	testData = data[-nTest:]
	trainX,trainY = processCSV(trainData)
	testX,testY = processCSV(testData)
	return trainX,trainY,testX,testY