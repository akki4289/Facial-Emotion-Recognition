import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from time import time
import cPickle
import csv
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from IO import load
from os import system,getpid

system("taskset -p 0xff %d" % getpid())

nTestSamps = 3589
nTrainSamps = 25000

t0 = time()
trainX,trainY,testX,testY = load(nTrainSamps,nTestSamps)
print 'loaded and formatted data in',time()-t0

scaler = StandardScaler().fit(trainX)
trainX = scaler.transform(trainX)
testX = scaler.transform(testX)

# clf = SVC(verbose=True)
# clf = KNeighborsClassifier()
clf = GradientBoostingClassifier(verbose=2)

t1 = time()
clf.fit(trainX,trainY)
print 'fitted in',time()-t1

t2 = time()
predictions = clf.predict(testX)
print 'predicted in',time()-t2

print np.sum((predictions == testY))/float(len(testY))