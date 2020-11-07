''' BagLearner: boostrap aggregating (bagging). Tri Bui tbui61
API specifications:
mport BagLearner as bl
learner = bl.BagLearner(learner = al.ArbitraryLearner, kwargs = {"argument1":1, "argument2":2}, bags = 20, boost = False, verbose = False)
learner.addEvidence(Xtrain, Ytrain)
Y = learner.query(Xtest)
'''

import numpy as np
import pandas as pd
import RTLearner as rtl
from scipy import stats

class BagLearner:
    def __init__(self, learner, kwargs = {}, bags = 20, boost = False, verbose = False):
        super().__init__()
        learners = []
        for i in range(bags):
            learners.append(learner(**kwargs))
        self.learners = learners
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        # wait I don't need a tree because for each input I can just query individual learner and TAKE THE MODE

    def author(self):
        return 'tbui61'
    
    def addEvidence(self, Xtrain, Ytrain):
        #for each learner in learners, create a random index out of the training set, which gives a 'randomized' data set
        indexMax = Xtrain.shape[0]
        for learner in self.learners:
            index = np.random.choice(indexMax, indexMax)
            Xrand = Xtrain[index]
            Yrand = Ytrain[index]
            learner.addEvidence(Xrand, Yrand)

    def query(self, Xtest):
        #note: predArray will be 2d (1d for each learner, 1d for the actual predictions for each Xtest row)
        predArray = np.array([learner.query(Xtest) for learner in self.learners])
        #Ypred = np.mean(predArray,axis=0)
        Ypred = stats.mode(predArray, axis=0).mode
        return Ypred.T

