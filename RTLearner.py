''' Regression tree learner. Tri Bui tbui61
Specified API as followed:
import RTLearner as dt
learner = dt.RTLearner(leaf_size = 1, verbose = False) # constructor
learner.addEvidence(Xtrain, Ytrain) # training step
Y = learner.query(Xtest) # query
'''

import numpy as np
import pandas as pd
from scipy import stats 

class RTLearner:
    def __init__(self, leaf_size = 5, verbose = False):
        super().__init__()
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        return 'tbui61'

    # iterate over rows, set index to root of tree, while tree branch is not a leaf 
    # I set feature to the root feature, check on the split val and go down either the left or right tree 
    # by setting the index there
    def query(self, Xtest):
        Ypred = []
        for row in Xtest:
            rootIndex = 0
            splitIndex = self.tree[rootIndex, 0]
            while (splitIndex != None):
                splitIndex = int(splitIndex)
                if (row[splitIndex] <= self.tree[rootIndex, 1]):
                    rootIndex += int(self.tree[rootIndex, 2])   #follow left tree
                else:
                    rootIndex += int(self.tree[rootIndex, 3])   #follow right tree
                splitIndex = self.tree[rootIndex, 0]
            # when it breaks out of while loop means we reach a leaf
            Ypred.append(self.tree[rootIndex,1])
        Ypred = np.asarray(Ypred)
        return Ypred

    # Add the training dataset. After adding, the function should combine X and Y into 1 ndarray
    def addEvidence(self, Xtrain, Ytrain):
        # The tree is built with both X and Y data (Y = the leaf) => merge X and Y first
        data = np.append(Xtrain, Ytrain.reshape((len(Ytrain),1)), axis=1)
        self.tree = self.buildTree(data)

    # Tree structure: [None(leaf) or splitIndex, splitVal, left tree index, right tree index]
    def buildTree(self, data):
        if (data.shape[0] <= self.leaf_size):
            #Yvalue = np.mean(data[:,-1])
            Yvalue = stats.mode(data[:,-1], axis=None).mode[0]   #classify as mode
            result = np.array([[None, Yvalue, None, None]])
            return result
        # If all data in y are the same then we need to make a leaf (avoid infinite recursion)
        elif (all(x==data[0,-1] for x in data[:,-1])):
            Yvalue = data[0,-1]
            result = np.array([[None, Yvalue, None, None]])
            return result
        else:
            splitIndex = np.random.randint(0,data.shape[1]-1)
            splitVal = np.median(data[:,splitIndex])
            # if all values in splitIndex column = same ie. splitVal won't split anything
            # then we need to make a leaf
            tempData = data[data[:,splitIndex]<=splitVal]
            if (tempData.shape[0] == data.shape[0]):
                #Yvalue = np.mean(data[:,-1])
                Yvalue = stats.mode(data[:,-1], axis=None).mode[0]    #classify as mode
                result = np.array([[None, Yvalue, None, None]])
                return result     
            lefttree = self.buildTree(data[data[:,splitIndex]<=splitVal])
            righttree = self.buildTree(data[data[:,splitIndex]>splitVal])
            root = np.array([[int(splitIndex), splitVal, 1, lefttree.shape[0]+1]])
            root = np.concatenate((root, lefttree, righttree), axis=0)
            return root

