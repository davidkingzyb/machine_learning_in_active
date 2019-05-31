'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def createDataSet2():
    dataSet=[
['young','myope','no','reduced','no lenses'],
['young','myope','no','normal','soft'],
['young','myope','yes','reduced','no lenses'],
['young','myope','yes','normal','hard'],
['young','hyper','no','reduced','no lenses'],
['young','hyper','no','normal','soft'],
['young','hyper','yes','reduced','no lenses'],
['young','hyper','yes','normal','hard'],
['pre','myope','no','reduced','no lenses'],
['pre','myope','no','normal','soft'],
['pre','myope','yes','reduced','no lenses'],
['pre','myope','yes','normal','hard'],
['pre','hyper','no','reduced','no lenses'],
['pre','hyper','no','normal','soft'],
['pre','hyper','yes','reduced','no lenses'],
['pre','hyper','yes','normal','no lenses'],
['presbyopic','myope','no','reduced','no lenses'],
['presbyopic','myope','no','normal','no lenses'],
['presbyopic','myope','yes','reduced','no lenses'],
['presbyopic','myope','yes','normal','hard'],
['presbyopic','hyper','no','reduced','no lenses'],
['presbyopic','hyper','no','normal','soft'],
['presbyopic','hyper','yes','reduced','no lenses'],
['presbyopic','hyper','yes','normal','no lenses'],
    ]
    labels=['age','prescript','astigmatic','tearRate']
    return dataSet,labels

def calcShannonEnt(dataMat):
    numEntries = len(dataMat)
    labelCounts = {}
    for featVec in dataMat: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt # -sum(log2(prob)*prob) average infomation
    
def splitDataSet(dataMat, axis, value):
    resultDataMat = []
    for featVec in dataMat:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            resultDataMat.append(reducedFeatVec)
    return resultDataMat
    
def chooseBestFeatureToSplit(dataMat):
    numFeatures = len(dataMat[0]) - 1      #the last column is used for the class
    baseEntropy = calcShannonEnt(dataMat)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataMat] # [feature_i,...]
        uniqueVals = set(featList) # {feature_i_a,...} featurn_i 
        newEntropy = 0.0
        for value in uniqueVals:
            subDataMat = splitDataSet(dataMat, i, value) # featurn_i_a => [[feature_except_i,...,cls],...]
            prob = len(subDataMat)/float(len(dataMat))
            newEntropy += prob * calcShannonEnt(subDataMat) # smaller better    
        infoGain = baseEntropy - newEntropy     # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       
            bestInfoGain = infoGain         
            bestFeature = i
    return bestFeature                      # returns best feature index

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    print('sortedClassCount',sortedClassCount)
    return sortedClassCount[0][0]

def createTree(dataMat,labels):
    """
    dataMat [[feature,...,cls],...]
    labels [feature_name,...] featurn label
    """
    classList = [example[-1] for example in dataMat] # class array [cls,...]
    # stop splitting when all of the classes are equal
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    # stop splitting when there are no more features in dataMat
    if len(dataMat[0]) == 1: 
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataMat)
    bestFeatLabel = labels[bestFeat]
    theTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # subtree
    featValues = [example[bestFeat] for example in dataMat]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        theTree[bestFeatLabel][value] = createTree(splitDataSet(dataMat, bestFeat, value),subLabels)
    return theTree # {feature_label_i:{feature_i_a:subtree|cls,...}}                      
    
def classify(tree,featureLabel,targetVec):
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    featIndex = featureLabel.index(firstStr)
    key = targetVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featureLabel, targetVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(tree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(tree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    

if __name__ == '__main__':
    
    dataSet,labels=createDataSet()
    # dataSet,labels=createDataSet2()
    tree=createTree(dataSet,labels.copy())
    print(tree)
    # target=['presbyopic','hyper','yes','normal']
    target=[1,0]
    classLabel=classify(tree,labels.copy(),target)
    print(classLabel)
