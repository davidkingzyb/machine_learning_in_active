# Machine Learning in Action

**The example of <Machine Learning in Action>**

2019/5/15 by DKZ



## kNN

`distance=sqrt(sum((target-train)**2))`

1. data [x,y,...] labal
2. train data matrix [data,...] and [labal,] 
3. normMat(trainMat) and normVec(targetData)
4. kNN(targetVec,trainMat,labals,k) return nearist labal
    1. calc distance
    2. sort
    3. find max count label

```py
def normMat(dataMat):
    minVals = dataMat.min(0)
    maxVals = dataMat.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataMat))
    m = dataMat.shape[0]
    normDataSet = dataMat - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

def normVec(dataVec,minVals,ranges):
    return (dataVec-minVals)/ranges

def kNN(targetVec, trainMat, labels, k):
    """
    targetVec [num,...] 
    trainMat [[num,...],[num,...],...] 
    labels [str,str,...] train data label
    k int count range
    """
    trainMatSize = trainMat.shape[0]
    diffMat = tile(targetVec, (trainMatSize,1)) - trainMat # targetArr to targetMat [target,...] then [[target-train],...] 
    sqDiffMat = diffMat**2 # [[(target-train)**2]]
    sqDistances = sqDiffMat.sum(axis=1) # [sum([(target-train)**2]),...]
    distances = sqDistances**0.5 # useless?
    sortedDistIndicies = distances.argsort() # sort distance array [index,...]
    # find k nearist train data count label return max
    classCount={} 
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
```

## Decision Tree 

ID3

`entropy=-sum(log2(prob)*prob)`

1. data [feature,...,cls]
2. train data matrix [[feature,...,cls],...] feature labels [feature_name,...]
3. creatTree(trainMat,labels)
    1. get sub matrix by every unique type in features 
    2. calc `entropy*prob` find smallest as best feature
    3. splic sub matrix by best feature
    4. recursive creat sub tree
4. classify by tree
    

```py
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
        uniqueVals = set(featList) # unique type in feature_i {feature_it,...} featurn_i 
        newEntropy = 0.0
        for value in uniqueVals:
            subDataMat = splitDataSet(dataMat, i, value) # featurn_it => [[feature_except_i,...,cls],...]
            prob = len(subDataMat)/float(len(dataMat))
            newEntropy += prob * calcShannonEnt(subDataMat) # smaller better    
        infoGain = baseEntropy - newEntropy     # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       
            bestInfoGain = infoGain         
            bestFeature = i
    return bestFeature                      # returns best feature index

def createTree(dataMat,labels):
    """
    ID3
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
```

## Naive Bayes

```
P(A|B) = P(B|A)P(A)/P(B)
```

1. creat dictionary (a unique word vector)
    - calculate most frequence word and delect from dictionary
    - or remove from stop word list
2. transform wordVec to dataVec 
    - set-of-words model or bag-of-words model
    - mark in or not at dictionary 
    - dataVec to dataMat

```py
def createDictionary(wordMat):
    vocabSet = set([])  #create empty set
    for document in wordMat:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def wordVecToDataVec(dictionary, wordVec):
    returnVec = [0]*len(dictionary)
    for word in wordVec:
        if word in dictionary:
            # returnVec[dictionary.index(word)] = 1
            returnVec[dictionary.index(word)] += 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec    
```

3. train naive bayes 
4. classify

```py
def trainNaiveBayes(trainMat,labels):
    numTrainDocs = len(trainMat)
    numWords = len(trainMat[0])
    pClass1 = sum(labels)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones() 
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    # p0Denom = 0; p1Denom =0                       
    for i in range(numTrainDocs):
        if labels[i] == 1:
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1Vect = log(p1Num/p1Denom)          #change to log() for better distribution
    p0Vect = log(p0Num/p0Denom)          
    # p1Vect = p1Num/p1Denom         
    # p0Vect = p0Num/p0Denom         
    return p0Vect,p1Vect,pClass1

def classifyNB(targetVec, p0Vec, p1Vec, pClass1):
    p1 = sum(targetVec * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(targetVec * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
```
