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

## Logistics Regres

```
sigmoid(inX)=1/(1+exp(-inX))

weights=weights+alpha*error*dataMat.transpose()

```

1. calculate sigmoid result
2. find error and new weight
3. regres weight

```py
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

# random optimized
def stocGradAscent(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0
```

## SVN

## Linear Regression

### Ordinary Least Squares Methods

```
err=sum(yi-xi.T*w)^2
w=(X.T*X)^-1*X.T*y #min err

y=x*w
```

```py
def standRegres(xData,yArr):
    xMat = mat(xData); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

y=xData*ws
corr=corrcoef(y.T,yArr)
```

### Locally Weighted Linear Regression


`w=(X.T*W*X)^-1*X.T*W*y`


```py
def lwlr(testPoint,xData,yArr,k=1.0): # k smaller near point weight biger
    xMat = mat(xData); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws
```

### Ridge Regression

feature > sample

`w=(X.T*X+lamda*I)^-1*X.T*y`

```py
def ridgeRegres(xMat,yMat,lam=0.2):# xMat=mat(xData) yMat=mat(yArr).T
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws
```

### Stage Regres

```py
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        # print(ws.T)
        lowestError = inf; 
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
    return ws.T
```

## kMeans

1. for each data point assign it to the closest centroid
2. for each centriod recalculate it to mean
3. loop until centriod dont change 

```py
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        # print(centroids)
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
    return centroids, clusterAssment# mat [index,distance] 
```

```py
# dichotomy optimize
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            # print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        # print('the bestCentToSplit is: ',bestCentToSplit)
        # print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment
```

## PCA

```py
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects #transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat
```

## SVD

```
U,Sigma,VT=svd(datamat)
lowdatamat=Uk*Sigmak*VTk
```
