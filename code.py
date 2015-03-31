import os
import sys
import pdb
import glob
import math
import numpy
import random
from sets import Set
import scipy as scipy
from scipy.misc import *
from scipy import linalg

# This function loads the images, divides the data intro training and test set
def LoadImages(directory, split):
    # get a list of all the picture filenames
    gifs = glob.glob(directory + '/*.gif')
    # uncomment the below line when trying an unknown file
    #extraGif = glob.glob("/media/cosmos/Data/College Notes/M.Tech/Semester 4/Statistical Methods in AI/Project - Face Recognition/2.gif")
    classMap = {}
    testGIF = []
    allLabels = []
    testLabels = []
    trainingGIF = []
    trainingLabels = []
    for i in range(len(gifs)):
        if random.random() < split:
            trainingGIF.append(gifs[i])
            l = gifs[i].split("/");
            labelName = l[len(l)-1].split(".")[0][-2:]
            trainingLabels.append(labelName)
            allLabels.append(labelName)
        else:
            testGIF.append(gifs[i])
            l = gifs[i].split("/");
            labelName = l[len(l)-1].split(".")[0][-2:]
            testLabels.append(labelName)
            allLabels.append(labelName)
    # uncomment the below 2 lines when trying an unknown file
    #testGIF.append(extraGif[0])
    #testLabels.append("un")
    #allLabels.append("un")
    trainingImgs = numpy.array([imread(i, True).flatten() for i in trainingGIF])
    testImgs = numpy.array([imread(i, True).flatten() for i in testGIF])
    # creating a list of class labels
    allLabels = set(allLabels)
    noOfClasses = len(allLabels)
    sortedLabels = []
    for i in allLabels:
        sortedLabels.append(i)
    sortedLabels = sorted(sortedLabels)
    # creating a mapping for confusion matrix
    j = 0
    for i in sortedLabels:
        classMap[i] = j
        j = j + 1
    return trainingImgs,testImgs,trainingLabels,testLabels,noOfClasses,classMap

def EuclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def Mahanalobis(x, y):
    return scipy.spatial.distance.mahalanobis(x,y,np.linalg.inv(np.cov(x,y)))

# Run Principal Component Analysis on the input data.
# INPUT  : data    - an n x p matrix
# OUTPUT : e_faces - 
#          weights -
#          mu      -
def PCA(data):
    # mean
    mu = numpy.mean(data, 0)
    # mean adjust the data
    ma_data = data - mu
    # run SVD
    e_faces, sigma, v = linalg.svd(ma_data.transpose(), full_matrices=False)
    # compute weights for each image
    weights = numpy.dot(ma_data, e_faces)
    return e_faces, weights, mu

# This function calculates the weight of the test data
def InputWeight(testData, mu, e_faces):
    ma_data = testData - mu
    weights = numpy.dot(ma_data, e_faces)
    return weights

# Reconstruct an image using the given number of principal components.
def Reconstruct(imgIDx, e_faces, weights, mu, npcs):
	# dot weights with the eigenfaces and add to mean
	recon = mu + numpy.dot(weights[imgIDx, 0:npcs], e_faces[:, 0:npcs].T)
	return recon

# Saves the image in the given directory
def SaveImage(outDIR, subdir, imgID, imgDims, data):
	directory = outDIR + "/" + subdir
	if not os.path.exists(directory): os.makedirs(directory)
	imsave(directory + "/image_" + str(imgID) + ".jpg", data.reshape(imgDims))

# Prints the final results
def PrintResults(wrongPredictedClassCount, unknownLabels, accuracy, correctlyClassifiedDistances, maxDist, confusionMatrix):
    print "Number of Wrongly Predicted Labels:",wrongPredictedClassCount
    print "Number of Unknown Labels:",unknownLabels
    print "Accuracy:",accuracy,"%"
    print "Max Distances among correcltly classified:",correctlyClassifiedDistances[len(correctlyClassifiedDistances) - 1]
    print "Max Distances among all:",maxDist
    PrintConfusionMatrix(confusionMatrix)

# Prints the confustion matrix
def PrintConfusionMatrix(confusionMatrix):
    print "Confusion Matrix:"
    for i in range(0, len(confusionMatrix)):
        print confusionMatrix[i]

# Predicts the class of test data
def PredictLabelsFromTestData(testData, noOfClasses, mu, e_faces, trainingWeights, testLabels, trainingLabels, classMap, thresholdDistance):
    correctlyClassifiedDistances = []
    confusionMatrix = [[0 for i in xrange(noOfClasses)] for i in xrange(noOfClasses)]
    wrongPredictedClassCount = 0
    unknownLabels = 0
    for i in range(len(testData)):
        testWeight=InputWeight(testData[i],mu,e_faces)
        distances = []
        for x in range(len(trainingWeights)):
            dist = EuclideanDistance(testWeight, trainingWeights[x], len(testWeight))
            #dist = Mahanalobis(testWeight, trainingWeights[x])
            distances.append(dist)
        actualLabel = testLabels[i]
        predictedLabel = ""
        minDist = sys.maxint
        maxDist = -sys.maxint
        for j in range(len(distances)):
            if minDist > distances[j]:
                minDist = distances[j]
                predictedLabel = trainingLabels[j]
            if maxDist < distances[j]:
                maxDist = distances[j]
        confusionMatrix[classMap[actualLabel]][classMap[predictedLabel]] = confusionMatrix[classMap[actualLabel]][classMap[predictedLabel]] + 1
        #print "Actual class:",actualLabel
        #print "Predicted class:",predictedLabel
        #print "Min Dist:",minDist
        #print "---------"
        if minDist >= thresholdDistance:
            predictedLabel = "Unknown"
            unknownLabels = unknownLabels + 1
        elif predictedLabel != actualLabel:
            wrongPredictedClassCount = wrongPredictedClassCount + 1
        else:
            correctlyClassifiedDistances.append(minDist)
    # calculate accuracy
    accuracy = (1 - wrongPredictedClassCount / float(len(testData))) * 100
    correctlyClassifiedDistances.sort()
    PrintResults(wrongPredictedClassCount, unknownLabels, accuracy, correctlyClassifiedDistances, maxDist, confusionMatrix)
    return accuracy

def main():
    inDIR  = "/media/cosmos/Data/College Notes/M.Tech/Semester 4/Statistical Methods in AI/Project - Face Recognition/yalefaces/yalefaces"
    outDIR = "/media/cosmos/Data/College Notes/M.Tech/Semester 4/Statistical Methods in AI/Project - Face Recognition/yalefaces/SampleOutput"
    imgDims = (243, 320)
    split = 0.8
    noOfDimensions = 100
    thresholdDistance = 18000.0
    trainingData, testData, trainingLabels, testLabels, noOfClasses, classMap = LoadImages(inDIR, split)
    e_faces, trainingWeights, mu = PCA(trainingData)
	# save mean photo
    imsave(outDIR + "/mean.gif", mu.reshape(imgDims))	
    # save each eigenface as an image
    for i in range(e_faces.shape[1]):
		SaveImage(outDIR, "eigenfaces", i, imgDims, e_faces[:,i])
	# reconstruct each face image using an increasing number of principal components
    reconstructed = []
    for p in range(trainingData.shape[0]):
        reconstructed.append(Reconstruct(p, e_faces, trainingWeights, mu, len(trainingWeights)))
        imgID = p
        SaveImage(outDIR, "reconstructed/" + str(p), imgID,imgDims, reconstructed[p])
    # Predicting Classes for test data
    accuracy = PredictLabelsFromTestData(testData, noOfClasses, mu, e_faces, trainingWeights, testLabels, trainingLabels, classMap, thresholdDistance)
    return accuracy

total = 0.0
for i in range(0, 5):
    total = total + main()
    print "_____________________________________________"
print "Mean Accuracy:",total / 5,"%"