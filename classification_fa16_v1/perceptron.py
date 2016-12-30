import numpy as np 
import data_classification_utils
from util import raiseNotDefined
import random

class Perceptron(object):
    def __init__(self, categories, numFeatures):
        """categories: list of strings 
           numFeatures: int"""
        self.categories = categories
        self.numFeatures = numFeatures

        """YOUR CODE HERE"""
        self.weights = np.zeros((len(self.categories), self.numFeatures))



    def classify(self, sample):
        """sample: np.array of shape (1, numFeatures)
           returns: category with maximum score, must be from self.categories"""

        """YOUR CODE HERE"""
        scores = np.zeros([len(self.weights)])

        for i in range(len(self.weights)):
          scores[i] = np.dot(self.weights[i], sample)

        index = np.argmax(scores)
        return self.categories[index]


    def train(self, samples, labels):
        """samples: np.array of shape (numFeatures, numSamples)
           labels: list of numSamples strings, all of which must exist in self.categories 
           performs the weight updating process for perceptrons by iterating over each sample once."""

        """YOUR CODE HERE"""
        
        for i in range(len(samples)):
          sample = samples[i]
          label = labels[i]
          yp_index = self.classify(sample)
          if yp_index != label:
            self.weights[label] = self.weights[label] + sample
            self.weights[yp_index] -= sample


          


