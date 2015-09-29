#!/usr/bin/python

#################
# ex3.py
#################
# Neural networks - Representation (One-vs-all logistic regression)
#################

import numpy as np
import matplotlib.pyplot as plt
import scipy.io #module to load files formatted for other languages (here for matlab files)
import scipy.misc #module containing function to display image from numpy array
import scipy.special #for logistic function

def displayData(X):
  width = 20
  rows, cols = 10, 10
  out = np.zeros(( width * rows, width * cols ))

  rand_indices = np.random.permutation ( 5000 )[0:rows * cols]

  counter = 0
  for y in range(0, rows):
    for x in range(0, cols):
      start_x = x * width
      start_y = y * width
      out[start_x:start_x+width, start_y:start_y+width] = X[rand_indices[counter]].reshape(width, width).T
      counter += 1

  img = scipy.misc.toimage( out )
  figure  = plt.figure()
  axes    = figure.add_subplot(111)
  axes.imshow( img )

  #plt.show()

def sigmoid(z):
#  g = 1.0 / (1.0 + np.exp(-z))
  g = scipy.special.expit(z)
  return g

def computeCost(theta, X, y):
  m = len(y)
  hypo = sigmoid(X.dot(theta.T))
  term1 = np.log(hypo).T.dot(-y)
  term2 = np.log(1-hypo).T.dot(1-y)
  sum = term1 - term2
  return (sum / m).flatten()

def gradientCost(theta, X, y):
  m = len(y)
  sum = X.T.dot(sigmoid(X.dot(theta.T))-y)
  return (sum / m)

def costFunction(theta, X, y):
  cost = computeCost(theta,X,y)
  grad = gradientCost(theta,X,y)
  return cost

######################
# Parts of assignment
######################

#Load matlab data file
def part1_1(): 
  mat = scipy.io.loadmat('ex3data1')
  X, y = mat['X'], mat['y']

#Visualize the data
def part1_2():
  mat = scipy.io.loadmat('ex3data1')
  X, y = mat['X'], mat['y']
  displayData(X)  

#Vectorizing logistic regression
def part1_3():  
  mat = scipy.io.loadmat('ex3data1')
  X, y = mat['X'], mat['y']
  m = len(y)
  X = np.c_[np.ones( (m,1) ), X]

  num_pixels = np.shape(X)[1]
  num_classes = 10

  theta = np.zeros( (num_classes,num_pixels) )



  cost = computeCost(theta, X, y)  
  grad = gradientCost(theta, X, y)  

  print cost
  print np.shape(cost)
  print grad
  print np.shape(grad)
  

def main():

#  part1_1()
  part1_2()
  part1_3()

if __name__ == "__main__":
  main()

