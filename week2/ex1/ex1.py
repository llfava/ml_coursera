#!/usr/bin/python

# ex1.py is a modified version of ex1.m from the coursera machine learning course

import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# warmUpExercise function to return the 5x5 identity matrix
def warmUpExercise(n):
  A = np.identity(n)
  #Alternatively could use numpy.eye(5) instead

#TODO: Fix y-axis tick mark frequency and labels
def plotData(x,y): 
  fig = plt.figure()
  ax1 = plt.subplot(111)
  ax1.plot(x,y,'rx')
  ax1.set_xlabel('Population of City in 10,000s')
  ax1.set_ylabel('Profit in $10,000s')
  v = [4, 24, -5, 25]
  ax1.axis(v)
  #plt.show()

def hypothesis(x, theta):
  print x.dot(theta)
  return x.dot(theta)

def computeCost(x,y,theta):
  m = len(y)
  cum_sum = 0
  for i in range(0,m):
    cum_sum += (hypothesis(x[i],theta) -y[i])**2
  cum_sum = (1.0 / (2 * m)) * cum_sum
  return cum_sum

# Do part 2.1
def part2_1():
  data = np.genfromtxt("ex1data1.txt", delimiter=',') #Read in comma-separated data
  X, y = data[:,0], data[:,1]
  m = len(y)
  plotData(X,y)
  return data

def part2_2(data):
  y = data[:,1] 
  m = len(y)
  y = y.reshape(m,1)
  X = np.c_[np.ones(m),data[:,0]]
  theta = np.zeros((2,1))
  print theta
  iterations = 1500
  alpha = 0.01

  cost = computeCost(X,y,theta)

  
def main():
  n = 5
  warmUpExercise(n)

  data = part2_1()
  part2_2(data)
  return

if __name__=="__main__":
  main()
