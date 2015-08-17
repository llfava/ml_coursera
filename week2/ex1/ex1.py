#!/usr/bin/python

# ex1.py is a modified version of ex1.m from the coursera machine learning course

import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import copy

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
  return x.dot(theta)

def computeCost(x,y,theta):
  m = len(y)

  ##This is the loop version of finding J
  cum_sum = 0
  for i in range(0,m):
    cum_sum += (hypothesis(x[i],theta) -y[i])**2
  cum_sum = (1.0 / (2 * m)) * cum_sum #This is J

  ##This is the vectorized version of finding J
  term = hypothesis(x,theta) - y
  return (term.T.dot(term) / (2 * m))[0,0]#This is J
  #What is this [0,0] factor for??


def gradientDescent(x, y, theta, alpha, iterations):
  m = len(y)
  grad = copy.copy(theta)

  for iteration in range(0,iterations):
    grad -= (alpha / m) * x.T.dot(hypothesis(x,theta) - y)

  return grad


# Do part 2.1
def part2_1():
  data = np.genfromtxt("ex1data1.txt", delimiter=',') #Read in comma-separated data
  X, y = data[:,0], data[:,1]
  m = len(y)
#  plotData(X,y)
  return data

def part2_2(data):
  y = data[:,1] 
  m = len(y)
  y = y.reshape(m,1)
  X = np.c_[np.ones(m),data[:,0]]
  theta = np.zeros((2,1))
  iterations = 1500
  alpha = 0.01

  cost = computeCost(X,y,theta)
  theta = gradientDescent(X, y, theta, alpha, iterations)
  print theta  

  predict1 = np.array([1,3.5]).dot(theta)
  predict2 = np.array([1,7]).dot(theta)
#  print predict1
#  print predict2

#  print y
#  print theta

  #fig = plt.figure()
  #ax1 = plt.subplot(111)

#  plt.plot(X[:,1], y, 'rx')
  plt.plot(X[:,1], X.dot(theta), 'b-')
#  plt.show()

def part2_4():
  data = np.genfromtxt("ex1data1.txt", delimiter=',') #Read in comma-separated data
  X, y = data[:,0], data[:,1]
  m = len(y)
  y = y.reshape(m,1)
  X = np.c_[np.ones(m),data[:,0]]

  #Grid over which we calculate J
  theta0_vals = np.linspace(-10, 20, 100)
  theta1_vals = np.linspace(-1, 10, 100)

  #Initialize J_vals to a matrix of zeros
  J_vals = np.zeros((len(theta0_vals), len(theta1_vals))) 
 
  #Fill out J_vals
  for i, _ in enumerate(theta0_vals):
    for j, _ in enumerate(theta1_vals):
      theta = np.array((theta0_vals[i], theta1_vals[j])).reshape(2,1)
      J_vals[i,j] = computeCost(X, y, theta)

  print theta[0], theta[1]


  R, P = np.meshgrid(theta0_vals, theta1_vals) #Creates grid with these values
  
  fig = plt.figure()
  ax1 = fig.add_subplot(2,1,1, projection='3d') #Need Axes3D for this to work
  ax1.plot_surface(R, P, J_vals)

  ax2 = fig.add_subplot(2,1,2)
  ax2.contour(R,P,J_vals, np.logspace(-2,3,20))
  ax2.plot(theta[0], theta[1], 'rx', markersize = 10)
  plt.show()


def main():
  n = 5
  warmUpExercise(n)

  data = part2_1()
  part2_2(data)
  part2_4()
  return

if __name__=="__main__":
  main()
