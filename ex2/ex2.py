#!/usr/bin/python

############
# ex2.py
############
# Logistic regression
############


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize, scipy.special

def plotData(data):
  pos = data[data[:,2] == 1] #Select X data for y=1
  neg = data[data[:,2] == 0] #Select X data for y=0

  ax1 = plt.subplot(111)
  ax1.set_xlabel("Exam 1 score")
  ax1.set_ylabel("Exam 2 score")
  ax1.scatter(pos[:,0], pos[:,1], s=40, c='k', marker='+', label="Admitted")
  ax1.scatter(neg[:,0], neg[:,1], s=40, c='y', marker='o', label="Not admitted")
  ax1.legend()
  v = [30, 100, 30, 100]
  ax1.axis(v)  
  return

def plotBoundary(data, x, theta):
  plotData(data)
  plot_x = np.array( [min(x[:,1]), max(x[:,1])] ) #only need to points to plot line
  plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0]) #get y-values using y-int form
  plt.plot( plot_x, plot_y)
  return


def sigmoid(z):
  g = 1.0 / (1.0 + np.exp(-z))
  return g

def computeCost(theta, x, y):
  m = len(y)
  sum = np.log(sigmoid(x.dot(theta))).T.dot(-y)-np.log(1-sigmoid(x.dot(theta))).T.dot(1-y)
  return (sum / m).flatten()

def gradientCost(theta, x, y):
  m = len(y)
  sum = x.T.dot(sigmoid(x.dot(theta))-y)
  return (sum / m).flatten()

def costFunction(theta, x, y):
  cost = computeCost(theta,x,y)
  grad = gradientCost(theta,x,y)
  return cost

def findMinTheta(theta, x, y):
#  result = scipy.optimize.fmin(costFunction, x0=theta, args=(x,y), 
#                               maxiter=500, full_output=True) 
#  print result[0], result[1]

  result = scipy.optimize.minimize(costFunction, x0=theta, args=(x,y), method='Nelder-Mead',
                                   options={'maxiter':500})
  return result.x, result.fun
  #return result[0], result[1]

def part1_1(): #Plot the data
  data = np.genfromtxt("ex2data1.txt", delimiter=',') #Read in comma separated data
  plotData(data) 

def part1_2():
  data = np.genfromtxt("ex2data1.txt", delimiter=',') #Read in comma separated data
  m, n = np.shape(data)[0], np.shape(data)[1]-1
  X = np.c_[np.ones(m), data[:,:n]]
  y = data[:,n:n+1] #TODO: Note difference between data[:,n] and data[:,n:n+1] - why is this?
  theta = np.zeros((n+1,1))

  theta, cost = findMinTheta(theta,X,y)
  plotBoundary(data,X,theta)



def main():

  ## Load data; first two columns contains the exam scores, and the third column
  ## contains the label

  #part1_1()
  part1_2()
  plt.show()




  



if __name__ == "__main__":
  main()
