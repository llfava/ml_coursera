#!/usr/bin/python
############# 
# ex2_reg.py
############
# Regularized logistic regression
############

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize, scipy.special

def plotData(data):
  pos = data[data[:,2] == 1] #Select X data for y=1
  neg = data[data[:,2] == 0] #Select X data for y=0

  ax1 = plt.subplot(111)
  ax1.set_xlabel("Microchip Test 1")
  ax1.set_ylabel("Microchip Test 2")
  ax1.scatter(pos[:,0], pos[:,1], s=40, c='k', marker='+', label="Accepted")
  ax1.scatter(neg[:,0], neg[:,1], s=40, c='y', marker='o', label="Rejected")
  ax1.legend()
  v = [-1, 1.5, -0.8, 1.2]
  ax1.axis(v)
  return

def mapFeature(x1, x2):
  degrees = 6
  out = np.ones( (np.shape(x1)[0], 1) )

  for i in range(1, degrees+1):
    for j in range(0, degrees):
      term1 = x1 ** (i-j)
      term2 = x2 ** (j)
      m = np.shape(term1)[0]
      term = np.reshape((term1 * term2),(m,1))
      out = np.hstack((out,term))
  return out

def part2_1():
  data = np.genfromtxt("ex2data2.txt", delimiter=',') #Read in comma separated data
  plotData(data)
  plt.show()
  return

def part2_2():
  data = np.genfromtxt("ex2data2.txt", delimiter=',') #Read in comma separated data
  X = mapFeature( data[:,0], data[:,1] )

def main():

  #part2_1()
  part2_2()

if __name__ == "__main__":
  main()
