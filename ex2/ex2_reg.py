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

def part2_1():
  data = np.genfromtxt("ex2data2.txt", delimiter=',') #Read in comma separated data
  plotData(data)
  return

def main():

  part2_1()
#  plt.show()

if __name__ == "__main__":
  main()
