#!/usr/bin/python

############
# ex2.py
############
# Logistic regression
############


import numpy as np
import matplotlib.pyplot as plt


def part1_1(data):
  pos = data[data[:,2] == 1] #Select X data for y=1
  neg = data[data[:,2] == 0] #Select X data for y=0

  ax1 = plt.subplot(111)
  ax1.set_xlabel("Exam 1 score")
  ax1.set_ylabel("Exam 2 score")
  ax1.scatter(pos[:,0], pos[:,1], s=40, c='k', marker='+', label="Admitted")
  ax1.scatter(neg[:,0], neg[:,1], s=40, c='y', marker='o', label="Not admitted")
  ax1.legend()
#  plt.show()

def sigmoid(z):
  g = 1.0 / (1.0 + np.exp(-z))
  return g

def main():

  ## Load data; first two columns contains the exam scores, and the third column
  ## contains the label
  data = np.genfromtxt("ex2data1.txt", delimiter=',') #Read in comma separated data
  m, n = np.shape(data)[0], np.shape(data)[1]-1
  X = np.c_[np.ones(m), data[:,:n]]
  y = data[:,n:n+1] #TODO: Note difference between data[:,n] and data[:,n:n+1] - why is this?

  part1_1(data) #Plot the data
  
  g = sigmoid(data)
  print g

if __name__ == "__main__":
  main()
