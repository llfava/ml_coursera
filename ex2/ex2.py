#!/usr/bin/python

############
# ex2.py
############
# Logistic regression
############


import numpy as np
import matplotlib.pyplot as plt




def main():

  ## Load data; first two columns contains the exam scores, and the third column
  ## contains the label
  data = np.genfromtxt("ex2data1.txt", delimiter=',') #Read in comma separated data
  m, n = np.shape(data)[0], np.shape(data)[1]-1
  X = np.c_[np.ones(m), data[:,:n]]
  y = data[:,n:n+1] #TODO: Note difference between data[:,n] and data[:,n:n+1] - why is this?
  print X




if __name__ == "__main__":
  main()
