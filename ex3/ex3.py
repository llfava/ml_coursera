#!/usr/bin/python

#################
# ex3.py
#################
# Neural networks - Representation (One-vs-all logistic regression)
#################

import numpy as np
import matplotlib.pyplot as plt
import scipy.io #module to load files formatted for other languages (here for matlab files)
import scipy.misc

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

  plt.show()

#Load matlab data file
def part1_1(): 
  mat = scipy.io.loadmat('ex3data1')
  X, y = mat['X'], mat['y']

#Visualize the data
def part1_2():
  mat = scipy.io.loadmat('ex3data1')
  X, y = mat['X'], mat['y']
  displayData(X)  
  

def main():

  part1_1()
  part1_2()

if __name__ == "__main__":
  main()

