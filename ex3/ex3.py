#!/usr/bin/python

#################
# ex3.py
#################
# Neural networks - Representation (One-vs-all logistic regression)
#################

import numpy as np
import matplotlib.pyplot as plt
import scipy.io #module to load files formatted for other languages (here for matlab files)

#Load matlab data file
def part1_1(): 
  mat = scipy.io.loadmat('ex3data1')
  X, y = mat['X'], mat['y']

def main():

  part1_1()

if __name__ == "__main__":
  main()

