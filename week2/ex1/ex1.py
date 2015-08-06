#!/usr/bin/python

# ex1.py is a modified version of ex1.m from the coursera machine learning course

import sys

import numpy as np

# warmUpExercise function to return the 5x5 identity matrix
def warmUpExercise(n):
  A = np.identity(n)
  print A
  
  #Alternatively could use numpy.eye(5) instead

def main():
  n = 5
  warmUpExercise(n)

  return

if __name__=="__main__":
  main()
