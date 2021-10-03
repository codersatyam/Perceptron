import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

class perceptron:
  def __init__(self,eta,epochs):
    #eta is learning rate
    np.random.seed(42)
    self.weights=np.random.randn(3) * 1e-4 # Random weight initilization
    print(f"Initial weights before Training: {self.weights}") 
    self.eta=eta
    self.epochs=epochs

  def activationfunction(self,input,weights):

    z=np.dot(input,weights) # z=x * w
    return np.where(z>0,1,0)
    
    
  def fit(self,x,y):
    self.x=x
    self.y=y
    
    x_with_bias=np.c_[self.x,-np.ones((len(self.x),1))] #concatination
    print(f"x_with_bias :{x_with_bias}")
    for epochs in range(self.epochs):
      print("--"*10)
      print(f"for epoch :{epochs}")
      print("--"*10)
      y_hat=self.activationfunction(x_with_bias,self.weights)  # forward propogation
      print(f"predicted value:{y_hat}")  
      self.error=self.y -y_hat
      print(f"error:\n{self.error}")
      self.weights=self.weights + self.eta * np.dot(x_with_bias.T,self.error) # backward propogation
      print(f"updated weights after epoch:{epochs}/{self.epochs}:{self.weights}")
      print("##"*10)

  def predict(self,x):
    x_with_bias=np.c_[x,np.ones((len(x),1))]
    return self.activationfunction(x_with_bias,self.weights)

  def total_loss(self):
    total_loss=np.sum(self.error)
    print(f"total_loss:{total_loss}")
    return total_loss
