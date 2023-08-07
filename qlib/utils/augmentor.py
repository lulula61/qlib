import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
from augmentts.augmenters.vae import LSTMVAE, VAEAugmenter  


def augmentor(X,y,mu=0,sigma=0.05,iterations=5):
    
  print("Initial shape of X is {}".format(X.shape))
  print("Initial shape of y is {}".format(y.shape))


  iterations = 5 #do the below loop this many times
  mu, sigma = 0, 0.03 #the mean and the standart deviation of the noise
  X_new = np.zeros((X.shape[0],X.shape[1],X.shape[2])) #create sparse arrays
  y_new = np.zeros((y.shape[0],y.shape[1]))

  #Maybe there is a better way to do this without for loops
  for i in range(iterations):
      noise = np.random.normal(mu, sigma, [X.shape[0],X.shape[1],X.shape[2]]) #create the noise matrix for each iteration
      #the noises will have both positive and negative values, which is good to randomly add or subtract values to out original dataset
      Xt = np.multiply(noise,X) + X     #multiply the noise dataset with our dataset and add it to our dataset
      X_new = np.concatenate([X_new, Xt], axis = 0) #concatenate the new matrix to itself
      print("Done with the {}th iteration".format(i)) #print which step are we in 
  X_new = X_new[X.shape[0]:]    #remove the first set of data since it is all zeros

  for i in range(iterations): #since the concat order is the same. the one hot vectors can be repeated in the same pattern.
      y_new = np.concatenate([y, y_new], 0)

  y_new = y_new[~np.all(y_new == 0, axis=1)] #remove rows that only contain 0's

  #save our augmented dataset with the values we defined above. This comes in handy if you want to do multiple
  #augmentation trials and keep track of the parameters you used
  #be sure to edit your dataset path accordingly
  print(X_new.shape)
  print(y_new.shape)
  
  return X_new,y_new

def DA_Jitter(X, sigma=0.05):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise

def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,120)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise

def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    data = []
    for i in range(0,120):
        cs = CubicSpline(xx[:,i],yy[:,i])
        data.append(cs(x_range))
    
    return np.array(data).transpose()

def DA_MagWarp(X, sigma=0.2):
    return X * GenerateRandomCurves(X, sigma)

def AugmentTS(data, sigma=0.2):
    data = np.expand_dims(data,axis=1)
    vae = LSTMVAE(series_len=120)
    augmenter = VAEAugmenter(vae)
    augmenter.fit(data, epochs=50, batch_size=32)

    samples = augmenter.sample(n=10000)
    samples = np.squeeze(samples)
    
    return samples

def DA_Permutation(X, nPerm=4, minSegLength=1000):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        X_new[pp:pp+len(x_temp),:] = x_temp
        pp += len(x_temp)
    return(X_new)
