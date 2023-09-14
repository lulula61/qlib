import numpy as np
from scipy.interpolate import CubicSpline      # for warping
from augmentts.augmenters.vae import LSTMVAE, VAEAugmenter  
import numpy as np

class Augmentator:
    def __init__(self):
        pass
    
    def add_peak(self, ts, peak_num=200,len_peak=4, peak_factor=2.):
        lenn = len(ts)
        ts_augm = ts.astype(float).copy()
        for i in range(peak_num):
            ind_start = np.random.randint(0, lenn-len_peak-1)
            ind_end = ind_start + len_peak
            ts_augm[ind_start:ind_end] *= peak_factor
        return ts_augm
    
    # def add_flipping(self, ts, b=None):
    #     if b is None:
    #         b = np.mean(ts)

    #     return np.where(2 * b - ts > 0, 2 * b - ts,  0)

    # def DA_Jitter02(self, X, mu=0, sigma=0.5):
    #     noise = np.random.normal(mu, sigma, [X.shape[0], X.shape[1]])
    #     X_new = np.multiply(noise, X) + X
    #     return X_new

    def DA_Jitter(self, X, sigma=0.05):
        myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
        return X+myNoise

    def DA_Scaling(self, X, sigma=0.1):
        scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,120)
        myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
        return X*myNoise

    def GenerateRandomCurves(self, X, sigma=0.2, knot=4):
        xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
        x_range = np.arange(X.shape[0])
        data = []
        for i in range(0,120):
            cs = CubicSpline(xx[:,i],yy[:,i])
            data.append(cs(x_range))
        
        return np.array(data).transpose()

    def DA_MagWarp(self, X, sigma=0.2):
        return X * self.GenerateRandomCurves(X, sigma)

    def AugmentTS(self, data, sigma=0.2):
        data = np.expand_dims(data,axis=1)
        vae = LSTMVAE(series_len=120)
        augmenter = VAEAugmenter(vae)
        augmenter.fit(data, epochs=50, batch_size=32)

        samples = augmenter.sample(n=10000)
        samples = np.squeeze(samples)
        
        return samples

    def DA_Permutation(self, X, nPerm=4, minSegLength=1000):
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
    
    def DistortTimesteps(self, X, sigma=0.2):
        tt = self.GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
        tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
        # Make the last value to have X.shape[0]
        len = X.shape[0]-1
        feature_num = X.shape[1]
        for i in range(0,feature_num):
            t_scale = len/tt_cum[-1,i]
            tt_cum[:,i] = tt_cum[:,i]*t_scale

        return tt_cum


    def DA_TimeWarp(self, data, sigma=0.2):
        tt_new = self.DistortTimesteps(data, sigma)
        data_new = np.zeros(data.shape)
        x_range = np.arange(data.shape[0])
        feature_num = data.shape[1]
        for k in range(0, feature_num):
            data_new[:, k] = np.interp(x_range, tt_new[:, k], data[:, k])
        return data_new
