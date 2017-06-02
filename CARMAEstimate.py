import carmcmc as cmcmc
import numpy as np
import math as m
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#from mpl_settings import *
import cPickle as cP
#import pdb
#import sys as s

s1 = 2
s2 = 9
fwid = 13
fhgt = 13
dotsPerInch = 300
nbins = 250
# set_plot_params(fontsize=12)

keplerPath = "/home/vpk24/Documents/Research/code/tag/Paper3/Kepler/"
outPath = "/home/vpk24/Documents/Research/code/tag/Paper3/Results/"

keplerObj = "kplr006932990Carini"
startCad = 11914
endCad = 72529
#endCad = 13914

data = np.loadtxt(keplerPath+keplerObj+'/'+keplerObj+'-calibrated.dat', skiprows=2)
t = list()
y = list()
yerr = list()

for i in range(np.where(data[:, 0] == startCad)[0][0], np.where(data[:, 0] == endCad)[0][0]+1):
    if (data[i, 2] != 0.0):
        t.append(data[i, 2])
        y.append(data[i, 3])
        yerr.append(data[i, 4])
t = np.array(t)
y = np.array(y)
yerr = np.array(yerr)

model = cmcmc.CarmaModel(t, y, yerr)

pmax = 10

MLE, pqlist, AICc_list = model.choose_order(pmax, njobs=16)  # , njobs=-1)

sample = model.run_mcmc(50000)

'''sample.assess_fit()

psd_low, psd_hi, psd_mid, frequencies = sample.plot_power_spectrum(percentile=95.0, nsamples=5000)'''

outputSample = open(outPath+'Sample.pkl', 'wb')
cP.dump(sample, outputSample, -1)
outputSample.close()
