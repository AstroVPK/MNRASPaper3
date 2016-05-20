import carmcmc as cmcmc
import numpy as np
import numpy.polynomial.polynomial as poly
import math as m
import cmath as cm
import random as rm
import cPickle as cP
from scipy.optimize import fsolve
import pdb
import sys as s
import time

goldenRatio=1.61803398875
fhgt=10.0
fwid=fhgt*goldenRatio
largeFontSize=48
normalFontSize=32
smallFontSize=24
footnoteFontSize=20
scriptFontSize=16
tinyFontSize=12

keplerPath = "/home/vpk24/Documents/MNRASPaper3/Kepler/"
outPath = "/home/vpk24/Documents/MNRASPaper3/Zw229-15Results/PLATINUM/"

keplerObj = "kplr006932990Carini"
redShift = 0.0275
secPerSiderealDay = 86164.09053083
startCad = 11914
endCad = 72529
intTime = 6.019802903
readTime = 0.5189485261
numIntLC = 270
dt = (((intTime+readTime)*numIntLC)/(1.0 + redShift))/secPerSiderealDay

numSamples = 50000

outputSample = open(outPath+'Sample.pkl','rb')
sample = cP.load(outputSample)
outputSample.close()

outputChain = open(outPath+'Chain.pkl','wb')

mu = sample.get_samples('mu')
cP.dump(mu, outputChain)

var = sample.get_samples('var')
cP.dump(var, outputChain)

ar_coefs = sample.get_samples('ar_coefs')
cP.dump(ar_coefs, outputChain)

ma_coefs = sample.get_samples('ma_coefs')
cP.dump(ma_coefs, outputChain)

sigma = sample.get_samples('sigma')
cP.dump(sigma, outputChain)

measerr_scale = sample.get_samples('measerr_scale')
cP.dump(measerr_scale, outputChain)

ar_roots = sample.get_samples('ar_roots')
cP.dump(ar_roots, outputChain)

loglike = sample.get_samples('loglik')
cP.dump(loglike, outputChain)

logpost = sample.get_samples('logpost')
cP.dump(logpost, outputChain)

psd_width = sample.get_samples('psd_width')
cP.dump(psd_width, outputChain)

psd_centroid = sample.get_samples('psd_centroid')
cP.dump(psd_centroid, outputChain)

outputChain.close()