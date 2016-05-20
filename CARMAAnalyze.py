import carmcmc as cmcmc
import numpy as np
import numpy.polynomial.polynomial as poly
import math as m
import cmath as cm
import random as rm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import gridspec
import matplotlib.cm as colormap
import matplotlib.mlab as mlab
from mpl_settings import *
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
set_plot_params(fontfamily='serif',fontstyle='normal',fontvariant='normal',fontweight='normal',fontstretch='normal',fontsize=largeFontSize,useTex='True')

keplerPath = "/home/vish/Documents/MNRASPaper3/Kepler/"
outPath = "/home/vish/Documents/MNRASPaper3/Zw229-15Results/PLATINUM/"

keplerObj = "kplr006932990Carini"
redShift = 0.0275
secPerSiderealDay = 86164.09053083
startCad = 11914
endCad = 72529
intTime = 6.019802903
readTime = 0.5189485261
numIntLC = 270
dt = (((intTime+readTime)*numIntLC)/(1.0 + redShift))/secPerSiderealDay

makeEPS=False

makePDF=False

makeAllFigs=False

makeFig1=True
legendFig1PSD=True
legendFig1LC=True
plotMockLC=False
ptFactor=5000

makeFig2=True
legendFig2GFunc=True
numTimes=250

makeFig3=True

makeFig4=True
legendFig4distPSD=True
numFreqs=250

makeFig5=True
sampleFactor=1
nBinsFig5 = 50

makeFig6=True
nBinsFig6 = 50

makeFig7=True
nBinsFig7 = 50

makeFig8=True

makeFig9=True

def eval_greens_func(tVal,ar_root0,ar_root1):
	return (np.exp(ar_root0[:]*tVal[:]) - np.exp(ar_root1[:]*tVal[:]))/(ar_root0[:]-ar_root1[:])

def gfunc_etime(tVal,ar_root0,ar_root1,gfunc_efold):
	return gfunc_efold - (np.exp(ar_root0*tVal) - np.exp(ar_root1*tVal))/(ar_root0-ar_root1)

def get_greens_func(t,percentile,ar_roots):
	t=np.array(t)
	num_t = t.shape[0]
	num_samples=ar_roots.shape[0]
	temp_gfunc=np.zeros(num_samples,dtype=complex)
	gfunc_mid=np.zeros(num_t,dtype=complex)
	gfunc_high=np.zeros(num_t,dtype=complex)
	gfunc_low=np.zeros(num_t,dtype=complex)
	lower = (100.0 - percentile)/2.0  # lower and upper intervals for credible region
	upper = 100.0 - lower
	for i in xrange(num_t):
		temp_gfunc[:] = (np.exp(ar_roots[:,0]*t[i])-np.exp(ar_roots[:,1]*t[i]))/(ar_roots[:,0]-ar_roots[:,1])
		gfunc_mid[i]=complex(np.median(temp_gfunc[:].real),np.median(temp_gfunc[:].imag))
		gfunc_high[i]=complex(np.percentile(temp_gfunc[:].real,upper),np.percentile(temp_gfunc[:].imag,upper))
		gfunc_low[i]=complex(np.percentile(temp_gfunc[:].real,lower),np.percentile(temp_gfunc[:].imag,lower))
	return (gfunc_low[:].real, gfunc_high[:].real, gfunc_mid[:].real, t)

def get_greens_func_max(t_maxes,ar_roots):
	val = np.zeros((ar_roots.shape[0]))
	val[:] = (np.exp(ar_roots[:,0].real*t_maxes[:]) - np.exp(ar_roots[:,1]*t_maxes[:]))/(ar_roots[:,0] - ar_roots[:,1])
	return val

def get_dist_psd(f,percentile,ma_coefs):
	num_samples = ma_coefs.shape[0]
	num_f = f.shape[0]
	dist_psd_mid = np.zeros(num_f,dtype=complex)
	dist_psd_high = np.zeros(num_f,dtype=complex)
	dist_psd_low = np.zeros(num_f,dtype=complex)
	temp_psd = np.zeros(num_samples,dtype=complex)
	lower = (100.0 - percentile)/2.0  # lower and upper intervals for credible region
	upper = 100.0 - lower
	OneOverTwoPi = 1.0/(2.0*m.pi)
	for i in xrange(num_f):
		evalFreq = (4.0*m.pi*m.pi*f[i]*f[i])
		temp_psd[:] = OneOverTwoPi*(poly_coefs[:,1]*poly_coefs[:,1]*evalFreq + poly_coefs[:,0]*poly_coefs[:,0])
		dist_psd_mid[i] = np.median(temp_psd[:])
		dist_psd_high[i] = np.percentile(temp_psd[:],upper)
		dist_psd_low[i] = np.percentile(temp_psd[:],lower)
	return (dist_psd_low[:].real, dist_psd_high[:].real, dist_psd_mid[:].real, f)

def get_t_max(ar_roots,nsamples):
	num_samples = ar_roots.shape[0]
	ratio = int(num_samples/nsamples)
	return (np.log(ar_roots[::ratio,0]/ar_roots[::ratio,1]))/(ar_roots[::ratio,1]-ar_roots[::ratio,0])

data = np.loadtxt(keplerPath+keplerObj+'/'+keplerObj+'-calibrated.dat',skiprows=2)
t = list()
y = list()
yerr = list()
cadNo = list()

for i in range(np.where(data[:,0]==startCad)[0][0],np.where(data[:,0]==endCad)[0][0]+1):
	if (data[i,2] != 0.0):
		cadNo.append(data[i,0])
		t.append(data[i,2])
		y.append(data[i,3])
		yerr.append(data[i,4])
t = np.array(t)
y = np.array(y)
yerr = np.array(yerr)

mean_yerr = np.mean(yerr)

numCads = cadNo[-1]-cadNo[0]
numSamples = 50000

outputSample = open(outPath+'Sample.pkl','rb')
sample = cP.load(outputSample)
outputSample.close()

mu = sample.get_samples('mu')
var = sample.get_samples('var')

ar_coefs = sample.get_samples('ar_coefs')
ma_coefs = sample.get_samples('ma_coefs')
sigma = sample.get_samples('sigma')
measerr_scale = sample.get_samples('measerr_scale')

ar_roots = sample.get_samples('ar_roots')

loglike = -1.0*sample.get_samples('loglik')
logpost = sample.get_samples('logpost')
best_loglike = np.min(loglike)

psd_width = sample.get_samples('psd_width')
psd_centroid = sample.get_samples('psd_centroid')

poly_coefs = ma_coefs*sigma

timeFig1CStart=time.time()
if ((makeFig1==True) or (makeFig2==True) or (makeFig4==True) or (makeAllFigs==True)):
	if ((makeFig1==True) or (makeAllFigs==True)):
		time_predict = np.linspace(sample.time.min(), sample.time.max(), numCads/ptFactor)
		predicted_mean, predicted_var = sample.predict(time_predict, bestfit='map')
		predicted_low = predicted_mean - np.sqrt(predicted_var)
		predicted_high = predicted_mean + np.sqrt(predicted_var)

	nLevel = 2.0*dt*np.mean(measerr_scale*m.pow(mean_yerr,2.0))
	psd_low, psd_high, psd_mid, frequencies = sample.plot_power_spectrum(percentile=95.0, nsamples=numSamples/ptFactor, doShow=False)

	currf = 0
	for i in xrange(frequencies.shape[0]):
		currf = i
		if (psd_mid[i] < nLevel):
			break
	slope = (psd_mid[i-1] - psd_mid[i])/(frequencies[i-1] - frequencies[i])
	intercept = psd_mid[i] - frequencies[i]*slope
	noise_lim_freq = (nLevel - intercept)/slope
	noise_lim_time = 1.0/noise_lim_freq
timeFig1CFinish=time.time()
print "Fig1 Calculations: %f (sec) i.e. %f (min)"%((timeFig1CFinish-timeFig1CStart),(timeFig1CFinish-timeFig1CStart)/60.0)

timeFig2CStart=time.time()
if ((makeFig2==True) or (makeAllFigs==True)):
	times = np.logspace(m.log10(dt), m.log10(numCads*dt), num=numTimes)
	gfunc_low, gfunc_high, gfunc_mid, times = get_greens_func(times,95.0,ar_roots)
timeFig2CFinish=time.time()
print "Fig2 Calculations: %f (sec) i.e. %f (min)"%((timeFig2CFinish-timeFig2CStart),(timeFig2CFinish-timeFig2CStart)/60.0)

timeFig3CStart=time.time()
if ((makeFig3==True) or (makeFig1==True) or (makeAllFigs==True)):
	omega = np.sqrt(ar_coefs[:,2])
	freq = omega/(2.0*m.pi)
	tscale = 1.0/freq
	zeta = ar_coefs[:,1]/(2.0*omega[:])
	percentile = 95.0
	lower = (100.0 - percentile)/2.0  # lower and upper intervals for credible region
	upper = 100.0 - lower
	tscale_mid = np.median(tscale)
	zeta_mid = np.median(zeta)
	tscale_high = np.percentile(tscale,upper)
	zeta_high = np.percentile(zeta,upper)
	tscale_low = np.percentile(tscale,lower)
	zeta_low = np.percentile(zeta,lower)
timeFig3CFinish=time.time()
print "Fig3 Calculations: %f (sec) i.e. %f (min)"%((timeFig3CFinish-timeFig3CStart),(timeFig3CFinish-timeFig3CStart)/60.0)

timeFig4CStart=time.time()
if ((makeFig4==True) or (makeAllFigs==True)):
	distfreqs = np.logspace(m.log10(1.0/(dt*numCads)), m.log10(1.0/dt), numFreqs)
	dist_psd_low, dist_psd_high, dist_psd_mid, distfreqs = get_dist_psd(distfreqs,95.0,ma_coefs)
timeFig4CFinish=time.time()
print "Fig4 Calculations: %f (sec) i.e. %f (min)"%((timeFig4CFinish-timeFig4CStart),(timeFig4CFinish-timeFig4CStart)/60.0)

timeFig5CStart=time.time()
if ((makeFig5==True) or (makeFig2==True) or (makeAllFigs==True)):
	nfoldings = 1
	t_maxes = get_t_max(ar_roots,nsamples=numSamples/sampleFactor)
	gfunc_maxes=np.zeros(t_maxes.shape[0])
	gfunc_maxes[:] = eval_greens_func(t_maxes[:].real,ar_roots[:,0].real,ar_roots[:,1].real)
	gfunc_efold = gfunc_maxes/m.pow(m.e,nfoldings)
	tau_initial_guess = 50.0
	tau_solution = np.zeros(t_maxes.shape[0])
	for i in xrange(t_maxes.shape[0]):
		tau_solution[i] = fsolve(gfunc_etime, tau_initial_guess,args=(ar_roots[i,0].real,ar_roots[i,1].real,gfunc_efold[i]))
	percentile = 95.0
	lower = (100.0 - percentile)/2.0  # lower and upper intervals for credible region
	upper = 100.0 - lower
	t_max_med = np.median(t_maxes[:].real)
	t_max_high = np.percentile(t_maxes[:].real,upper)
	t_max_low = np.percentile(t_maxes[:].real,lower)
	tau_solution_med = np.median(tau_solution)
	tau_solution_high = np.percentile(tau_solution,upper)
	tau_solution_low = np.percentile(tau_solution,lower)
timeFig5CFinish=time.time()
print "Fig5 Calculations: %f (sec) i.e. %f (min)"%((timeFig5CFinish-timeFig5CStart),(timeFig5CFinish-timeFig5CStart)/60.0)

timeFig6CStart=time.time()
if ((makeFig6==True) or (makeFig4==True) or (makeAllFigs==True)):
	t_turns = 2.0*m.pi*poly_coefs[:,1]/poly_coefs[:,0]
timeFig6CFinish=time.time()
print "Fig6 Calculations: %f (sec) i.e. %f (min)"%((timeFig6CFinish-timeFig6CStart),(timeFig6CFinish-timeFig6CStart)/60.0)

timeFig7CStart=time.time()
'''if ((makeFig7==True) or (makeAllFigs==True)):
	t_turns = 2.0*m.pi*poly_coefs[:,1]/poly_coefs[:,0]'''
timeFig7CFinish=time.time()
print "Fig7 Calculations: %f (sec) i.e. %f (min)"%((timeFig7CFinish-timeFig7CStart),(timeFig7CFinish-timeFig7CStart)/60.0)

timeFig8CStart=time.time()
if ((makeFig8==True) or ((makeFig1==True) and (plotMockLC==True)) or (makeAllFigs==True)):
	p_sim = 2 # order of the AR polynomial
	sample_num = rm.randint(0,numSamples-1)
	sigmay_sim = sigma[sample_num]  # dispersion in the time series
	mu_sim = mu[sample_num]
	ar_roots_sim = ar_roots[sample_num,:]
	ma_coefs_sim = ma_coefs[sample_num,:]
	y_sim_no_noise = mu_sim + cmcmc.carma_process(t, sigmay_sim*sigmay_sim, ar_roots_sim, ma_coefs=ma_coefs_sim)
	yerr_sim = np.median(yerr)*m.sqrt(np.median(measerr_scale))
	y_sim = y_sim_no_noise + np.random.normal(0.0,yerr_sim,t.shape[0])
timeFig8CFinish=time.time()
print "Fig8 Calculations: %f (sec) i.e. %f (min)"%((timeFig8CFinish-timeFig8CStart),(timeFig8CFinish-timeFig8CStart)/60.0)

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

############################################################## Figure 1 #######################################################################

timeFig1PStart=time.time()
if ((makeFig1==True) or (makeAllFigs==True)):

	fig1 = plt.figure(2,figsize=(fwid,fhgt))
	gs = gridspec.GridSpec(500, 525) 

	ax1 = fig1.add_subplot(gs[:,:])
	ax1.errorbar(0,0,yerr=0,fmt='.',capsize=0,color='#d95f02',markeredgecolor='none',label=r'observed Zw 229-15 light curve')
	ax1.errorbar(t,y,yerr=yerr,fmt='.',capsize=0,color='#d95f02',markeredgecolor='none',zorder=10)
	if (plotMockLC==True):
		ax1.errorbar(0,0,yerr=0,fmt='.',capsize=0,color='#66a61e',markeredgecolor='none',label=r'simulated Zw 229-15 light curve')
		ax1.errorbar(t,y_sim,yerr=yerr_sim,fmt='.',capsize=0,color='#66a61e',markeredgecolor='none',zorder=-5)
	ax1.fill_between(time_predict, predicted_low, predicted_high, color='#b3b3b3',zorder=0)
	ax1.plot(time_predict, predicted_mean, '-',label=r'est. Zw 229-15 light curve',color='#666666',zorder=5)
	ax1.set_xlim(sample.time.min(),sample.time.max())
	ax1.set_ylim(4.83*m.pow(10.0,-7.0),8.83*m.pow(10.0,-7.0))
	ax1.set_xlabel(r'$t$~{\large (MJD)}')
	ax1.set_ylabel(r'$F$~{\large (arb. units)}')
	ax1.annotate(r'C-ARMA(2,1)',xy=(0.135,0.1),xycoords='axes fraction',textcoords='axes fraction',ha='center',va='center',multialignment='center',fontsize=32)
	ax1.annotate(r'AICc: -2333427.16925',xy=(0.135,0.05),xycoords='axes fraction',textcoords='axes fraction',ha='center',va='center',multialignment='center',fontsize=24)

	handles1,labels1=ax1.get_legend_handles_labels()
	LCVarPatch=mpatches.Patch(color='#b3b3b3',label=r'$\langle [\mathrm{Zw 229-15 LC}]^2 \rangle$')
	handles1.append(LCVarPatch)
	labels1.append(r'std dev of est. Zw 229-15 light curve')
	if (plotMockLC==True):
		newOrder1=[1,0,3,2]
	else:
		newOrder1=[1,0,2]
	handles1=[handles1[i] for i in newOrder1]
	labels1=[labels1[i] for i in newOrder1]
	if (legendFig1LC==True):
		ax1.legend(handles1,labels1,loc=4,ncol=1,fancybox=True,fontsize=scriptFontSize)

	'''colStart=100
	rowStart=50
	numRows=225
	numCols=numRows # The plot dimensions are already in the Golden ratio.'''
	fig2 = plt.figure(1,figsize=(fwid,fhgt))
	gs = gridspec.GridSpec(500, 525) 
	ax1 = fig1.add_subplot(gs[rowStart:rowStart+numRows,colStart:colStart+numCols])
	ax1.loglog(frequencies,psd_mid,'-', color='#666666',label='median PSD',zorder=5,subsx=[],subsy=[])
	ax1.fill_between(frequencies, psd_low, psd_high, color='#b3b3b3',zorder=0)

	ax1.axvspan(xmin=noise_lim_freq, xmax=np.max(frequencies), ymin=0, ymax=1, color='#fbb4ae', zorder=0)
	ax1.hlines(y=nLevel, xmin=np.min(frequencies), xmax=np.max(frequencies), colors='#e41a1c', linestyles='solid',linewidths=2,label=r'Noise level',zorder=10)
	ax1.set_title(r'Power Spectral Density')

	handles2,labels2=ax1.get_legend_handles_labels()
	PSDVarPatch=mpatches.Patch(color='#b3b3b3',label=r'$95^{\mathrm{th}}$-percentile of PSD')
	noisePatch=mpatches.Patch(color='#fbb4ae',label=r'noise-dominates')
	handles2.append(PSDVarPatch)
	handles2.append(noisePatch)
	labels2.append(r'$95^{\mathrm{th}}$-percentile of PSD')
	labels2.append(r'noise dominates')
	newOrder2=[0,2,1,3]
	handles2=[handles2[i] for i in newOrder2]
	labels2=[labels2[i] for i in newOrder2]
	if (legendFig1PSD==True):
		ax1.legend(handles2,labels2,loc=1,ncol=1,fancybox=True,fontsize=tinyFontSize)

	ax1.set_xlim(np.min(frequencies),np.max(frequencies))
	numXTicks=6
	xTicks=np.zeros(numXTicks)
	for i in xrange(numXTicks-1):
		xTicks[i]=m.pow(10.0,i-3)
	xTicks[5]=noise_lim_freq
	ax1.set_xticks(xTicks)
	xLabels = [item.get_text() for item in ax1.get_xticklabels()]
	for i in xrange(numXTicks-1):
		xLabels[i]=r'${\scriptscriptstyle 10^{%d}}$'%(m.log10(xTicks[i]))
	xLabels[5]=r'${\scriptscriptstyle t_{\mathrm{noise}}}$'
	ax1.set_xticklabels(xLabels)
	ax1.set_xlabel(r'${\scriptstyle \log_{10}\nu }$ {\normalsize (d$^{-1}$)}')

	ax1.set_ylim(np.min(psd_low),np.max(psd_high))
	numYTicks=3
	yTicks=np.zeros(numYTicks)
	for i in xrange(numYTicks):
		yTicks[i]=m.pow(10.0,4*i-22)
	ax1.set_yticks(yTicks)
	yLabels = [item.get_text() for item in ax1.get_yticklabels()]
	for i in xrange(numYTicks):
		yLabels[i]=r'${\scriptscriptstyle 10^{%d}}$'%(m.log10(yTicks[i]))
	ax1.set_yticklabels(yLabels)
	ax1.set_ylabel(r'${\scriptstyle \log_{10}S_{FF} }$ {\normalsize (arb. units)}')

	plt.tight_layout()
	if (plotMockLC==True):
		figure1JPG=outPath+'Zw229-15_mock+LC.jpg'
		plt.savefig(figure1JPG,dpi=300)
		if (makePDF==True):
			figure1PDF=outPath+'Zw229-15_mock+LC.pdf'
			plt.savefig(figure1PDF,dpi=300)
		if (makeEPS==True):
			figure1EPS=outPath+'Zw229-15_mock+LC.eps'
			plt.savefig(figure1EPS,dpi=300)
	else:
		figure1JPG=outPath+'Zw229-15_LC.jpg'
		plt.savefig(figure1JPG,dpi=300)
		if (makePDF==True):
			figure1PDF=outPath+'Zw229-15_LC.pdf'
			plt.savefig(figure1PDF,dpi=300)
		if (makeEPS==True):
			figure1EPS=outPath+'Zw229-15_LC.eps'
			plt.savefig(figure1EPS,dpi=300)
timeFig1PFinish=time.time()
print "Fig1 Plots: %f (sec) i.e. %f (min)"%((timeFig1PFinish-timeFig1PStart),(timeFig1PFinish-timeFig1PStart)/60.0)

############################################################ End Figure 1 ######################################################################

############################################################## Figure 2 #######################################################################

timeFig2PStart=time.time()
if ((makeFig2==True) or (makeAllFigs==True)):
	fig2 = plt.figure(3,figsize=(fwid,fhgt))
	numRows=1000
	numCols=numRows
	gs1 = gridspec.GridSpec(numRows, numCols)

	ax2 = fig2.add_subplot(gs1[:,:])
	ax2.plot(times, gfunc_mid, '-', color='#666666',label='median $G(t)$',zorder=5,subsx=[],subsy=[])
	ax2.fill_between(times, gfunc_low, gfunc_high, color='#b3b3b3',zorder=0)
	ax2.axvspan(xmin=np.min(times), xmax=noise_lim_time, ymin=0, ymax=1, color='#fbb4ae', zorder=0)
	ax2.vlines(x=t_max_med, ymin=m.pow(10.0,-11.0), ymax=m.pow(10.0,1.0), colors='#d95f02', linestyles='--',linewidths=2,label=r'$t_{\mathrm{max}}$',zorder=10)
	ax2.vlines(x=tau_solution_med, ymin=m.pow(10.0,-11.0), ymax=m.pow(10.0,1.0), colors='#d95f02', linestyles='-.',linewidths=2,label=r'$t_{\mathrm{e-fold}}$',zorder=10)

	ax2.set_xlabel(r'{${\textstyle t}$~{\large (d)}')
	ax2.set_xlim(np.min(times),np.max(times))
	numXTicks=6
	xTicks=np.zeros(numXTicks)
	for i in xrange(numXTicks-1):
		xTicks[i]=m.pow(10.0,i-1)
	xTicks[5]=noise_lim_time
	ax2.set_xticks(xTicks)
	xLabels = [item.get_text() for item in ax2.get_xticklabels()]
	for i in xrange(numXTicks-1):
		xLabels[i]=r'${\scriptscriptstyle 10^{%d}}$'%(m.log10(xTicks[i]))
	xLabels[5]=r'${\scriptstyle t_{\mathrm{noise}}}$'
	ax2.set_xticklabels(xLabels)

	ax2.set_ylabel(r'${\scriptstyle  \log_{10}G(t) }$~{\normalsize (arb. units)}')
	ax2.set_ylim(m.pow(10.0,-11.0),m.pow(10.0,1.0))
	numYTicks=3
	yTicks=np.zeros(numYTicks)
	for i in xrange(numYTicks):
		yTicks[i]=m.pow(10.0,-4*i-2)
	ax2.set_yticks(yTicks)
	yLabels = [item.get_text() for item in ax2.get_yticklabels()]
	for i in xrange(numYTicks):
		yLabels[i]=r'${\scriptscriptstyle 10^{%d}}$'%(m.log10(yTicks[i]))
	ax2.set_yticklabels(yLabels)

	handles2,labels2=ax2.get_legend_handles_labels()
	GFuncVarPatch=mpatches.Patch(color='#b3b3b3',label=r'$95^{\mathrm{th}}$-percentile of $G(t)$')
	noisePatch=mpatches.Patch(color='#fbb4ae',label=r'noise-dominates')
	handles2.append(GFuncVarPatch)
	handles2.append(noisePatch)
	labels2.append(r'$95^{\mathrm{th}}$-percentile of $G(t)$')
	labels2.append(r'noise dominates')
	newOrder2=[0,3,1,2,4]
	handles2=[handles2[i] for i in newOrder2]
	labels2=[labels2[i] for i in newOrder2]
	if (legendFig2GFunc==True):
		ax2.legend(handles2,labels2,loc=3,ncol=1,fancybox=True,fontsize=scriptFontSize)

	colStart=175
	rowStart=175
	numRows=625
	numCols=numRows # The plot dimensions are already in the Golden ratio.

	ax1 = fig2.add_subplot(gs1[rowStart:rowStart+numRows,colStart:colStart+numCols])

	ax1.set_title(r'$t_{\mathrm{max}}$ v/s $t_{\mathrm{e-fold}}$',fontsize=footnoteFontSize)
	scatPlot = ax1.scatter(t_maxes[:].real,tau_solution[:],c=best_loglike-loglike,marker='.',cmap=colormap.gist_rainbow_r,linewidth = 0,zorder=20)

	colStart=205
	rowStart=475
	numRows=180
	numCols=numRows

	ax3 = fig2.add_subplot(gs1[rowStart:rowStart+numRows,colStart:colStart+numCols])
	ax3.spines['top'].set_color('none')
	ax3.spines['bottom'].set_color('none')
	ax3.spines['left'].set_color('none')
	ax3.spines['right'].set_color('none')
	ax3.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
	numCBarTicks=3
	cBarTicks=np.zeros(numCBarTicks)
	for i in xrange(numCBarTicks):
		cBarTicks[i]=-4.5*i-1.5
	cBar = plt.colorbar(scatPlot, ax=ax3, orientation='horizontal',ticks=cBarTicks,format=r'$\scriptscriptstyle %2.1f$')
	cBar.set_label(r'Relative Likelihood',fontsize=scriptFontSize)

	ax1.set_xlim(5.0,6.3)
	numXTicks=3
	xTicks=np.zeros(numXTicks)
	for i in xrange(numXTicks):
		xTicks[i]=0.4*i+5.2
	ax1.set_xticks(xTicks)
	xLabels = [item.get_text() for item in ax1.get_xticklabels()]
	for i in xrange(numXTicks):
		xLabels[i]=r'${\scriptscriptstyle %2.1f}$'%(xTicks[i])
	ax1.set_xticklabels(xLabels)
	ax1.set_xlabel(r'$t_{\mathrm{max}}$~{\large (d)}')

	ax1.set_ylim(40.0,100.0)
	numYTicks=3
	yTicks=np.zeros(numYTicks)
	for i in xrange(numYTicks):
		yTicks[i]=20*i+50
	ax1.set_yticks(yTicks)
	yLabels = [item.get_text() for item in ax1.get_yticklabels()]
	for i in xrange(numYTicks):
		yLabels[i]=r'${\scriptscriptstyle %2.1f}$'%(yTicks[i])
	ax1.set_yticklabels(yLabels)
	ax1.set_ylabel(r'$t_{\mathrm{e-fold}}$~{\large (d)}')

	ax1.annotate(r'$t_{\mathrm{max}} = %3.1f^{+%2.1f}_{-%2.1f}$ (d)'%(t_max_med,t_max_high-t_max_med,t_max_med-t_max_low),xy=(0.7,0.85),xycoords='axes fraction',textcoords='axes fraction',ha='center',va='center',multialignment='center',fontsize=24)
	ax1.annotate(r'$t_{\mathrm{e-fold}} = %2.1f^{+%2.1f}_{-%2.1f}$ (d)'%(tau_solution_med,tau_solution_high-tau_solution_med,tau_solution_med-tau_solution_low),xy=(0.71,0.75),xycoords='axes fraction',textcoords='axes fraction',ha='center',va='center',multialignment='center',fontsize=24)

	plt.tight_layout()
	figure2JPG=outPath+'Zw229-15_GF.jpg'
	plt.savefig(figure2JPG,dpi=300)
	if (makePDF==True):
		figure2PDF=outPath+'Zw229-15_GF.pdf'
		plt.savefig(figure2PDF,dpi=300)
	if (makeEPS==True):
		figure2EPS=outPath+'Zw229-15_GF.eps'
		plt.savefig(figure2EPS,dpi=300)
timeFig2PFinish=time.time()
print "Fig2 Plots: %f (sec) i.e. %f (min)"%((timeFig2PFinish-timeFig2PStart),(timeFig2PFinish-timeFig2PStart)/60.0)

############################################################ End Figure 2 ######################################################################

############################################################## Figure 3 #######################################################################

timeFig3PStart=time.time()
if ((makeFig3==True) or (makeAllFigs==True)):

	fig3 = plt.figure(4,figsize=(fwid,fhgt))
	numRows=1000
	numCols=numRows
	gs1 = gridspec.GridSpec(numRows, numCols)

	ax1 = fig3.add_subplot(gs1[:,:])
	scatPlot = ax1.scatter(ar_roots[:,0].real,ar_roots[:,1].real,c=best_loglike-loglike,marker='.',cmap=colormap.gist_rainbow_r,linewidth = 0)

	ax1.set_xlim(-0.85,-0.5)
	numXTicks=3
	xTicks=np.zeros(numXTicks)
	for i in xrange(numXTicks):
		xTicks[i]=-0.1*i-0.575
	ax1.set_xticks(xTicks)
	xLabels = [item.get_text() for item in ax1.get_xticklabels()]
	for i in xrange(numXTicks):
		xLabels[i]=r'${\scriptscriptstyle %4.3f}$'%(xTicks[i])
	ax1.set_xticklabels(xLabels)
	ax1.set_xlabel(r'${\scriptstyle \rho_{1}}$')# {\large (d$^{-1}$)}')

	ax1.set_ylim(-0.0275,-0.01)
	numYTicks=3
	yTicks=np.zeros(numYTicks)
	for i in xrange(numYTicks):
		yTicks[i]=-0.005*i-0.0125
	ax1.set_yticks(yTicks)
	yLabels = [item.get_text() for item in ax1.get_yticklabels()]
	for i in xrange(numYTicks):
		yLabels[i]=r'${\scriptscriptstyle %4.3f}$'%(yTicks[i])
	ax1.set_yticklabels(yLabels)
	ax1.set_ylabel(r'${\scriptstyle \rho_{2}}$')# {\large (d$^{-1}$)}')

	numCBarTicks=5
	cBarTicks=np.zeros(numCBarTicks)
	for i in xrange(numCBarTicks):
		cBarTicks[i]=-3.0*i-1.5
	cBar = plt.colorbar(scatPlot, ax=ax1, orientation='vertical',ticks=cBarTicks,format=r'$\scriptstyle %2.1f$')
	cBar.set_label(r'Relative Likelihood',fontsize=footnoteFontSize)

	ax1.annotate(r'$T_{\mathrm{dHO}} = %3.1f^{+%2.1f}_{-%2.1f}$ (d)'%(tscale_mid,tscale_high-tscale_mid,tscale_mid-tscale_low),xy=(0.7125,0.25),xycoords='axes fraction',textcoords='axes fraction',ha='center',va='center',multialignment='center',fontsize=24)
	ax1.annotate(r'$\zeta_{\mathrm{dHO}} = %3.2f^{+%3.2f}_{-%3.2f}$'%(zeta_mid,zeta_high-zeta_mid,zeta_mid-zeta_low),xy=(0.7,0.15),xycoords='axes fraction',textcoords='axes fraction',ha='center',va='center',multialignment='center',fontsize=24)


	plt.tight_layout()
	figure3JPG=outPath+'Zw229-15_DampedHO.jpg'
	plt.savefig(figure3JPG,dpi=300)
	if (makePDF==True):
		figure3PDF=outPath+'Zw229-15_DampedHO.pdf'
		plt.savefig(figure3PDF,dpi=300)
	if (makeEPS==True):
		figure3EPS=outPath+'Zw229-15_DampedHO.eps'
		plt.savefig(figure3EPS,dpi=300)
timeFig3PFinish=time.time()
print "Fig3 Plots: %f (sec) i.e. %f (min)"%((timeFig3PFinish-timeFig3PStart),(timeFig3PFinish-timeFig3PStart)/60.0)

############################################################ End Figure 3 ######################################################################

############################################################## Figure 4 #######################################################################

timeFig4PStart=time.time()
if ((makeFig4==True) or (makeAllFigs==True)):
	fig4 = plt.figure(5,figsize=(fwid,fhgt))
	numRows=1000
	numCols=numRows
	gs1 = gridspec.GridSpec(numRows, numCols)

	ax1 = fig4.add_subplot(gs1[:,:])
	ax1.loglog(distfreqs, dist_psd_mid, '-', color='#666666',label=r'median $S_{uu}$',zorder=5,subsx=[],subsy=[])
	ax1.fill_between(distfreqs, dist_psd_low, dist_psd_high, color='#b3b3b3',zorder=0)
	ax1.vlines(x=np.median(t_turns[:].real), ymin=m.pow(10.0,-17.25), ymax=m.pow(10.0,-13), colors='#d95f02', linestyles='--',linewidths=2,label=r'$t_{\mathrm{turn}}$',zorder=10)
	ax1.axvspan(xmin=noise_lim_freq, xmax=np.max(distfreqs), ymin=0, ymax=1, color='#fbb4ae', zorder=0)

	ax1.set_xlim(np.min(distfreqs),np.max(distfreqs))
	numXTicks=6
	xTicks=np.zeros(numXTicks)
	for i in xrange(numXTicks-1):
		xTicks[i]=m.pow(10.0,i-3)
	xTicks[5]=noise_lim_freq
	ax1.set_xticks(xTicks)
	xLabels = [item.get_text() for item in ax1.get_xticklabels()]
	for i in xrange(numXTicks-1):
		xLabels[i]=r'${\scriptscriptstyle 10^{%d}}$'%(m.log10(xTicks[i]))
	xLabels[5]=r'${\scriptstyle t_{\mathrm{noise}}}$'
	ax1.set_xticklabels(xLabels)
	ax1.set_xlabel(r'{${\textstyle \nu}$~{\large (d$^{-1}$)}')

	ax1.set_ylim(m.pow(10.0,-17.25),m.pow(10.0,-13))
	numYTicks=5
	yTicks=np.zeros(numYTicks)
	for i in xrange(numYTicks):
		yTicks[i]=m.pow(10.0,-i-13)
	ax1.set_yticks(yTicks)
	yLabels = [item.get_text() for item in ax1.get_yticklabels()]
	for i in xrange(numYTicks):
		yLabels[i]=r'${\scriptscriptstyle 10^{%d}}$'%(m.log10(yTicks[i]))
	ax1.set_yticklabels(yLabels)
	ax1.set_ylabel(r'${\scriptstyle  \log_{10}S_{uu}}$~{\normalsize (arb. units)}')

	handles2,labels2=ax1.get_legend_handles_labels()
	distPSDVarPatch=mpatches.Patch(color='#b3b3b3',label=r'$95^{\mathrm{th}}$-percentile of $S_{uu}$')
	noisePatch=mpatches.Patch(color='#fbb4ae',label=r'noise-dominates')
	handles2.append(distPSDVarPatch)
	handles2.append(noisePatch)
	labels2.append(r'$95^{\mathrm{th}}$-percentile of $S_{uu}$')
	labels2.append(r'noise dominates')
	newOrder2=[0,2,1,3]
	handles2=[handles2[i] for i in newOrder2]
	labels2=[labels2[i] for i in newOrder2]
	if (legendFig4distPSD==True):
		ax1.legend(handles2,labels2,loc=4,ncol=1,fancybox=True,fontsize=scriptFontSize)

	colStart=160
	rowStart=395
	numRows=250
	numCols=numRows

	ax3 = fig4.add_subplot(gs1[rowStart:rowStart+numRows,colStart:colStart+numCols])
	ax3.spines['top'].set_color('none')
	ax3.spines['bottom'].set_color('none')
	ax3.spines['left'].set_color('none')
	ax3.spines['right'].set_color('none')
	ax3.tick_params(labelcolor='white', top='off', bottom='off', left='off', right='off')

	colStart=110
	rowStart=100
	numRows=625
	numCols=numRows # The plot dimensions are already in the Golden ratio.

	ax2 = fig4.add_subplot(gs1[rowStart:rowStart+numRows,colStart:colStart+numCols])
	scatPlot = ax2.scatter(poly_coefs[:,0],poly_coefs[:,1],c=best_loglike-loglike,marker='.',cmap=colormap.gist_rainbow_r,linewidth = 0)
	ax2.set_title(r'MA Polynomial Coefficients',fontsize=footnoteFontSize)

	numCBarTicks=5
	cBarTicks=np.zeros(numCBarTicks)
	for i in xrange(numCBarTicks):
		cBarTicks[i]=-3.0*i-1.5
	cBar = plt.colorbar(scatPlot, ax=ax3, orientation='horizontal',ticks=cBarTicks,format=r'$\scriptscriptstyle %2.1f$')
	cBar.set_label(r'Relative Likelihood',fontsize=scriptFontSize)

	ax2.set_xlim(np.min(poly_coefs[:,0]),np.max(poly_coefs[:,0]))
	ax2.get_xaxis().get_major_formatter().set_useOffset(False)
	numXTicks=3
	xTicks=np.zeros(numXTicks)
	for i in xrange(numXTicks):
		xTicks[i]=(0.4*i+6.4)*m.pow(10.0,-9.0)
	ax2.set_xticks(xTicks)
	xLabels = [item.get_text() for item in ax2.get_xticklabels()]
	trial = ax2.get_xaxis().get_offset_text()
	for i in xrange(numXTicks):
		xLabels[i]=r'${\scriptscriptstyle %2.1f \times 10^{9}}$'%(xTicks[i]*m.pow(10.0,9.0))
	ax2.set_xticklabels(xLabels)
	ax2.set_xlabel(r'${\scriptstyle \beta_{2}}$')# {\large (d$^{-1}$)}')

	ax2.set_ylim(np.min(poly_coefs[:,1]),np.max(poly_coefs[:,1]))
	ax2.get_yaxis().get_major_formatter().set_useOffset(False)
	numYTicks=4
	yTicks=np.zeros(numYTicks)
	for i in xrange(numYTicks):
		yTicks[i]=(0.04*i+1.12)*m.pow(10.0,-9.0)
	ax2.set_yticks(yTicks)
	yLabels = [item.get_text() for item in ax2.get_yticklabels()]
	for i in xrange(numYTicks):
		yLabels[i]=r'${\scriptscriptstyle %3.2f \times 10^{9}}$'%(yTicks[i]*m.pow(10.0,9.0))
	ax2.set_yticklabels(yLabels)
	ax2.set_ylabel(r'${\scriptstyle \beta_{1}}$')# {\large (d$^{-1}$)}')


	plt.tight_layout()
	figure4JPG=outPath+'Zw229-15_distPSD.jpg'
	plt.savefig(figure4JPG,dpi=300)
	if (makePDF==True):
		figure4PDF=outPath+'Zw229-15_distPSD.pdf'
		plt.savefig(figure4PDF,dpi=300)
	if (makeEPS==True):
		figure4EPS=outPath+'Zw229-15_distPSD.eps'
		plt.savefig(figure4EPS,dpi=300)
timeFig4PFinish=time.time()
print "Fig4 Plots: %f (sec) i.e. %f (min)"%((timeFig4PFinish-timeFig4PStart),(timeFig4PFinish-timeFig4PStart)/60.0)

############################################################ End Figure 4 ######################################################################

############################################################## Figure 5 #######################################################################

timeFig5PStart=time.time()
if ((makeFig5==True) or (makeAllFigs==True)):
	fig5 = plt.figure(6,figsize=(fwid,fhgt))
	numRows=1000
	numCols=numRows
	gs1 = gridspec.GridSpec(numRows, numCols)

	ax1 = fig5.add_subplot(gs1[:,:])

	percentile = 95.0
	lower = (100.0 - percentile)/2.0  # lower and upper intervals for credible region
	upper = 100.0 - lower
	t_max_mid = np.median(t_maxes[:].real)
	t_max_high = np.percentile(t_maxes[:].real,upper)
	t_max_low = np.percentile(t_maxes[:].real,lower)
	nums, bins, patches = ax1.hist(t_maxes[:].real, nBinsFig5, normed=0, edgecolor='#666666',facecolor='#b3b3b3')

	ax1.axvline(t_max_mid,0,1,linestyle='-.',linewidth=2, color='#d95f02')
	ax1.axvline(t_max_high,0,1,linestyle='--',linewidth=2, color='#d95f02')
	ax1.axvline(t_max_low,0,1,linestyle='--',linewidth=2, color='#d95f02')

	ax1.set_xlabel(r'{${\scriptstyle t_{\mathrm{max}}}$~{\normalsize (d)}')
	ax1.set_xlim(5.15,6.15)
	numXTicks=3
	xTicks=np.zeros(numXTicks)
	xTicks[0]=t_max_low
	xTicks[1]=t_max_mid
	xTicks[2]=t_max_high
	ax1.set_xticks(xTicks)
	xLabels = [item.get_text() for item in ax1.get_xticklabels()]
	for i in xrange(numXTicks):
		xLabels[i]=r'${\scriptscriptstyle %3.2f}$'%(xTicks[i])
	ax1.set_xticklabels(xLabels)

	ax1.set_ylabel(r'Counts',fontsize=footnoteFontSize)
	ax1.set_ylim(0.0,4000.0)
	numYTicks=5
	yTicks=np.zeros(numYTicks)
	for i in xrange(numYTicks):
		yTicks[i]=1000*i
	ax1.set_yticks(yTicks)
	yLabels = [item.get_text() for item in ax1.get_yticklabels()]
	for i in xrange(numYTicks):
		yLabels[i]=r'${\scriptscriptstyle %d}$'%(yTicks[i])
	ax1.set_yticklabels(yLabels)

	plt.tight_layout()
	figure5JPG=outPath+'Zw229-15_tmax.jpg'
	plt.savefig(figure5JPG,dpi=300)
	if (makePDF==True):
		figure5PDF=outPath+'Zw229-15_tmax.pdf'
		plt.savefig(figure5PDF,dpi=300)
	if (makeEPS==True):
		figure5EPS=outPath+'Zw229-15_tmax.eps'
		plt.savefig(figure5EPS,dpi=300)

	'''fig9 = plt.figure(10,figsize=(fwid,fhgt))
	numRows=1000
	numCols=numRows
	gs1 = gridspec.GridSpec(numRows, numCols)

	ax1 = fig9.add_subplot(gs1[:,:])
	scatPlot = ax1.scatter(t_maxes[:].real,tau_solution[:],c=best_loglike-loglike,marker='.',cmap=colormap.gist_rainbow_r,linewidth = 0)

	ax1.set_xlim(5.0,6.3)
	numXTicks=3
	xTicks=np.zeros(numXTicks)
	for i in xrange(numXTicks):
		xTicks[i]=0.4*i+5.2
	ax1.set_xticks(xTicks)
	xLabels = [item.get_text() for item in ax1.get_xticklabels()]
	for i in xrange(numXTicks):
		xLabels[i]=r'${\scriptscriptstyle %2.1f}$'%(xTicks[i])
	ax1.set_xticklabels(xLabels)
	ax1.set_xlabel(r'$t_{\mathrm{max}}$~{\large (d)}')

	ax1.set_ylim(40.0,100.0)
	numYTicks=3
	yTicks=np.zeros(numYTicks)
	for i in xrange(numYTicks):
		yTicks[i]=20*i+50
	ax1.set_yticks(yTicks)
	yLabels = [item.get_text() for item in ax1.get_yticklabels()]
	for i in xrange(numYTicks):
		yLabels[i]=r'${\scriptscriptstyle %3.2f}$'%(yTicks[i])
	ax1.set_yticklabels(yLabels)
	ax1.set_ylabel(r'$t_{\mathrm{e-fold}}$~{\large (d)}')

	numCBarTicks=5
	cBarTicks=np.zeros(numCBarTicks)
	for i in xrange(numCBarTicks):
		cBarTicks[i]=-3.0*i-1.5
	cBar = plt.colorbar(scatPlot, ax=ax1, orientation='vertical',ticks=cBarTicks,format=r'$\scriptstyle %2.1f$')
	cBar.set_label(r'Relative Likelihood',fontsize=footnoteFontSize)

	ax1.annotate(r'$t_{\mathrm{max}} = %3.1f^{+%2.1f}_{-%2.1f}$ (d)'%(t_max_med,t_max_high-t_max_med,t_max_med-t_max_low),xy=(0.31,0.25),xycoords='axes fraction',textcoords='axes fraction',ha='center',va='center',multialignment='center',fontsize=24)
	ax1.annotate(r'$t_{\mathrm{e-fold}} = %3.2f^{+%3.2f}_{-%3.2f}$'%(tau_solution_med,tau_solution_high-tau_solution_med,tau_solution_med-tau_solution_low),xy=(0.3,0.15),xycoords='axes fraction',textcoords='axes fraction',ha='center',va='center',multialignment='center',fontsize=24)

	plt.tight_layout()
	figure9JPG=outPath+'Zw229-15_tmax_tefold.jpg'
	plt.savefig(figure9JPG,dpi=300)
	if (makePDF==True):
		figure9PDF=outPath+'Zw229-15_tmax_tefold.pdf'
		plt.savefig(figure9PDF,dpi=300)
	if (makeEPS==True):
		figure9EPS=outPath+'Zw229-15_tmax_tefold.eps'
		plt.savefig(figure9EPS,dpi=300)'''

timeFig5PFinish=time.time()
print "Fig5 Plots: %f (sec) i.e. %f (min)"%((timeFig5PFinish-timeFig5PStart),(timeFig5PFinish-timeFig5PStart)/60.0)

############################################################ End Figure 5 ######################################################################

############################################################## Figure 6 #######################################################################

timeFig6PStart=time.time()
if ((makeFig6==True) or (makeAllFigs==True)):
	fig6 = plt.figure(7,figsize=(fwid,fhgt))
	numRows=1000
	numCols=numRows
	gs1 = gridspec.GridSpec(numRows, numCols)

	ax1 = fig6.add_subplot(gs1[:,:])
	t_turn_mid = np.median(t_turns[:].real)
	percentile = 95.0
	lower = (100.0 - percentile)/2.0  # lower and upper intervals for credible region
	upper = 100.0 - lower
	t_turn_high = np.percentile(t_turns[:].real,upper)
	t_turn_low = np.percentile(t_turns[:].real,lower)
	nums, bins, patches = ax1.hist(t_turns[:].real, nBinsFig6, normed=0, edgecolor='#666666',facecolor='#b3b3b3')

	ax1.axvline(t_turn_mid,0,1,linestyle='-.',linewidth=2, color='#d95f02')
	ax1.axvline(t_turn_high,0,1,linestyle='--',linewidth=2, color='#d95f02')
	ax1.axvline(t_turn_low,0,1,linestyle='--',linewidth=2, color='#d95f02')

	ax1.set_xlabel(r'{${\scriptstyle t_{\mathrm{turn}}}$~{\normalsize (d)}')
	ax1.set_xlim(0.98,1.18)
	numXTicks=3
	xTicks=np.zeros(numXTicks)
	xTicks[0]=t_turn_low
	xTicks[1]=t_turn_mid
	xTicks[2]=t_turn_high
	ax1.set_xticks(xTicks)
	xLabels = [item.get_text() for item in ax1.get_xticklabels()]
	for i in xrange(numXTicks):
		xLabels[i]=r'${\scriptscriptstyle %3.2f}$'%(xTicks[i])
	ax1.set_xticklabels(xLabels)

	ax1.set_ylabel(r'Counts',fontsize=footnoteFontSize)
	ax1.set_ylim(0.0,4000.0)
	numYTicks=5
	yTicks=np.zeros(numYTicks)
	for i in xrange(numYTicks):
		yTicks[i]=1000*i
	ax1.set_yticks(yTicks)
	yLabels = [item.get_text() for item in ax1.get_yticklabels()]
	for i in xrange(numYTicks):
		yLabels[i]=r'${\scriptscriptstyle %d}$'%(yTicks[i])
	ax1.set_yticklabels(yLabels)

	plt.tight_layout()
	figure5JPG=outPath+'Zw229-15_tturn.jpg'
	plt.savefig(figure5JPG,dpi=300)
	if (makePDF==True):
		figure5PDF=outPath+'Zw229-15_tturn.pdf'
		plt.savefig(figure5PDF,dpi=300)
	if (makeEPS==True):
		figure5EPS=outPath+'Zw229-15_tturn.eps'
		plt.savefig(figure5EPS,dpi=300)
timeFig6PFinish=time.time()
print "Fig6 Plots: %f (sec) i.e. %f (min)"%((timeFig6PFinish-timeFig6PStart),(timeFig6PFinish-timeFig6PStart)/60.0)

############################################################ End Figure 6 ######################################################################

############################################################## Figure 7 #######################################################################

timeFig7PStart=time.time()
if ((makeFig7==True) or (makeAllFigs==True)):
	fig7 = plt.figure(8,figsize=(fwid,fhgt))
	numRows=1000
	numCols=numRows
	gs1 = gridspec.GridSpec(numRows, numCols)

	ax1 = fig7.add_subplot(gs1[:,:])
	percentile = 95.0
	lower = (100.0 - percentile)/2.0  # lower and upper intervals for credible region
	upper = 100.0 - lower
	MErr_mid = np.median(measerr_scale)
	MErr_high = np.percentile(measerr_scale,upper)
	MErr_low = np.percentile(measerr_scale,lower)
	nums, bins, patches = ax1.hist(measerr_scale, nBinsFig7, normed=0, edgecolor='#666666',facecolor='#b3b3b3')

	ax1.axvline(MErr_mid,0,1,linestyle='-.',linewidth=2, color='#d95f02')
	ax1.axvline(MErr_high,0,1,linestyle='--',linewidth=2, color='#d95f02')
	ax1.axvline(MErr_low,0,1,linestyle='--',linewidth=2, color='#d95f02')

	ax1.set_xlabel(r'{${\scriptstyle M_{\mathrm{Err}}}$}')
	ax1.set_xlim(1.26,1.34)
	numXTicks=3
	xTicks=np.zeros(numXTicks)
	xTicks[0]=MErr_low
	xTicks[1]=MErr_mid
	xTicks[2]=MErr_high
	ax1.set_xticks(xTicks)
	xLabels = [item.get_text() for item in ax1.get_xticklabels()]
	for i in xrange(numXTicks):
		xLabels[i]=r'${\scriptscriptstyle %3.2f}$'%(xTicks[i])
	ax1.set_xticklabels(xLabels)

	ax1.set_ylabel(r'Counts',fontsize=footnoteFontSize)
	ax1.set_ylim(0.0,4000.0)
	numYTicks=5
	yTicks=np.zeros(numYTicks)
	for i in xrange(numYTicks):
		yTicks[i]=1000*i
	ax1.set_yticks(yTicks)
	yLabels = [item.get_text() for item in ax1.get_yticklabels()]
	for i in xrange(numYTicks):
		yLabels[i]=r'${\scriptscriptstyle %d}$'%(yTicks[i])
	ax1.set_yticklabels(yLabels)

	plt.tight_layout()
	figure5JPG=outPath+'Zw229-15_MErr.jpg'
	plt.savefig(figure5JPG,dpi=300)
	if (makePDF==True):
		figure5PDF=outPath+'Zw229-15_MErr.pdf'
		plt.savefig(figure5PDF,dpi=300)
	if (makeEPS==True):
		figure5EPS=outPath+'Zw229-15_MErr.eps'
		plt.savefig(figure5EPS,dpi=300)
timeFig7PFinish=time.time()
print "Fig7 Plots: %f (sec) i.e. %f (min)"%((timeFig7PFinish-timeFig7PStart),(timeFig7PFinish-timeFig7PStart)/60.0)

############################################################ End Figure 7 ######################################################################

############################################################## Figure 8 #######################################################################

timeFig8PStart=time.time()
if ((makeFig8==True) or (makeAllFigs==True)):
	fig8 = plt.figure(9,figsize=(fwid,fhgt))
	numRows=1000
	numCols=numRows
	gs1 = gridspec.GridSpec(numRows, numCols)

	ax1 = fig8.add_subplot(gs1[:,:])
	ax1.errorbar(0,0,yerr=0,fmt='.',capsize=0,color='#d95f02',markeredgecolor='none',label=r'observed Zw 229-15 light curve')
	ax1.errorbar(t,y,yerr=yerr,fmt='.',capsize=0,color='#d95f02',markeredgecolor='none',zorder=10)

	ax1.errorbar(0,0,yerr=0,fmt='.',capsize=0,color='#666666',markeredgecolor='none',label=r'simulated Zw 229-15 light curve')
	ax1.errorbar(t,y_sim,yerr=yerr_sim,fmt='.',capsize=0,color='#666666',markeredgecolor='none',zorder=10)

	ax1.set_xlim(sample.time.min(),sample.time.max())
	ax1.set_ylim(np.min(y)*0.9,np.max(y)*1.1)
	ax1.set_xlabel(r'$t$~{\large (MJD)}')
	ax1.set_ylabel(r'$F$~{\large (arb. units)}')

	handles1,labels1=ax1.get_legend_handles_labels()
	LCVarPatch=mpatches.Patch(color='#b3b3b3',label=r'$\langle [\mathrm{Zw 229-15 LC}]^2 \rangle$')
	handles1.append(LCVarPatch)
	labels1.append(r'std dev of est. Zw 229-15 light curve')
	newOrder1=[1,0,2]
	handles1=[handles1[i] for i in newOrder1]
	labels1=[labels1[i] for i in newOrder1]
	if (legendFig1LC==True):
		ax1.legend(handles1,labels1,loc=4,ncol=1,fancybox=True,fontsize=scriptFontSize)

	plt.tight_layout()
	figure8JPG=outPath+'Zw229-15_mockLC.jpg'
	plt.savefig(figure8JPG,dpi=300)
	if (makePDF==True):
		figure8PDF=outPath+'Zw229-15_mockLC.pdf'
		plt.savefig(figure8PDF,dpi=300)
	if (makeEPS==True):
		figure8EPS=outPath+'Zw229-15_mockLC.eps'
		plt.savefig(figure8EPS,dpi=300)
timeFig8PFinish=time.time()
print "Fig8 Plots: %f (sec) i.e. %f (min)"%((timeFig8PFinish-timeFig8PStart),(timeFig8PFinish-timeFig8PStart)/60.0)
############################################################ End Figure 8 ######################################################################

timeFig9PStart=time.time()
if ((makeFig9==True) or (makeAllFigs==True)):

	fig9 = plt.figure(10,figsize=(fwid,fhgt))
	numRows=1000
	numCols=numRows
	gs1 = gridspec.GridSpec(numRows, numCols)

	ax1 = fig3.add_subplot(gs1[:,:])
	scatPlot = ax1.scatter(tscale,zeta,c=best_loglike-loglike,marker='.',cmap=colormap.gist_rainbow_r,linewidth = 0)

	ax1.set_xlim(54.0,66.0)
	numXTicks=5
	xTicks=np.zeros(numXTicks)
	for i in xrange(numXTicks):
		xTicks[i]=2.0*i+56.0
	ax1.set_xticks(xTicks)
	xLabels = [item.get_text() for item in ax1.get_xticklabels()]
	for i in xrange(numXTicks):
		xLabels[i]=r'${\scriptscriptstyle %3.2f}$'%(xTicks[i])
	ax1.set_xticklabels(xLabels)
	ax1.set_xlabel(r'$T_{\mathrm{dHO}}$ {\large (d)}')

	ax1.set_ylim(2.0,4.5)
	numYTicks=3
	yTicks=np.zeros(numYTicks)
	for i in xrange(numYTicks):
		yTicks[i]=i+2.25
	ax1.set_yticks(yTicks)
	yLabels = [item.get_text() for item in ax1.get_yticklabels()]
	for i in xrange(numYTicks):
		yLabels[i]=r'${\scriptscriptstyle %3.2f}$'%(yTicks[i])
	ax1.set_yticklabels(yLabels)
	ax1.set_ylabel(r'$\zeta$')

	numCBarTicks=5
	cBarTicks=np.zeros(numCBarTicks)
	for i in xrange(numCBarTicks):
		cBarTicks[i]=-3.0*i-1.5
	cBar = plt.colorbar(scatPlot, ax=ax1, orientation='vertical',ticks=cBarTicks,format=r'$\scriptstyle %2.1f$')
	cBar.set_label(r'Relative Likelihood',fontsize=footnoteFontSize)

	ax1.annotate(r'$T_{\mathrm{dHO}} = %3.1f^{+%2.1f}_{-%2.1f}$ (d)'%(tscale_mid,tscale_high-tscale_mid,tscale_mid-tscale_low),xy=(0.7125,0.25),xycoords='axes fraction',textcoords='axes fraction',ha='center',va='center',multialignment='center',fontsize=24)
	ax1.annotate(r'$\zeta_{\mathrm{dHO}} = %3.2f^{+%3.2f}_{-%3.2f}$'%(zeta_mid,zeta_high-zeta_mid,zeta_mid-zeta_low),xy=(0.7,0.15),xycoords='axes fraction',textcoords='axes fraction',ha='center',va='center',multialignment='center',fontsize=24)


	plt.tight_layout()
	figure3JPG=outPath+'Zw229-15_DampedHO.jpg'
	plt.savefig(figure3JPG,dpi=300)
	if (makePDF==True):
		figure3PDF=outPath+'Zw229-15_DampedHO.pdf'
		plt.savefig(figure3PDF,dpi=300)
	if (makeEPS==True):
		figure3EPS=outPath+'Zw229-15_DampedHO.eps'
		plt.savefig(figure3EPS,dpi=300)
timeFig9PFinish=time.time()
print "Fig9 Plots: %f (sec) i.e. %f (min)"%((timeFig9PFinish-timeFig9PStart),(timeFig9PFinish-timeFig9PStart)/60.0)

############################################################ End Figure 3 ######################################################################
