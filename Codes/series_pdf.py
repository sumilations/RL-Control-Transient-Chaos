# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import scipy.stats



snn_f =  np.loadtxt("/home/user/Desktop/sunspots/codes/Solar-Cycle/Model/)sunspot_HAVOK_eigen-corordinates.csv", delimiter =",")[:,2]#np.loadtxt("/home/user/Desktop/lorenz_transient_terminal__test_full5.csv", delimiter = ",")

snn = (snn_f)#(np.random.randn(100000))#abs(np.sin(2.*np.pi*l))
#snn = (snn.astype(np.float) - np.min(snn))/(np.max(snn) - np.min(snn))

snn_len = np.linspace(1, len(snn)+1, len(snn))
cut_months = 1

len_2   = int((len(snn)/(float(cut_months))))
snn_cut = np.zeros(int(len_2))
snn_cut_len = np.zeros(int(len_2))


for i in range(0, int(len_2)):
	snn_cut[i] = snn[i*cut_months]
	snn_cut_len[i] = snn_len[i*cut_months]
	

threshold = np.mean(snn)
snn_cross = 5*np.ones(len(snn_cut))
			
a = snn_cut#/ float(len(snn_cross))
#a=np.log(a)
x = np.max(snn_cut)
print(x,a)
bins = np.linspace(np.min(snn_cut), np.max(snn_cut), 100)
	
print(snn_cross)
hist = np.histogram(a, bins=bins, density = True)	
hist_dist = scipy.stats.rv_histogram(hist)	
#dx = l[1]-l[0]
#cdf = np.cumsum(hist)*dx
ax = plt.subplot(111)
plt.hist(a, bins=bins)


plt.show()

"""
#plt.scatter(np.log(a), np.log(hist_dist.pdf(a)), label='threshold_crossing')

ax.plot((a), (hist_dist.pdf(a)), 'o', c='red', alpha=0.05, markeredgecolor='none')
#ax.set_yscale('log')
#ax.plot(l,m, linewidth=3,label='$\mathrm{0.58e^{-x/82}}$',color='b')
#ax.plot(l,n, linewidth=3,label='$\mathrm{0.15e^{-x/112}}$',color='k')
#ax.set_xscale('log')
plt.ylabel("pdf")
plt.xlabel("x")
#plt.legend(loc='upper right')
#plt.ylim(0.0001,1.3)
#plt.xlim(10, 600)
#plt.setp(ax1.get_xticklabels(), fontsize=6)
#plt.legend()
x=scipy.arange(len(a))
dist_names = [ 'pareto']

'''
for dist_name in dist_names:
	dist = getattr(
	scipy.stats, dist_name)
	param = dist.fit(a)
	pdf_fitted = np.log(dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]))
	plt.plot(pdf_fitted, label=dist_name)
	plt.title(' cutoff = 1 month')
	plt.xlabel('days')
	plt.xlim(0.45, 100)
plt.legend(loc='upper right')	

print(param)
'''
#ax2 = plt.subplot(212)
#plt.plot(snn_cut_len, snn_cut, label='sunspot numbers (normalized)')
#plt.plot(snn_cut_len, snn_cross, label='sunspot numbers (normalized)')

plt.savefig("pdf_rho_pert.pdf")
plt.show()

"""
