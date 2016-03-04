#!/usr/bin/env python

#print plt.imshow.__doc__

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import scipy
from scipy import signal
import collections
from functions import running_avr



def nintd(x):
    nintd = int(math.floor(x))
    return nintd

I = np.array(np.fromfile('/home/jezyk/Pulpit/data/testi.dat', dtype=np.float32, count=-1, sep=''))
#I = np.array(np.fromfile('/home/jezyk/Pulpit/data/i.dat', dtype=np.float32, count=-1, sep=''))
#Q = np.fromfile('/home/jezyk/Pulpit/data/q.dat', dtype=np.float32, count=-1, sep='')
#U = np.fromfile('/home/jezyk/Pulpit/data/u.dat', dtype=np.float32, count=-1, sep='')
#V = np.fromfile('/home/jezyk/Pulpit/data/v.dat', dtype=np.float32, count=-1, sep='')

I = I[:len(I)/200]

#period =0.2530651649482
period = 0.2530776649482
t_res = 0.00024576

print 'pulsar period', period
print 'original sample length',len(I)

tlen = len(I) * t_res
print 'full time series length', tlen

nb = period / t_res
print 'number of bins per period', nb, 'integer number of bins per period', int(nb)

nper = tlen / period
print 'total number of periods', nper, 'integer number of periods',int(nper)


print 'number of bins in iteger number of periods', int(nper)*nb
print 'number of integer bins in iteger number of periods',int(nper)*int(nb)


print 'diffrence in bin number',int(nper)*nb-int(nper)*int(nb)

excc = nb-int(nb)
delbin = excc/int(nb)
print 'bin excces in one period / one bin', excc, delbin

bin1 = int(nb)/excc
print 'one/half extra bin after no of bins',bin1,bin1/2.


s = 0. 
c = 0
rem =[]
for el in xrange(len(I)):
    if s >= 1:
        c = c + 1
#        print c,el
        rem.append(el)
        s = 0
    s = s + delbin
print c,el


#Itemp = I[:int(nper)*int(nb)]
#i2dtemp = np.reshape(Itemp, ( len(Itemp)/int(nb),int(nb)))
#i2dtemptolist  = i2dtemp.tolist()

Irem = np.delete(I,rem)
print len(I), len(Irem), len(I)-len(Irem)
Irem = Irem[:int(nper)*int(nb)]
print len(Irem)
i2d = np.reshape(Irem, ( len(Irem)/int(nb),int(nb)))
#i2dlist = i2d.tolist()
print 'new shape dimension',i2d.shape, type(i2d)#, type(i2dlist)

su = np.apply_along_axis(np.mean, 0, i2d)
#sut= np.apply_along_axis(np.mean, 0, i2dtemptolist)


step_sdv = 100

su0 = np.apply_along_axis(running_avr, 1, i2d, step_sdv, 0)
su1 = np.apply_along_axis(running_avr, 1, i2d, step_sdv, 1)
print su0, type(su0), su0.shape
print su1, type(su1), su1.shape


min_sdv = np.apply_along_axis(min, 1, su1)

min_pos = np.apply_along_axis(np.argmin, 1, su1)
print min_sdv, min_pos, type(min_sdv), type(min_pos)

print min_pos.shape, su0.shape, su0.T.shape

np.choose(min_pos, su0.T)

#print su0.shape, su1.shape, min_sdv.shape#, min_pos.shape#, min_val.shape
#print min_val
#print min_sdv, min_pos, min_sdv.shape




#print min_pos, min_pos.shape
#avr [pos]

#i2d_bT = i2d.T - min_avr# i2d minus baseline
#print i2d_bT.shape
#i2d_b = i2d_bT.T

#print i2d_b.shape
exit()


plt.subplots_adjust(hspace=0.1)
plt.subplot(211)
#plt.xlim([700,900])
plt.xlim([0,nb])
plt.ylim([0,nper])
imgplot = plt.imshow(i2d, cmap=plt.get_cmap('afmhot'), aspect='auto')
x=[800.,800.]
y=[0.,nper]
plt.plot(x,y)

plt.subplot(212)
#plt.xlim([700,900])
plt.xlim([0,nb])
plt.plot(su)
plt.show()
















