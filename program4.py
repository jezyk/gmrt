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
from matplotlib.colors import LogNorm
import matplotlib.font_manager as font_manager


def nintd(x):
    nintd = int(math.floor(x))
    return nintd

I = np.array(np.fromfile('/home/jezyk/Pulpit/data/testi.dat', dtype=np.float32, count=-1, sep=''))
Q = np.array(np.fromfile('/home/jezyk/Pulpit/data/testq.dat', dtype=np.float32, count=-1, sep=''))
U = np.array(np.fromfile('/home/jezyk/Pulpit/data/testu.dat', dtype=np.float32, count=-1, sep=''))
V = np.array(np.fromfile('/home/jezyk/Pulpit/data/testv.dat', dtype=np.float32, count=-1, sep=''))
#I = np.array(np.fromfile('/home/jezyk/Pulpit/data/i.dat', dtype=np.float32, count=-1, sep=''))
#Q = np.fromfile('/home/jezyk/Pulpit/data/q.dat', dtype=np.float32, count=-1, sep='')
#U = np.fromfile('/home/jezyk/Pulpit/data/u.dat', dtype=np.float32, count=-1, sep='')
#V = np.fromfile('/home/jezyk/Pulpit/data/v.dat', dtype=np.float32, count=-1, sep='')

I = I[:len(I)/20]
Q = Q[:len(I)]
U = U[:len(I)]
V = V[:len(I)]

print len(I), len(Q), len(U), len(V)


#period =0.2530651649482
#period = 0.2530776649482
period = 0.253078395

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
        I[el] = (I[el] + I[el+1]) /2.
        Q[el] = (Q[el] + Q[el+1]) /2.
        U[el] = (U[el] + U[el+1]) /2.
        V[el] = (V[el] + V[el+1]) /2.
        s = 0
    s = s + delbin
print c,el

#print rem


Irem = np.delete(I,rem)
Qrem = np.delete(Q,rem)
Urem = np.delete(U,rem)
Vrem = np.delete(V,rem)
#print len(I), len(Irem), len(I)-len(Irem)
#print type(Irem), type(Qrem),type(Urem),type(Vrem)

#for i in xrange(5):
#    print Irem[i], Qrem[i], Urem[i], Vrem[i]


Irem = Irem[:int(nper)*int(nb)]
Qrem = Qrem[:int(nper)*int(nb)]
Urem = Urem[:int(nper)*int(nb)]
Vrem = Vrem[:int(nper)*int(nb)]
print len(Irem), len(Qrem), len(Urem), len(Vrem)


i2d = np.reshape(Irem, (len(Irem)/int(nb),int(nb)))
q2d = np.reshape(Qrem, (len(Qrem)/int(nb),int(nb)))
u2d = np.reshape(Urem, (len(Urem)/int(nb),int(nb)))
v2d = np.reshape(Vrem, (len(Vrem)/int(nb),int(nb)))
#i2dlist = i2d.tolist()
print 'new shape dimension',i2d.shape,q2d.shape, u2d.shape, v2d.shape   # type(i2d)#, type(i2dlist)

#print i2d,q2d,u2d,v2d


#su = np.apply_along_axis(np.mean, 0, i2d)
#sut= np.apply_along_axis(np.mean, 0, i2dtemptolist)


step_sdv = 100

sdi0 = np.apply_along_axis(running_avr, 1, i2d, step_sdv, 0)
sdi1 = np.apply_along_axis(running_avr, 1, i2d, step_sdv, 1)
sdq0 = np.apply_along_axis(running_avr, 1, q2d, step_sdv, 0) #avr of Stokes Q in one row
sdq1 = np.apply_along_axis(running_avr, 1, q2d, step_sdv, 1) #stdev of Stokes Q in one row
sdu0 = np.apply_along_axis(running_avr, 1, u2d, step_sdv, 0) #avr of Stokes U in one row
sdu1 = np.apply_along_axis(running_avr, 1, u2d, step_sdv, 1) #stdev of Stokes U in one row
sdv0 = np.apply_along_axis(running_avr, 1, v2d, step_sdv, 0) #avr of Stokes V in one row
sdv1 = np.apply_along_axis(running_avr, 1, v2d, step_sdv, 1) #stdev of Stokes V in one row

#print sdi0, type(sdi0), sdi0.shape
#print sdi1, type(sdi1), sdi1.shape


min_sdv = np.apply_along_axis(min, 1, sdi1)
min_pos = np.apply_along_axis(np.argmin, 1, sdi1)

min_sdv_q = np.apply_along_axis(min, 1, sdq1)
min_pos_q = np.apply_along_axis(np.argmin, 1, sdq1)

min_sdv_u = np.apply_along_axis(min, 1, sdu1)
min_pos_u = np.apply_along_axis(np.argmin, 1, sdu1)

min_sdv_v = np.apply_along_axis(min, 1, sdv1)
min_pos_v = np.apply_along_axis(np.argmin, 1, sdv1)



min_val_i = []
for i, arr in zip(min_pos, sdi0):
     min_val_i.append(arr[i])

min_val_q = []
for i, arr in zip(min_pos_q, sdq0):
     min_val_q.append(arr[i])

min_val_u = []
for i, arr in zip(min_pos_u, sdu0):
     min_val_u.append(arr[i])

min_val_v = []
for i, arr in zip(min_pos_v, sdv0):
     min_val_v.append(arr[i])



#print min_val_i


i2d_b = (i2d.T - min_val_i).T  # i2d minus baseline in each SP separately
q2d_b = (q2d.T - min_val_q).T  # q2d minus baseline
u2d_b = (u2d.T - min_val_u).T  # u2d minus baseline
v2d_b = (v2d.T - min_val_v).T  # v2d minus baseline
#print i2d_bT.shape

#print i2d_b.shape,i2d_b
sdi2 = np.apply_along_axis(np.mean, 0, i2d_b) # mean Stokes
sdq2 = np.apply_along_axis(np.mean, 0, q2d_b)
sdu2 = np.apply_along_axis(np.mean, 0, u2d_b)
sdv2 = np.apply_along_axis(np.mean, 0, v2d_b)

aiew = running_avr(sdi2, step_sdv, 0)
siew = running_avr(sdi2, step_sdv, 1)
min_pos_siew = np.argmin(siew)

SIEW = siew[min_pos_siew]
print min_pos_siew, SIEW
print sdi2.shape

#print L, L_b





# total polarisation intensity
#tot_pol_int = np.sqrt(np.array(q2d**2 + u2d**2 + v2d**2))

#print tot_pol_int.shape

# total polarisation fraction
#tot_pol_frac = tot_pol_int / i2d

#print tot_pol_int, tot_pol_frac



# linear polarisation
L = np.sqrt(q2d**2 + u2d**2)
#L_b = np.sqrt(q2d_b**2 + u2d_b**2)
L_b = np.sqrt(sdq2**2 + sdu2**2)





#print tot_pol_frac.shape

meanI = np.apply_along_axis(np.mean, 0, i2d)
#meanI_b = np.apply_along_axis(np.mean, 0, sdi2)

meanQ = np.apply_along_axis(np.mean, 0, q2d)
meanQ_b = np.apply_along_axis(np.mean, 0, sdq2)

meanU = np.apply_along_axis(np.mean, 0, u2d)
meanU_b = np.apply_along_axis(np.mean, 0, sdu2)

meanV = np.apply_along_axis(np.mean, 0, v2d)
#meanV_b = np.apply_along_axis(np.mean, 0, sdv2)

#meanL = np.apply_along_axis(np.mean, 0, L)
meanL_b = np.apply_along_axis(np.mean, 0, L_b)

LQU = np.sqrt(meanQ_b**2 + meanU_b**2)

#sdL0 = running_avr(meanL, step_sdv, 0)
#sdL1 = running_avr(meanL, step_sdv, 1)

#min_sdv_L = np.min(sdL1)
#min_pos_L = np.argmin(sdL1)

#min_val_L = []
#for i, arr in zip(min_pos_L, meanL):
#     min_val_L.append(arr[i])
L#2d_b = meanL - sdL0[min_pos_L]

#meanL2d_b = np.apply_along_axis(np.mean, 0, L2d_b)
#print min_sdv_L, min_pos_L, sdL0[min_pos_L], L2d_b.shape

#print L_b.shape, L_b

#print len(L_b)

# ==========================================
# polarisation L Everett, Weisberg & Mitra, Li
LEW = np.zeros(len(L_b))  
LEW[L_b/SIEW >= 1.57] = np.sqrt((L_b[L_b/SIEW >= 1.57])**2-SIEW**2)

#for i in xrange(len(LEW)):
#    if (L_b[i]/SIEW >= 1.57):
#        LEW[i] = np.sqrt((L_b[i])**2 - SIEW**2)
#        LEW[i] = L_b[i]
#print LEW

LML = L_b - SIEW
# ==========================================

#mean_tot_pol_frac = np.apply_along_axis(np.mean, 0, tot_pol_frac)
#maxL= np.apply_along_axis(max, 1, L)
#maxLpos = np.apply_along_axis(np.argmin, 1, L)
#print maxL, maxLpos

#for i in xrange(u2d.shape[0]):
#    uppa = 0.0
#    qppa = 0.0

#    uppa = np.apply_along_axis(sum, 0, u2d)
#    qppa = np.apply_along_axis(sum, 0, q2d) 

#    ppa = np.array (0.5 * np.arctan2(uppa,qppa))#**(-1.)
#    print ppa
    
#print np.degrees(ppa), ppa.shape

#exit()

#i2d_b_l = np.log10(i2d_b)
#i2d_b_l[np.isnan(i2d_b_l)]=1e-10







plt.subplots_adjust(hspace=0.1)
#plt.subplot(211)
##plt.xlim([700,900])
#plt.xlim([0,nb])
#plt.ylim([0,nper])
#imgplot = plt.imshow(i2d_b, cmap=plt.get_cmap('afmhot'), aspect='auto')
#x=[800.,800.]
#y=[0.,nper]
#plt.plot(x,y)

plt.subplot(212)
#plt.xlim([700,900])
plt.xlim([0,nb])
plt.plot(sdi2, label='meanI_b')
#plt.plot(meanI, label='meanI')
#plt.plot(sdi2, label='meanI_b')
#plt.plot(meanQ, label='meanQ')
#plt.plot(sdq2, label='meanQ_b')
#plt.plot(meanU, label='meanU')
#plt.plot(sdu2, label='meanU_b')
#plt.plot(meanV, label='meanV')
plt.plot(sdv2, label='meanV_b')
#plt.plot(meanL, label='meanL')
#plt.plot(meanL_b, label='meanL_b')
#plt.plot(LEW, label='LEW', lw=0.5)
#plt.plot(LML, label='LML', lw=0.5)
plt.plot(LQU, label='LQU', lw=0.5)
#plt.plot(L2d_b, label='L-b', lw=0.5)

#plt.plot(meanV**2, label='meanV2')
#plt.plot(meanU**2, label='meanU2')
#plt.plot(mean_tot_pol_frac)
plt.legend(loc='upper left',prop=font_manager.FontProperties(size=10))

#plt.plot(L.T)
plt.savefig('stokes.ps', bbox_inches='tight')
#plt.show()


plt.subplots_adjust(hspace=0.1)
plt.subplot(211)
plt.plot(np.degrees(ppa),  label='ppa', color='red')

#plt.savefig('ppa.ps', bbox_inches='tight')
#plt.show()











