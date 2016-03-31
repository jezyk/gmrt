#!/usr/bin/env python

#print plt.imshow.__doc__

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import scipy
import pylab
from scipy import signal
import collections
from functions import running_avr
from matplotlib.colors import LogNorm
import matplotlib.font_manager as font_manager

rms_L_SP = 3.  # === N sigma level in every SP
rms_L = 3.  # === N sigma in average L


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

I = I[:len(I)]
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

print 'BASIC CALCULATIONS ---> DONE'

Irem = np.delete(I,rem)
Qrem = np.delete(Q,rem)
Urem = np.delete(U,rem)
Vrem = np.delete(V,rem)


Irem = Irem[:int(nper)*int(nb)]
Qrem = Qrem[:int(nper)*int(nb)]
Urem = Urem[:int(nper)*int(nb)]
Vrem = Vrem[:int(nper)*int(nb)]
print len(Irem), len(Qrem), len(Urem), len(Vrem)


i2d = np.reshape(Irem, (len(Irem)/int(nb),int(nb)))
q2d = np.reshape(Qrem, (len(Qrem)/int(nb),int(nb)))
u2d = np.reshape(Urem, (len(Urem)/int(nb),int(nb)))
v2d = np.reshape(Vrem, (len(Vrem)/int(nb),int(nb)))


print 'REBINNING AND RESHAPING ---> DONE'

#i2dlist = i2d.tolist()
print 'new shape dimension',i2d.shape,q2d.shape, u2d.shape, v2d.shape   # type(i2d)#, type(i2dlist)


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

print 'AVERAGE AND STDDEV IN STOKES ---> DONE'


min_sdv = np.apply_along_axis(min, 1, sdi1)
min_pos = np.apply_along_axis(np.argmin, 1, sdi1)

min_sdv_q = np.apply_along_axis(min, 1, sdq1)
min_pos_q = np.apply_along_axis(np.argmin, 1, sdq1)

min_sdv_u = np.apply_along_axis(min, 1, sdu1)
min_pos_u = np.apply_along_axis(np.argmin, 1, sdu1)

min_sdv_v = np.apply_along_axis(min, 1, sdv1)
min_pos_v = np.apply_along_axis(np.argmin, 1, sdv1)


print 'AVERAGE STOKES ---> DONE'

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


i2d_b = (i2d.T - min_val_i).T  # i2d minus baseline in each SP separately
q2d_b = (q2d.T - min_val_q).T  # q2d minus baseline
u2d_b = (u2d.T - min_val_u).T  # u2d minus baseline
v2d_b = (v2d.T - min_val_v).T  # v2d minus baseline
#print i2d_bT.shape


print 'REBINNED AND RESHAPED - BASELINE ---> DONE'


#print i2d_b.shape,i2d_b
sdi2 = np.apply_along_axis(np.mean, 0, i2d_b) # mean Stokes === average profile
sdq2 = np.apply_along_axis(np.mean, 0, q2d_b)
sdu2 = np.apply_along_axis(np.mean, 0, u2d_b)
sdv2 = np.apply_along_axis(np.mean, 0, v2d_b)

aiew = running_avr(sdi2, step_sdv, 0)
siew = running_avr(sdi2, step_sdv, 1)
min_pos_siew = np.argmin(siew)

SIEW = siew[min_pos_siew]
print min_pos_siew, SIEW
print sdi2.shape



# linear polarisation
L = np.sqrt(q2d**2 + u2d**2)
L_b2 = np.sqrt(q2d_b**2 + u2d_b**2)
L_b = np.sqrt(sdq2**2 + sdu2**2)
L_b2alt = np.sqrt((np.apply_along_axis(sum, 0, q2d_b))**2 + (np.apply_along_axis(sum, 0, u2d_b))**2)/int(nper)

print 'L_b.shape',L_b.shape, L_b2.shape, L_b2alt.shape
sdL_b20 = np.apply_along_axis(running_avr, 1, L_b2, step_sdv, 0) # srednia biegnaca w wierszu
sdL_b21 = np.apply_along_axis(running_avr, 1, L_b2, step_sdv, 1) # odchylenie standardowe dla sredniej biegnacej w wierszu

min_sdv_L_b2 = np.apply_along_axis(min, 1, sdL_b21)  # minimalna wartosc odchylenia dla sr. bieg. w wierszu
min_pos_L_b2 = np.apply_along_axis(np.argmin, 1, sdL_b21) # pozycja 

min_val_L_b2 = []
for i, arr in zip(min_pos_L_b2, sdL_b20):
     min_val_L_b2.append(arr[i])

#print min_val_L_b2



#a = np.array([[3, 4, 5, 6, 7], [11, 12, 13, 14, 15]])  #L_b3
#b = np.array([5, 13]) # rms_L_SP * min_sdv_L_b2  ## 3(N) sigma dla danego wiersza
#c = (a.T < b).T
#d = np.zeros(a.shape)
#d[c == True] = a[c == True]
#s = csr_matrix(d)
#e = np.ma.array(a, mask=c)
#e.mean(axis=0)
#e.mean(axis=1)




# ==========================================
# polarisation L Everett, Weisberg & Mitra, Li
LEW = np.zeros(len(L_b))  
LEW[L_b/SIEW >= 1.57] = np.sqrt((L_b[L_b/SIEW >= 1.57])**2-SIEW**2)
LEW = np.array(LEW)
#for i in xrange(len(LEW)):
#    if (L_b[i]/SIEW >= 1.57):
#        LEW[i] = np.sqrt((L_b[i])**2 - SIEW**2)
#        LEW[i] = L_b[i]
#print LEW

LML = L_b - SIEW

LEW0 =  running_avr(LEW, step_sdv, 0)
LEW1 =  running_avr(LEW, step_sdv, 1)
min_sdv_LEW = min(LEW1)
min_pos_LEW = np.argmin(LEW1)
print 'min_sdv_LEW, min_pos_LEW', min_sdv_LEW, min_pos_LEW
# ==========================================

print 'CIRCULAR POLARISATION ---> DONE'

#mean_tot_pol_frac = np.apply_along_axis(np.mean, 0, tot_pol_frac)
#maxL= np.apply_along_axis(max, 1, L)
#maxLpos = np.apply_along_axis(np.argmin, 1, L)
#print maxL, maxLpos

#for i in xrange(u2d.shape[0]):
#    uppa = 0.0
#    qppa = 0.0

#    uppa = np.apply_along_axis(sum, 0, u2d_b)
#    qppa = np.apply_along_axis(sum, 0, q2d_b) 


ppa_all = np.array (0.5 * np.arctan2(u2d_b,q2d_b))
#ppa_all = np.array (0.5 * np.arctan(u2d_b/q2d_b))
#print ppa_all.shape
#    print ppa

L_b3 = (L_b2.T - min_val_L_b2).T # L_b2 z odjetym baselinem dla kazdego SP indywidualnie
Nsigma_SP_L = rms_L_SP * min_sdv_L_b2  # N * rms (L) === w kazdym SP osobno
mask = (L_b3.T < Nsigma_SP_L).T # tablica z boolami gdzie TRUE jest dla wartosci L w danym binie wiekszej od Nsigma

ppa2 = np.zeros(L_b3.shape)
#print ppa2.shape, ppa2, ppa_all, Nsigma_SP_L, L_b3

#print type(L_b3), type(Nsigma_SP_L), type(mask), type(out1)
ppa2[mask == False] = ppa_all[mask == False]

ppa2 = np.ma.array(ppa2, mask=mask)
#print mask, ppa2

ppa = scipy.sparse.csr_matrix(ppa2)

f1 = scipy.sparse.find(ppa)[0]
f2 = scipy.sparse.find(ppa)[1]
f3 = scipy.sparse.find(ppa)[2]

PPA_ALL = 0.5 * np.arctan2(np.apply_along_axis(np.mean, 0, u2d_b), np.apply_along_axis(np.mean, 0, q2d_b))
Nsigma_L = rms_L * np.array(LEW1)
mask_L = (LEW[:len(LEW1)].T < Nsigma_L).T # stddev L ma dlugosc 930 (NBIN - step_sdv)
PPA2 = np.zeros(LEW[:len(LEW1)].shape)
PPA2[mask_L == False] = PPA_ALL[:len(LEW1)][mask_L == False]
PPA2 = np.ma.array(PPA2, mask=mask_L)
#print PPA2

PPA = scipy.sparse.csr_matrix(PPA2)



F1 = scipy.sparse.find(PPA)[0]
F2 = scipy.sparse.find(PPA)[1]
F3 = scipy.sparse.find(PPA)[2]

#print F1, F2, np.degrees(F3)

#print mask_L
#



print 'PPA ---> DONE'





plt.subplots_adjust(hspace=0.1)
#plt.subplot(311)
##plt.xlim([700,900])
#plt.xlim([0,nb])
#plt.ylim([0,nper])
#imgplot = plt.imshow(i2d_b, cmap=plt.get_cmap('afmhot'), aspect='auto')
#x=[800.,800.]
#y=[0.,nper]
#plt.plot(x,y)

plt.subplot(312)
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
#plt.plot(meanL_b2, label='meanL_b2')
#plt.plot(meanL_b, label='meanL_b')
#plt.plot(LEW, label=r'${\mathrm{L_{EW}}}$', lw=0.5)
plt.plot(LEW, label=r'${L_{EW}}$', lw=0.5)
#plt.plot(LML, label='LML', lw=0.5)
#plt.plot(LQU, label='LQU', lw=0.5)
#plt.plot(L2d_b, label='L-b', lw=0.5)
plt.plot(L_b2alt, label='L_b2alt', lw=0.5)

#plt.plot(meanV**2, label='meanV2')
#plt.plot(meanU**2, label='meanU2')
#plt.plot(mean_tot_pol_frac)
plt.legend(loc='upper left',prop=font_manager.FontProperties(size=10))

#plt.plot(L.T)
#plt.savefig('stokes.ps', bbox_inches='tight')
#plt.show()


plt.subplots_adjust(hspace=0.7)
plt.subplot(313)
plt.xlim([0,nb])
#plt.plot(np.degrees(ppa),  label='PPA - arctan', color='r', marker='.', linestyle='none')
#plt.plot(np.degrees(ppa1), label='PPA - arctan2', color='b', marker='.', linestyle='none')
#plt.plot(np.degrees(e), label='PPA - arctan2', color='b', marker='.', linestyle='none')
#print 'ppa', ppa
#plt.spy(ppa, aspect="auto", markersize=.1)#, label='PPA - arctan2', color='b', marker='.', linestyle='none')

plt.plot(f2, np.degrees(f3), label='PPA - arctan2', color='black', marker='.', markersize=0.1, linestyle='none')
plt.plot(F2, np.degrees(F3), label='PPA - arctan2', color='red', marker='.', markersize=0.5, linestyle='none')

plt.xlabel('BIN')
plt.ylabel(r'PPA [$^{\circ}$]')
plt.title('Polarisation Position Angle')
#plt.legend(loc='upper left',prop=font_manager.FontProperties(size=8))

#for i in xrange (u2d_b.shape[0]):
#for i in xrange(1):
#    plt.plot (np.degrees(ppa[i,]), marker='.', markersize=0.1, linestyle='none', color='b')
#    print i


#plt.savefig('ppa.ps', bbox_inches='tight')
#plt.show()


plt.savefig('stokes.ps', bbox_inches='tight')









