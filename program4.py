#!/usr/bin/env python

#print plt.imshow.__doc__

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import scipy
import pylab
import sys
from scipy import signal
import collections
from functions import running_avr, width, razor
from matplotlib.colors import LogNorm
import matplotlib.font_manager as font_manager
import matplotlib.gridspec as gridspec
import ConfigParser


def nintd(x):
    nintd = int(math.floor(x))
    return nintd

Config = ConfigParser.ConfigParser()
Config.read("parameters")
Config.sections()
sample_percent = float(Config.get('BasicData', 'sample_percent').split(' ')[0])
step_sdv  = int(Config.get('BasicData', 'step_sdv').split(' ')[0])
rms_L_SP = float(Config.get('BasicData', 'rms_L_SP').split(' ')[0])
rms_L = float(Config.get('BasicData', 'rms_L').split(' ')[0])
flag_X_axis_deg = int(Config.get('BasicData', 'flag_X_axis_deg').split(' ')[0])
flag_rmv_weak_sp = int(Config.get('BasicData', 'flag_rmv_weak_sp').split(' ')[0])
flag_rmv_weak_sp_sig = float(Config.get('BasicData', 'flag_rmv_weak_sp').split(' ')[1])
flag_razor = (Config.get('BasicData', 'flag_razor').split(' ')[0:5])
flag_which_width =  int(Config.get('BasicData', 'flag_which_width').split(' ')[0])
flag_which_width_N = float(Config.get('BasicData', 'flag_which_width').split(' ')[1])
flag_mp = int(Config.get('BasicData', 'flag_which_width').split(' ')[2])


if (sample_percent > 100):
    sample_percent = 100
    print 'sample_percent > 100 in parameters file. Set on 100'

#print 'flag razor', flag_razor, type(flag_razor), int(flag_razor[2])


#parameter =[]
#with open('parameters','r') as parameters_file:
#    for line in  parameters_file:
#        parameter.append(line.split(' ', 1)[0])
#parameters_file.close()



I = np.array(np.fromfile('/home/jezyk/Pulpit/data/testi.dat', dtype=np.float32, count=-1, sep=''))
Q = np.array(np.fromfile('/home/jezyk/Pulpit/data/testq.dat', dtype=np.float32, count=-1, sep=''))
U = np.array(np.fromfile('/home/jezyk/Pulpit/data/testu.dat', dtype=np.float32, count=-1, sep=''))
V = np.array(np.fromfile('/home/jezyk/Pulpit/data/testv.dat', dtype=np.float32, count=-1, sep=''))
#I = np.array(np.fromfile('/home/jezyk/Pulpit/data/i.dat', dtype=np.float32, count=-1, sep=''))
#Q = np.fromfile('/home/jezyk/Pulpit/data/q.dat', dtype=np.float32, count=-1, sep='')
#U = np.fromfile('/home/jezyk/Pulpit/data/u.dat', dtype=np.float32, count=-1, sep='')
#V = np.fromfile('/home/jezyk/Pulpit/data/v.dat', dtype=np.float32, count=-1, sep='')

I = I[:len(I) / int(100./sample_percent)]
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


print 'min sdv', min_sdv, type(min_sdv), min_sdv.shape

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

print 'size i2d_b', i2d_b.size, sys.getsizeof(i2d_b)

#sdi00 = np.apply_along_axis(np.mean, 1, i2d_b)
#sdi11 = np.apply_along_axis(np.std, 1, i2d_b)
#sdi111= np.mean(sdi11)
##print sdi00, sdi11, sdi111

#sdi11_sort = zip(xrange(len(sdi11)), sdi11)
#sdi11_sort = sorted(sdi11_sort, key=lambda param: param[1])

#print sdi11_sort



#val = []
#val = np.array(val)
#ls = flag_rmv_weak_sp_sig # local sigma
##print ls, type(ls)


##print sdi11_sort[0][1], sdi11_sort[1][1], len(sdi11_sort[:])-1, tresh

#for i in xrange(len(sdi11_sort[:])):
#    if (i == 0):
#        tresh = ls * (sdi11_sort[0][1])
##        print tresh
#    if (i != 0):
#        tresh = ls * np.mean(val)

#    print i, tresh, sdi11_sort[i][1]

#    if (sdi11_sort[i][1] < tresh):
##        print i, sdi11_sort[i][1] , tresh

#        val = np.append(val, sdi11_sort[i][1])
##        print i, val, tresh
#    else:
#        print i, tresh, sdi11_sort[i][1], 'rejected'
##        break
##        print 'rejected'
##        print i, sdi11_sort[i][0], sdi11_sort[i][1], tresh
#    print

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

maxIval = np.max(sdi2)
maxIpos = np.argmax(sdi2)
SNR_I = maxIval/SIEW
print 'max I postion in average profile, min STDDEV, SNR', maxIpos, maxIval, SIEW, SNR_I



# ==== calculate width ======

w0 =  width(sdi2, 0, flag_which_width_N, step_sdv)
print 'wus_Nsig', w0
w1 =  width(sdi2, 1, flag_which_width_N, step_sdv)
print 'wus_10  ', w1
w2 =  width(sdi2, 2, flag_which_width_N, step_sdv)
print 'wus_50  ', w2

print 'CALCULATING WIDTH ---> DONE'
# ============================


if (int(flag_razor[0]) == 1):
    print 'razor on', flag_razor
    good = razor(flag_razor, w0[2:4], sdi2, i2d_b, min_sdv, flag_rmv_weak_sp_sig, flag_which_width_N) # flag_razor[], window, mean_I, SP I, array_of_min_sdv_I, sigma for weak SP, sigma for width





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
#print len(LEW)

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
#PPA_ALL = 0.5 * np.arctan(np.apply_along_axis(np.mean, 0, u2d_b)/ np.apply_along_axis(np.mean, 0, q2d_b))


Nsigma_L = rms_L * min_sdv_LEW
mask_L = (LEW.T < Nsigma_L).T 
PPA2 = np.zeros(LEW.shape)
PPA2[mask_L == False] = PPA_ALL[mask_L == False]
PPA2 = np.ma.array(PPA2, mask=mask_L)
#print PPA2
PPA = scipy.sparse.csr_matrix(PPA2)

errPPA2 = np.zeros(LEW.shape)
errPPA2[mask_L == False] = (1/2. * 180./np.pi) * SIEW / LEW[mask_L == False]
errPPA2 = np.ma.array(errPPA2, mask=mask_L)
errPPA = scipy.sparse.csc_matrix(errPPA2)
#errPPA = []
#errPPA1 =  (1/2. * 180./np.pi) * SIEW / LEW
#print errPPA.shape, LEW, errPPA



F1 = scipy.sparse.find(PPA)[0] # nr wiersza
F2 = scipy.sparse.find(PPA)[1] # pozycja w wierszu
F3 = scipy.sparse.find(PPA)[2] # wartosc

#print len(F1), len(F2), len(F3)
print 'PPA ---> DONE'


print 'int(nb)',int(nb)

if (flag_X_axis_deg == 1):
    X_axis=[]
    X_axis.append(0.0)
    print X_axis
    for i in xrange(1,int(nb)):    
        X_axis.append(i * 360./int(nb))



min_val_for_plot = np.min(np.concatenate([sdi2, sdq2, sdu2, sdv2]))
max_val_for_plot = np.abs(min_val_for_plot)
max_val_for_plot1= np.max(sdi2)
#print ' min_val_for_plot', min_val_for_plot

plt.subplots_adjust(hspace=0.1)
plt.subplot(311)
#plt.xlim([700,900])
if (flag_X_axis_deg == 0):
    plt.xlim([0,int(nb)])
    plt.ylabel('Pulse number')
    imgplot = plt.imshow(i2d_b, cmap=plt.get_cmap('afmhot'), aspect='auto')
    x=[maxIpos,maxIpos]

if (flag_X_axis_deg == 1):
    plt.xlim([0,360])
    plt.ylabel('Pulse number')
    imgplot = plt.imshow(i2d_b, cmap=plt.get_cmap('afmhot'), aspect='auto', extent=[0.,360.,0.,int(nper)])
    x=[maxIpos * 360./int(nb), maxIpos * 360./int(nb)]

plt.ylim([0,nper])
y=[0.,nper]
plt.plot(x,y)



plt.subplot(312)
if (flag_which_width == 0):
    wmp, wl, wp = w0[1:4]
if (flag_which_width == 1):
    wmp, wl, wp = w1[1:4]
if (flag_which_width == 2):
    wmp, wl, wp = w2[1:4]


y=[min_val_for_plot, max_val_for_plot]
if (flag_X_axis_deg == 0):
    plt.xlim([0,int(nb)])
    plt.ylabel('Arbitrary units')
    plt.plot(sdi2, label='meanI_b')
    plt.plot(sdv2, label='meanV_b')
    plt.plot(LEW, label=r'${L_{EW}}$', lw=0.5, ls='dashed', c='r')
    x=[wl, wl]
    plt.plot(x,y, c='m')
    x=[wp, wp]
    plt.plot(x,y, c='m')

    x = [0, int(nb)]
    y = [SIEW * flag_which_width_N, SIEW * flag_which_width_N]
    plt.plot (x,y, c='m')
#    x=[wmp, wmp]
#    y= [1. * max_val_for_plot1, 0.5 * max_val_for_plot1]
#    plt.plot(x,y, c='m')#, ls='dashed')
#    plt.arrow(W_Nsig_mp, 0.9 * max_val_for_plot, 0.0, 0.4 * max_val_for_plot,
#head_width=0.05, head_length=0.1)#,  c='m', ls='dashed')
    if (flag_mp == 1):
       plt.arrow(wmp, max_val_for_plot1, 0.0, (sdi2[wmp]-max_val_for_plot1)/2., head_width=5., head_length= 0.05 * max_val_for_plot1)#,  c='m', ls='dashed')

#    print sdi2.shape, type(sdi2)
#    print 'wmp, max', sdi2[wmp], max_val_for_plot1,(sdi2[wmp]-max_val_for_plot1)/2.

if (flag_X_axis_deg == 1):
    wl = wl * 360./int(nb)
    wp = wp * 360./int(nb)
    if (flag_mp == 1):
        plt.arrow(wmp* 360./int(nb), max_val_for_plot1, 0.0, (sdi2[int(wmp)]-max_val_for_plot1)/2., head_width=5., head_length= 0.05 * max_val_for_plot1)
    print '', wmp, sdi2[wmp], max_val_for_plot1
    wmp=wmp * 360./int(nb) 
    plt.xlim([0,360])
    plt.ylabel('Arbitrary units')
    plt.plot(X_axis, sdi2, label='meanI_b')
    plt.plot(X_axis, sdv2, label='meanV_b')
    plt.plot(X_axis, LEW, label=r'${L_{EW}}$', lw=0.5, ls='dashed', c='r')
    x=[wl, wl]
#    y=[min_val_for_plot, max_val_for_plot]
    plt.plot(x,y, c='m')
    x=[wp, wp]
    plt.plot(x,y, c='m')

    x=[wmp, wmp]

#plt.xlim([700,900])
#plt.xlim([0,nb])
#plt.plot(X_axis, sdi2, label='meanI_b')
#plt.plot(meanI, label='meanI')
#plt.plot(sdi2, label='meanI_b')
#plt.plot(meanQ, label='meanQ')
#plt.plot(sdq2, label='meanQ_b')
#plt.plot(meanU, label='meanU')
#plt.plot(sdu2, label='meanU_b')
#plt.plot(meanV, label='meanV')
#plt.plot(sdv2, label='meanV_b')
#plt.plot(meanL_b2, label='meanL_b2')
#plt.plot(meanL_b, label='meanL_b')
#plt.plot(LEW, label=r'${\mathrm{L_{EW}}}$', lw=0.5)
#plt.plot(LEW, label=r'${L_{EW}}$', lw=0.5, ls='dashed', c='r')
#plt.plot(LML, label='LML', lw=0.5)
#plt.plot(LQU, label='LQU', lw=0.5)
#plt.plot(L2d_b, label='L-b', lw=0.5)
#plt.plot(L_b2alt, label='L_b2alt', lw=0.5)

#plt.plot(meanV**2, label='meanV2')
#plt.plot(meanU**2, label='meanU2')
#plt.plot(mean_tot_pol_frac)
plt.legend(loc='upper left',prop=font_manager.FontProperties(size=10))

#plt.plot(L.T)
#plt.savefig('stokes.ps', bbox_inches='tight')
#plt.show()


plt.subplots_adjust(hspace=0.7)
plt.subplot(313)

if (flag_X_axis_deg == 0):
    plt.xlim([0,int(nb)])
    plt.plot(f2, np.degrees(f3), label='PPA - arctan2', color='black', marker='.', markersize=0.1, linestyle='none')
    plt.plot(F2, np.degrees(F3), label='PPA - arctan2', color='red', marker='.', markersize=0.5, linestyle='none')
    plt.errorbar(F2, np.degrees(F3), yerr=scipy.sparse.find(errPPA)[2], fmt='.', c='r', markersize=1.5, capsize=1., elinewidth = 0.5)
    plt.xlabel('BIN')
    plt.ylabel(r'PPA [$^{\circ}$]')
    plt.title('Polarisation Position Angle')


if (flag_X_axis_deg == 1):
    plt.xlim([0,360])
    plt.plot(f2/float(int(nb))*360., np.degrees(f3), label='PPA - arctan2', color='black', marker='.', markersize=0.1, linestyle='none')
    plt.plot(F2/float(int(nb))*360., np.degrees(F3), label='PPA - arctan2', color='red', marker='.', markersize=0.5, linestyle='none')
    plt.errorbar(F2/float(int(nb))*360., np.degrees(F3), yerr=scipy.sparse.find(errPPA)[2], fmt='.', c='r', markersize=1.5, capsize=1., elinewidth = 0.5)
    plt.xlabel('BIN')
    plt.ylabel(r'PPA [$^{\circ}$]')
    plt.title('Polarisation Position Angle')


#plt.xlim([0,nb])
#plt.plot(np.degrees(ppa),  label='PPA - arctan', color='r', marker='.', linestyle='none')
#plt.plot(np.degrees(ppa1), label='PPA - arctan2', color='b', marker='.', linestyle='none')
#plt.plot(np.degrees(e), label='PPA - arctan2', color='b', marker='.', linestyle='none')
#print 'ppa', ppa
#plt.spy(ppa, aspect="auto", markersize=.1)#, label='PPA - arctan2', color='b', marker='.', linestyle='none')

#plt.plot(f2, np.degrees(f3), label='PPA - arctan2', color='black', marker='.', markersize=0.1, linestyle='none')
#plt.plot(F2, np.degrees(F3), label='PPA - arctan2', color='red', marker='.', markersize=0.5, linestyle='none')
#plt.plot(np.degrees(PPA_ALL), label='PPA - arctan2', color='red', marker='.', markersize=0.5, linestyle='none')
#plt.errorbar(F2, np.degrees(F3), yerr=scipy.sparse.find(errPPA)[2], fmt='.', c='r', markersize=1.5, capsize=1., elinewidth = 0.5)#scipy.sparse.find(PPA)[2])
#print len(F2), len(F3), errPPA.shape
#plt.errorbar(100., 0.0, 10., c='b')

#plt.xlabel('BIN')
#plt.ylabel(r'PPA [$^{\circ}$]')
#plt.title('Polarisation Position Angle')
#plt.legend(loc='upper left',prop=font_manager.FontProperties(size=8))

#for i in xrange (u2d_b.shape[0]):
#for i in xrange(1):
#    plt.plot (np.degrees(ppa[i,]), marker='.', markersize=0.1, linestyle='none', color='b')
#    print i


#plt.savefig('ppa.ps', bbox_inches='tight')
#plt.show()


plt.savefig('stokes.ps', bbox_inches='tight', dpi=300, figsize=(7,11))



plt.clf()
gs1 = gridspec.GridSpec(3, 2)
x=[1,2,3,4]
y=[1,3,6,8]

plt.plot(x,y)
gs1.update(left=0.05, right=0.95)#, wspace=0.05)
ax1 = plt.subplot(gs1[:0, :0])

x=[1,2,3,4]
y=[1,3,6,8]

plt.plot(x,y)

#ax1.set_yticks(list())
#ax1.set_yticklabels(list())
#ax1.set_xticks(list())
#ax1.set_xticklabels(list())
#ax1.xaxis.set_major_locator(pylab.NullLocator())
#plt.gca().xaxis.set_major_locator(plt.NullLocator())
#plt.tick_params(
#    axis='y',          # changes apply to the x-axis
#    which='both',      # both major and minor ticks are affected
#    bottom='off',      # ticks along the bottom edge are off
#    top='off',         # ticks along the top edge are off
#    labelbottom='off', # labels along the bottom edge are off
#    right='off', 
#    left='off', 
#    labelleft='off')
#plt.axis('on')
#ax1.set_xticklabels([])
#ax1.set_yticklabels([])
ax1 = plt.subplots_adjust(hspace = .00)

ax2 = plt.subplot(gs1[:1, :])
plt.axis('on')
ax2.set_xticklabels([])
ax2.set_yticklabels([])

ax3 = plt.subplot(gs1[1:2, :])
plt.axis('on')
ax3.set_xticklabels([])
ax3.set_yticklabels([])

ax4 = plt.subplot(gs1[2:3, :])
plt.axis('on')
ax4.set_xticklabels([])
ax4.set_yticklabels([])
#ax2 = plt.subplot(gs1[:, :-1])
#ax3 = plt.subplot(gs1[-1, -1])

gs2 = gridspec.GridSpec(3, 1)
gs2.update(left=0.95, right=0.98)#, hspace=0.05)
ax4 = plt.subplot(gs2[0:1, :])
plt.axis('on')
ax4.set_xticklabels([])
ax4.set_yticklabels([])
#ax5 = plt.subplot(gs2[:-1, -1])
#ax6 = plt.subplot(gs2[-1, -1])
#gs = gridspec.GridSpec(3, 2,
#                       width_ratios=[20,1])
#ax11 = plt.subplot(gs[0, 0:1])
#plt.axis('on')
#ax11.set_xticklabels([])

#ax12 = plt.subplot(gs[0, 1:2])
#plt.axis('on')
#ax12.set_xticklabels([])
#ax12.set_yticklabels([])
#plt.tight_layout()

#ax21 = plt.subplot(gs[1, 0:1])
#plt.axis('on')
#ax21.set_xticklabels([])
#ax31 = plt.subplot(gs[2, 0:1])

#gs1 = gridspec.GridSpec(1, 1)
#ax00 = plt.subplot(gs1[0, 0:1])


#ax1 = plt.subplot2grid((3,2), (0,0))
#plt.axis('on')
#ax1.set_xticklabels([])
#plt.tight_layout()
#plt.subplots_adjust(hspace = .001)

#ax2 = plt.subplot2grid((3,2), (0,1))
#plt.axis('on')
#ax2.set_xticklabels([])
#ax2.set_yticklabels([])
#plt.tight_layout()
#plt.subplots_adjust(hspace = .001)

#ax3 = plt.subplot2grid((3,2), (1,0), colspan=2)
#plt.axis('on')
#ax3.set_xticklabels([])
#plt.tight_layout()
#plt.subplots_adjust(hspace = .001)

#ax4 = plt.subplot2grid((3,2), (2,0), colspan=2)
#plt.tight_layout()


#gs = gridspec.GridSpec(2, 2,
#                       width_ratios=[20,1],
#                       height_ratios=[1,1]
#                       )
#plt.axes.xaxis.set_major_locator(NullLocator)
#ax1 = plt.subplot(gs[0])
#ax2 = plt.subplot(gs[1])

#gs1 = gridspec.GridSpec(2, 1)
#gs1.update(left=0.05, right=0.78, wspace=0.05, hspace=0.12)
#ax1 = plt.subplot(gs1[0, 0])
#ax11= plt.subplot(gs1[0, 1])
#ax4 = plt.subplot(gs1[:-1, -1])
#ax2 = plt.subplot(gs1[-1, :-1])
#ax3 = plt.subplot(gs1[-1, -1])


#gs2 = gridspec.GridSpec(2, 1)
#gs2.update(left=0.78, right=0.80, wspace=0.05, hspace=0.12)
#ax11= plt.subplot(gs1[0, 0], c='r')

#gs1.update(left=0.80, right=0.85, wspace=0.05, hspace=0.12)
#ax1 = plt.subplot(gs1[:-1, :-1])
#gs2 = gridspec.GridSpec(2, 2)
#gs2.update(left=0.55, right=0.98, hspace=0.05)
#ax4 = plt.subplot(gs2[:, :-1])
#ax5 = plt.subplot(gs2[:-1, -1])
#ax6 = plt.subplot(gs2[-1, -1])

plt.savefig('stokes_new.ps', bbox_inches='tight')

