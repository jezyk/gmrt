#!/usr/bin/env python

def running_avr(x, step, flag, jump = None):
    import numpy as np
    if (jump is None):
        jump = 20
        if (jump < 1.):
            jump = 1
    else:
        print 'jump =',jump
        
    
#    x=[2,3,5,3,9,5,20,3,2,5,4,2,5,7,8,89,0,6,4,3,2]
#    x=[2,3,5,4,9,5,20]

    avr = []
    stddev=[]
#    step = int(step_param [0])
#    jump = int(step_param [1])

    if (len(x) < step):    
        print '(len(x) < step) in running_avr function' 
        # consider the two options below (change the step lenght or break the program)
        # step = len(x)
        # break or exit()

    start = 0
    for j in xrange(len(x)-step+1):
#        start = j 
        end   = start + step
        if (end > len(x)):
            break

        b = x[start:end]
        
        if (flag == 0 or flag == 2):
            avr_1    = np.average(b, axis=None, weights=None, returned=False)
            avr.append(avr_1)
#        print avr_1
        if (flag == 1 or flag == 2):
            stddev_1 = np.std(b, axis=None, dtype=None, out=None, ddof=0, keepdims=False)
            stddev.append(stddev_1)
        
#        print step, start, end, b, avr_1, stddev_1

        start = start + jump


    if (flag == 0):
        return avr
    elif (flag == 1):
        return stddev
    elif (flag == 2):
        return avr, stddev
    elif (flag != 0 or flag !=1):
        print 'wrong flag value:: exiting program'
        exit()


def SNRinSP(x, sdmin):
#    minval = np.min(x)
#    minpos = np.argmin(x)
    maxval = np.max(x)
    maxpos = np.argmax(x)
    SNR = maxval / sdmin
    return SNR, maxpos


def width(x, cond, Nsig, step_sdv):
    #Nsig used only for where width is searced at N sig level

    import numpy as np
    xsd = running_avr(x, step_sdv, 1)
    min_pos_xsd = np.argmin(xsd)
    xsd = xsd[min_pos_xsd]
#    print 'siew in functions',xsd

    maxxval = np.max(x)
    if (cond == 0):
        var = xsd * Nsig
    if (cond == 1):
        var = 0.1 * maxxval
    if (cond == 2):
        var = 0.5 * maxxval

    for i in xrange(len(x)):
        if (x[i] >= var):
            w_left = i
            break

    for i in xrange(len(x)-1, 0, -1): 
        if (x[i] >= var):
            w_right = i
            break

    w = w_right - w_left
    w_mp = w_left + w/2.  # midpoint

    return w, w_mp, w_left, w_right
    


def razor(flags, win, I, x, minsdv, N_SP, N_W):
# flags on/off options for razor elements
# signal window
# average profile
# Single Pulses
# array of min stddev in each SP

# N_SP sigma
#    N_SP = 3.

    import numpy as np
    import scipy
    import matplotlib.pyplot as plt

    print '====='
    print 'options switched on in razor:'
    if (int(flags[1]) == 1): # removes SP with RMS lower than N_SP*SDDEV(I)
        print '[SNR] = 1'
    if (int(flags[2]) == 1): # removes all/some(?) peaks/data in SP if higher then N_SP*SDDEV(I) outside the pulse window
        print '[lawn_mover] = 1'
    if (int(flags[3]) == 1): # removes SP with the RMS of analysed SP higher then N_SP*RMS from acumulated lowest RMSs (sorted) of all accepted SPs
        print '[high_stddev] = 1'
    if (int(flags[4]) == 1): # (removes SP) / checks normality of nosie distribution outside pulse window
        print '[noise distribution] = 1'
    dw = 0.05 * len(I)
    wl = int(win[0] - dw)
    wr = int(win[1] + dw)
    if (wr > len(I)):
        wr = len(I)-5
    if (wl < 0):
        wl = 0

    print 'win', win, wl, wr
    max_I = np.max(I)
    noise_I = running_avr(I, 100, 1)
    noise_I_pos = np.argmin(noise_I)
    noise_I = np.min(noise_I)

    SNR = max_I / noise_I
    Nsig= noise_I * N_SP

# method 1:: remove SP with where SNR is lower than N_SP sigma level
# 1 sigma is the lowest stddev in average I profile

#    print 'max_I, pos max_I', max_I, np.argmax(I)
    print 'noise I, pos. noise I, SNR, Nsig', noise_I, noise_I_pos, SNR, Nsig

# szukanie maximum w oknie pulsu dla kazdego SP osobno
    max_x = np.apply_along_axis(np.max, 1, x[:, wl:wr+1])
#    print 'max_x', max_x, max_x.shape

# reject SP with the min stddev of SP = 0
    accepted_0 = np.argwhere(max_x)
    rejected_0 = np.argwhere(max_x == 0)
    
#    for i in xrange(max_x.shape[0]):
#        if (max_x[i] > Nsig):
#            print ' max_x > Nsig', i, max_x[i]
#        if (max_x[i] <= Nsig):
#            print ' max_x <= Nsig', i, max_x[i]
#            np.append(rejected1, i)

#    print 'max_x', type(max_x)
#    rejected = np.argwhere(max_x < 2.)
#    print 'accepted by argwhere',accepted_0
    print 'rejected by argwhere',rejected_0
    print 'number of SP accepted by SNR option:', len(accepted_0)
    print 'number of SP rejected by SNR option:', len(rejected_0)
#    rejected1 = max_x[max_x <= Nsig]            
#    print 'rejected', rejected1

#    tmp = np.concatenate([I[:wl], I[wr:]])
#    print 'len concat.', len(tmp)
    

    print '====='
    print 'start:: method 2 '
# method 2:: lawn mover:: does something with peaks outside the signal window
# i) reject entire pulse; ii) replace peaks with a noise with 
# x0 = average value of noise and sigma = stddev "in 'nearby' area" of the peak

    a = x[0,:]
    max_x_in = np.max(a[wl:wr+1])

#    max_x_out = np.max(np.concatenate([a[0:wl], a[wr+1:]]) )
#    max_x_out_l = np.max(a[0:wl] )
#    max_x_out_r = np.max(a[wr+1:])

    print 'max_x_in, max_x_out', max_x_in#, max_x_out, Nsig, minsdv[0] * N_SP

#    out_high_l = np.where(a[0:wl] > minsdv[0] * N_SP)
#    out_high_r = np.where(a[wr+1:] > minsdv[0] * N_SP)
    
#    endl = out_high_l[:][0]
#    endr = out_high_r[:][0] + wr + 1
#    rejected_1 = np.append(endl, endr)
#    print 'rejected bins', rejected_1
    
#    a = x[0,:]
    max_x_in = np.apply_along_axis(np.max, 1, x[:,wl:wr+1])

    max_x_out = np.apply_along_axis(np.max, 1, np.concatenate([x[:, 0:wl], x[:, wr+1:]]) )
#    max_x_out_l = np.max(a[0:wl] )
#    max_x_out_r = np.max(a[wr+1:])

#    print 'max_x_in, max_x_out', max_x_in, max_x_out, Nsig, minsdv[0] * N_SP

#    out_high_l = np.where(a[0:wl] > minsdv[0] * N_SP)
#    out_high_r = np.where(a[wr+1:] > minsdv[0] * N_SP)
    
#    endl = out_high_l[:][0]
#    endr = out_high_r[:][0] + wr + 1
#    rejected_1 = np.append(endl, endr)
#    print 'rejected bins', rejected_1


    
#    for i in xrange(0):
#        a = x[i,:]
#        print a
#    plt.plot(a)
#    plt.show()


    print '====='
# method 3:: remove SP with the lowest stddev in SP higher than 'certain' level
    print 'start:: method 3 '

    sdi00 = np.apply_along_axis(np.mean, 1, x)# wartosc srednia z calego SP
    sdi11 = np.apply_along_axis(np.std, 1, x) # odchylenie z sdi00 
#    sdi111= np.mean(sdi11)
#    print sdi00, sdi11, sdi111
#
    sdi11_sort = zip(xrange(len(sdi11)), sdi11) # laczy kazdy SP z odpowiednim nr porzadkowym
    sdi11_sort = sorted(sdi11_sort, key=lambda param: param[1]) # sortuje rosnaco odchylenia z poszczegolnych SP
#    #print sdi11_sort




    val = []
    val = np.array(val)
    ls = N_SP # local sigma
#    print ls, type(ls)

    accepted2 = []
    accepted2 = np.array(accepted2)
    rejected2 = []
    rejected2 = np.array(rejected2)

##print sdi11_sort[0][1], sdi11_sort[1][1], len(sdi11_sort[:])-1, tresh

    for i in xrange(len(sdi11_sort[:])):
        if (i == 0):
            tresh = ls * (sdi11_sort[0][1])
    #        print tresh
        if (i != 0):
            tresh = ls * np.mean(val)
    
#        print tresh
    
        if (sdi11_sort[i][1] < tresh):
#            print i, sdi11_sort[i][1] , tresh
    
            val = np.append(val, sdi11_sort[i][1])
            accepted2 = np.append(accepted2,i)
    #        print i, val, tresh
        else:
 #           break
#            print i, tresh, sdi11_sort[i][1], 'rejected'
            rejected2 = np.append(rejected2, i)
#            print 'rejected'
    #        print i, sdi11_sort[i][0], sdi11_sort[i][1], tresh

    print 'accepted 2', accepted2
    print 'rejected 2', rejected2




    print '====='
    print 'start:: method 4 '
# method 4:: remove SP where the noise distribution outside the window is not gaussian

    print x.shape[0], x.shape[1]

    for i in xrange(x.shape[0]):
        out_window = np.concatenate((x[:, :wl], x[:, wr+1:]), axis = 1)
    print out_window.shape

#    out_window = np.apply_along_axis(np.concatenate(x[0,:wl], x[0,wr+1:] ), 1 , x)
    print 'out_window', type(out_window), type(x)
#np.apply_along_axis()

#    k, p = np.apply_along_axis(scipy.stats.mstats.normaltest(x, axis=1), axis=1, x)
    kn, pn = scipy.stats.mstats.normaltest(out_window, axis=1)
    wsw = np.apply_along_axis(scipy.stats.shapiro, 1, out_window)
    wks = np.apply_along_axis(scipy.stats.kstest, 1, out_window, 'norm', N=out_window.shape[1])

    print kn, pn
    print
    print wsw
    print
    print wks
    plt.hist(x[1 ,: ], 100, normed=1, histtype='bar')
    plt.show()

#    std_I  = running_avr(tmp, 100, 1)
#    std_I1 = running_avr(I[:wl], 100, 1)
#    std_I2 = running_avr(I[wr:], 100, 1)

#    print np.min(std_I), np.min(std_I1), np.min(std_I2)
#    print np.max(I)
#    print np.argmin(std_I), np.argmin(std_I1), np.argmin(std_I2)
#    print I.shape, x.shape

#    for i in xrange(5):
#        print I[0], x[i,0]
#        a = np.concatenate([x[i, :wl], x[i, wr:]])
#        print a, len(a)
#        print np.max (a), np.argmax(a), 5. * np.min(std_I)



    print '===== ===== ====='


if __name__ == '__main__':
    import numpy as np

    flag = 0
    x=[2,3,5,3,9,5,20,3,2,5,4,2,5,7,8,89,0,6,4,3,2]
    step = 5
    avr = []
    stddev=[]


    start = 0
    for j in xrange(len(x)-step+1):
        end   = start + step
        if (end > len(x))   :
            break     


        b = x[start:end]
        
        if (flag == 0 or flag == 2):
            avr_1    = np.average(b, axis=None, weights=None, returned=False)
            avr.append(avr_1)
#        print avr_1
        if (flag == 1 or flag == 2):
            stddev_1 = np.std(b, axis=None, dtype=None, out=None, ddof=0, keepdims=False)
            stddev.append(stddev_1)

        jump = 2
        print j, avr_1, start, end, jump
        start = start + jump

        
#        print step, start, end, b, avr_1, stddev_1



