#!/usr/bin/env python

def running_avr(x, step, flag):
    import numpy as np
    
#    x=[2,3,5,3,9,5,20,3,2,5]
#    x=[2,3,5,4,9,5,20]

    avr = []
    stddev=[]
    
    for j in xrange(len(x)-step+1):
        start = j 
        end   = j + step

        b = x[start:end]
        
        if (flag == 0 or flag == 2):
            avr_1    = np.average(b, axis=None, weights=None, returned=False)
            avr.append(avr_1)
#        print avr_1
        if (flag == 1 or flag == 2):
            stddev_1 = np.std(b, axis=None, dtype=None, out=None, ddof=0, keepdims=False)
            stddev.append(stddev_1)
        
#        print step, start, end, b, avr_1, stddev_1


    if (flag == 0):
        return avr
    elif (flag == 1):
        return stddev
    elif (flag == 2):
        return avr, stddev
    elif (flag != 0 or flag !=1):
        print 'wrong flag value:: exiting program'
        exit()




if __name__ == '__main__':
    import numpy as np
    x = [np.random.random() for n in xrange(10)]
    print running_avr(x, 5, 2)


