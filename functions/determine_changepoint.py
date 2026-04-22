# Code related to the paper:
# Fiedler, Hainzl, Zöller & Holschneider, Detection of Gutenberg–Richter
# b-Value Changes in Earthquake Time Series, BSSA, 108 (5A), 2778–2787, 2018

import numpy as np
from scipy.special import gammaln
from scipy.special import gammainc


def index_loglhchangepoint1(x, betamax):
    '''
    Loglikelihood function

    x:        magnitude values
    betamax:  prior distribution of beta = [0, betamax]
    return:
    cp_index: index of the change point = maximum LL-value
    '''
    n = len(x)
    logllh = np.zeros(n-2)
    for i in range(1, n-1):
        logllh[i-1] = - np.log(betamax*betamax)
        logllh[i-1] += - (i+1) * np.log(sum(x[0:i+1]))
        logllh[i-1] += gammaln(i+1)
        logllh[i-1] += np.log(gammainc(i+1, betamax*sum(x[0:i+1])))
        logllh[i-1] += - (n-i+1) * np.log(sum(x[i+1:n+1]))
        logllh[i-1] += gammaln(n-i+1)
        logllh[i-1] += np.log(gammainc(n-i+1, betamax*sum(x[i+1:n+1])))
    cp_index = np.argmax(logllh)
    return cp_index


def bayesfactor01(x, betamax):
    '''
    Calculation of the Bayesfactor B01

    x:       magnitude values
    betamax: prior distribution of beta = [0, betamax]
    return:
    B01:     Bayesfactor
    '''
    n = len(x)
    SUMm = sum(x[0:n])
    nenner = 0
    for i in range(n-1):
        n1 = len(x[0:i+1])
        SUM1 = sum(x[0:i+1])
        n2 = len(x[i+1:n])
        SUM2 = sum(x[i+1:n])
        if SUM1 == 0:
            print('i=%d   n=%d  SUM1=%f' % (i, n, SUM1))
        if SUM2 == 0:
            print('i=%d   n=%d  SUM2=%f' % (i, n, SUM2))
        exponent = -(n1+1) * np.log(SUM1) + gammaln(n1+1)
        exponent += np.log(gammainc(n1+1, betamax*SUM1))
        exponent += -(n2+1) * np.log(SUM2) + gammaln(n2+1)
        exponent += np.log(gammainc(n2+1, betamax*SUM2))
        exponent += - (-(n+1) * np.log(SUMm) + gammaln(n+1) +
                       np.log(gammainc(n+1, betamax*SUMm)))
        q = np.exp(exponent)
        nenner += q
    zaehler = betamax * (n - 1.)
    if nenner == 0.:
        B01 = 10.
        print('numerical problems --> bayesfactor set to 10: n=%d' % (n))
    else:
        B01 = zaehler / nenner
    return B01


def determineCP_Bayes(Bthreshold, m, i1, i2, betamax):
    '''
    Calculate the position of a change point between m[i1] and m[i2]

    return:
    sig=1:  change point is significant
    sig=0:  change point is not significant
    CPi:    position of the change point
    '''
    mm = m[i1:i2+1]
    N = len(mm)
    Nmin = 10
    if N < 2 * Nmin:
        sig = int(0)
        CPi = i1
    else:
        B01 = bayesfactor01(mm, betamax)
        if B01 >= Bthreshold:
            sig = int(0)
            CPi = i1
        else:
            sig = int(1)
            kmin = index_loglhchangepoint1(mm, betamax)
            if kmin >= Nmin and N-kmin > Nmin:
                sig = int(1)
            else:
                sig = int(0)
            CPi = int(i1+kmin)
    return (sig, CPi)


def findCP_bvalues(m):
    '''
    Iterative selection of change-points with significant Bayes-factor

    INPUT:  m:    magnitude values of the sequence
    OUTPUT: chps: significant change points (p<=alpha)
    '''
    # Bthreshold: Bayesfactor-threshold to be significant (B < Bthreshold)
    Bthreshold = 0.5
    # bmax: prior distribution of b = [0, bmax]
    bmax = 3.0
    betamax = np.log(10)*bmax
    chp = []
    N = len(m)
    (sig, chp) = determineCP_Bayes(Bthreshold, m, 0, N-1, betamax)
    return chp if sig else np.nan
