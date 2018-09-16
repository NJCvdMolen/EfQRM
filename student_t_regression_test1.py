#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estnorm.py

Purpose:
    Estimate a normal regression model, using lambda function

Version:
    1       Following estnorm.ox, using 1d parameter vectors

Date:
    2017/8/21

@author: cbs310
"""
###########################################################
### Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as opt
import math

###########################################################
### vh= _gh_stepsize(vP)
def _gh_stepsize(vP):
    """
    Purpose:
        Calculate stepsize close (but not too close) to machine precision

    Inputs:
        vP      1D array of parameters

    Return value:
        vh      1D array of step sizes
    """
    vh = 1e-8*(np.fabs(vP)+1e-8)   # Find stepsize
    vh= np.maximum(vh, 5e-6)       # Don't go too small

    return vh

###########################################################
### vG= gradient_2sided(fun, vP, *args)
def gradient_2sided(fun, vP, *args):
    """
    Purpose:
      Compute numerical gradient, using a 2-sided numerical difference

    Author:
      Charles Bos, following Kevin Sheppard's hessian_2sided, with
      ideas/constants from Jurgen Doornik's Num1Derivative

    Inputs:
      fun     function, as used for minimize()
      vP      1D array of size iP of optimal parameters
      args    (optional) extra arguments

    Return value:
      vG      iP vector with gradient

    See also:
      scipy.optimize.approx_fprime, for forward difference
    """
    iP = np.size(vP)
    vP= vP.reshape(iP)      # Ensure vP is 1D-array

    # f = fun(vP, *args)    # central function value is not needed
    vh= _gh_stepsize(vP)
    mh = np.diag(vh)        # Build a diagonal matrix out of h

    fp = np.zeros(iP)
    fm = np.zeros(iP)
    for i in range(iP):     # Find f(x+h), f(x-h)
        fp[i] = fun(vP+mh[i], *args)
        fm[i] = fun(vP-mh[i], *args)

    vhr = (vP + vh) - vP    # Check for effective stepsize right
    vhl = vP - (vP - vh)    # Check for effective stepsize left
    vG= (fp - fm) / (vhr + vhl)  # Get central gradient

    return vG

###########################################################
### mG= jacobian_2sided(fun, vP, *args)
def jacobian_2sided(fun, vP, *args):
    """
    Purpose:
      Compute numerical jacobian, using a 2-sided numerical difference

    Author:
      Charles Bos, following Kevin Sheppard's hessian_2sided, with
      ideas/constants from Jurgen Doornik's Num1Derivative

    Inputs:
      fun     function, return 1D array of size iN
      vP      1D array of size iP of optimal parameters
      args    (optional) extra arguments

    Return value:
      mG      iN x iP matrix with jacobian

    See also:
      numdifftools.Jacobian(), for similar output
    """
    iP = np.size(vP)
    vP= vP.reshape(iP)      # Ensure vP is 1D-array

    vF = fun(vP, *args)     # evaluate function, only to get size
    iN= vF.size

    vh= _gh_stepsize(vP)
    mh = np.diag(vh)        # Build a diagonal matrix out of h

    mGp = np.zeros((iN, iP))
    mGm = np.zeros((iN, iP))

    for i in range(iP):     # Find f(x+h), f(x-h)
        mGp[:,i] = fun(vP+mh[i], *args)
        mGm[:,i] = fun(vP-mh[i], *args)

    vhr = (vP + vh) - vP    # Check for effective stepsize right
    vhl = vP - (vP - vh)    # Check for effective stepsize left
    mG= (mGp - mGm) / (vhr + vhl)  # Get central jacobian

    return mG

###########################################################
### mH= hessian_2sided(fun, vP, *args)
def hessian_2sided(fun, vP, *args):
    """
    Purpose:
      Compute numerical hessian, using a 2-sided numerical difference

    Author:
      Kevin Sheppard, adapted by Charles Bos

    Source:
      https://www.kevinsheppard.com/Python_for_Econometrics

    Inputs:
      fun     function, as used for minimize()
      vP      1D array of size iP of optimal parameters
      args    (optional) extra arguments

    Return value:
      mH      iP x iP matrix with symmetric hessian
    """
    iP = np.size(vP,0)
    vP= vP.reshape(iP)    # Ensure vP is 1D-array

    f = fun(vP, *args)
    vh= _gh_stepsize(vP)
    vPh = vP + vh
    vh = vPh - vP

    mh = np.diag(vh)            # Build a diagonal matrix out of vh

    fp = np.zeros(iP)
    fm = np.zeros(iP)
    for i in range(iP):
        fp[i] = fun(vP+mh[i], *args)
        fm[i] = fun(vP-mh[i], *args)

    fpp = np.zeros((iP,iP))
    fmm = np.zeros((iP,iP))
    for i in range(iP):
        for j in range(i,iP):
            fpp[i,j] = fun(vP + mh[i] + mh[j], *args)
            fpp[j,i] = fpp[i,j]
            fmm[i,j] = fun(vP - mh[i] - mh[j], *args)
            fmm[j,i] = fmm[i,j]

    vh = vh.reshape((iP,1))
    mhh = vh @ vh.T             # mhh= h h', outer product of h-vector

    mH = np.zeros((iP,iP))
    for i in range(iP):
        for j in range(i,iP):
            mH[i,j] = (fpp[i,j] - fp[i] - fp[j] + f + f - fm[i] - fm[j] + fmm[i,j])/mhh[i,j]/2
            mH[j,i] = mH[i,j]

    return mH

###########################################################
### mS2= GetCovML(fnNAvgLnL, vP, iN, *args)
def GetCovML(fnNAvgLnL, vP, iN, *args):
    """
    Purpose:
      Get the covariance matrix of the ML estimation

    Inputs:
      fnNAvgLnL   function, negative average loglikelihood
      vP          iP array of parameters
      iN          integer, number of observations
      *args       (optional) additional arguments to function

    Return value:
      mS2         iP x iP covariance matrix
    """
    mH= hessian_2sided(fnNAvgLnL, vP, *args)
    print(mH)
    try:
        mS2= np.linalg.inv(mH)/iN
    except:
        print ("Hessian not of full rank...")
        mS2= np.full(mH.shape, np.nan)

    mS2= (mS2 + mS2.T)/2            # Force mS2 to be symmetric

    return mS2

###########################################################
### vS= GetSDevML(fnNAvgLnL, vP, iN, *args)
def GetSDevML(fnNAvgLnL, vP, iN, *args):
    """
    Purpose:
      Get the standard deviations of the ML estimation

    Inputs:
      fnNAvgLnL   function, negative average loglikelihood
      vP          iP array of parameters
      iN          integer, number of observations
      *args       (optional) additional arguments to function

    Return value:
      vS          iP vector of standard deviations
    """
    mS2= GetCovML(fnNAvgLnL, vP, iN, *args)
    vS= np.sqrt(np.diag(mS2))

    return vS


###########################################################


def GetPars(vP):
    """
    Purpose:
      Read out the parameters from the vector

    Inputs:
      vP        iK+1 vector with sigma and beta's

    Return value:
      dS        double, sigma
      vBeta     iK vector, beta's
    """
    iK= np.size(vP)-1
    # Force vP to be a 1D matrix
    vP= vP.reshape(iK+1,)
    dS= vP[0]   # np.fabs(vP[0])
    ddf = vP[1]
    vBeta= vP[2:]

    return dS, ddf, vBeta


###########################################################
def GetParNames(iK):
    """
    Purpose:
      Construct names for the parameters from the vector

    Inputs:
      iK        integer, number of beta's

    Return value:
      asP       iK array, with strings "sigma", "b1", ...
    """
    asP= ["B"+str(i+1) for i in range(iK)]
    asP= ["Sigma"] + asP

    return asP

###########################################################
def GenrX(iN, iK):
    """
    Purpose:
      Generate regressors, constant + uniforms

    Inputs:
      iN        integer, number of observations
      iK        integer, number of regressors

    Return values:
      mX        iN x iK matrix of regressors, constant + uniforms
    """
    mX= np.hstack([np.ones((iN, 1)), np.random.rand(iN, iK-2)])

    return mX
###########################################################
def GenrY(vP, mX):
    """
    Purpose:
      Generate regression data

    Inputs:
      vP        iK+1 vector of parameters, sigma and beta
      mX        iN x iK matrix of regressors

    Return values:
      vY        iN vector of data
    """
    iN= mX.shape[0]
    (dS, ddf, vBeta)= GetPars(vP)
    vY= mX@vBeta + dS * np.random.standard_t(ddf, iN)

    return vY

###########################################################
###
def LnLRegr(vP, vY, mX):
    """
    Purpose:
        Compute loglikelihood of regression model

    Inputs:
        vP      iK+1 1D-vector of parameters, with sigma and beta
        vY      iN 1D-vector of data
        mX      iN x iK matrix of regressors

    Return value:
        dLL     double, loglikelihood
    """
    (iN, iK)= mX.shape
    if (np.size(vP) != iK+2):         # Check if vP is as expected
        print ("Warning: wrong size vP= ", vP)

    (dSigma, dDegFree, vBeta)= GetPars(vP)
    print(GetPars(vP))
    if (dSigma <= 0):
        print ("x", end="")
        return -math.inf

    if (dDegFree < 2):
        print("x", end="")
        return -math.inf

    vE= vY - mX @ vBeta

    vLL = st.t.logpdf(x=vE, df=dDegFree, loc=dSigma)
    dLL= np.sum(vLL, axis= 0)

    print(".", end="")             # Give sign of life
    print(dLL)

    return dLL

###########################################################
### (vP, vS, dLL, sMess)= EstimateRegr(vY, mX)
def EstimateRegr(vY, mX):
    """
    Purpose:
      Estimate the regression model

    Inputs:
      vY        iN vector of data
      mX        iN x iK matrix of regressors

    Return value:
      vP        iK+1 vector of optimal parameters sigma and beta's
      vS        iK+1 vector of standard deviations
      dLL       double, loglikelihood
      sMess     string, output of optimization
    """
    (iN, iK)= mX.shape
    vP0= np.ones(iK+2)        # Get (bad...) starting values
    vP0[1] = 2

    # vB= np.linalg.lstsq(mX, vY)[0]
    # vP0= np.vstack([[[1]], vB])

    dLL= LnLRegr(vP0, vY, mX)
    print ("Initial LL= ", dLL, "\nvP0=", vP0)

    # Create lambda function returning NEGATIVE AVERAGE LL, as function of vP only
    AvgNLnLRegr= lambda vP: -LnLRegr(vP, vY, mX)/iN
    # Create function returning NEGATIVE average LL, as function of vP, vY, mX
    # AvgNLnLRegrXY= lambda vP, vY, mX: -LnLRegr(vP, vY, mX)/iN

    bounds = [(0, None), (2, None)]
    for i in range(iK):
        bounds.append((None, None))

    res= opt.minimize(AvgNLnLRegr, vP0, args=(), method="L-BFGS-B", bounds= bounds)
    # res= opt.minimize(AvgNLnLRegrXY, vP0, args=(vY, mX), method="BFGS")

    vP= res.x
    sMess= res.message
    dLL= -iN*res.fun
    print ("\nBFGS results in ", sMess, "\nPars: ", vP, "\nLL= ", dLL, ", f-eval= ", res.nfev)

    mS2= GetCovML(AvgNLnLRegr, vP, iN)
    vS= np.sqrt(np.diag(mS2))

    return (vP, vS, dLL, sMess)

###########################################################
### Output(mPPS, dLL, sMess)
def Output(mPPS, dLL, sMess):
    """
    Purpose:
      Provide output on screen
    """
    iK= mPPS.shape[1]-1
    print ("\n\nEstimation resulted in ", sMess)
    print ("Using ML with LL= ", dLL)

    print ("Parameter estimates:\n",
           pd.DataFrame(mPPS.T, index=GetParNames(iK), columns=["PTrue", "PHat", "s(P)"]))


###########################################################
### main
def main():
    vP0= [.1, 20, 5, 2, -2]    #dSigma, dDegFree and vBeta together
    iN= 1000
    iSeed= 1234

    #Generate data
    np.random.seed(iSeed)
    vP0= np.array(vP0)

    iK= vP0.size - 1
    mX= GenrX(iN, iK)
    vY= GenrY(vP0, mX)

    (vP, vS, dLnPdf, sMess)= EstimateRegr(vY, mX)
    Output(np.vstack([vP0, vP, vS]), dLnPdf, sMess);

###########################################################
### start main
if __name__ == "__main__":
    main()
