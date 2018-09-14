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
#import matplotlib.pyplot as plt
import scipy.optimize as opt
import math

###########################################################
### Get hessian and related functions
from lib.ppectr import *

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
    vBeta= vP[1:]

    return (dS, vBeta)

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
    mX= np.hstack([np.ones((iN, 1)), np.random.rand(iN, iK-1)])

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
    (dS, vBeta)= GetPars(vP)
    vY= mX@vBeta + dS * np.random.randn(iN)

    return vY

###########################################################
### dLL= LnLRegr(vP, vY, mX)
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
    if (np.size(vP) != iK+1):         # Check if vP is as expected
        print ("Warning: wrong size vP= ", vP)

    (dSigma, vBeta)= GetPars(vP)
    if (dSigma <= 0):
        print ("x", end="")
        return -math.inf

    vE= vY - mX @ vBeta

    vLL= -0.5*(np.log(2*np.pi) + 2*np.log(dSigma) + np.square(vE/dSigma))
    dLL= np.sum(vLL, axis= 0)

    print (".", end="")             # Give sign of life

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
    vP0= np.ones(iK+1)        # Get (bad...) starting values

    # vB= np.linalg.lstsq(mX, vY)[0]
    # vP0= np.vstack([[[1]], vB])

    dLL= LnLRegr(vP0, vY, mX)
    print ("Initial LL= ", dLL, "\nvP0=", vP0)

    # Create lambda function returning NEGATIVE AVERAGE LL, as function of vP only
    AvgNLnLRegr= lambda vP: -LnLRegr(vP, vY, mX)/iN
    # Create function returning NEGATIVE average LL, as function of vP, vY, mX
    # AvgNLnLRegrXY= lambda vP, vY, mX: -LnLRegr(vP, vY, mX)/iN

    res= opt.minimize(AvgNLnLRegr, vP0, args=(), method="BFGS")
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
    vP0= [.1, 5, 2, -2]    #dSigma and vBeta together
    iN= 100
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
