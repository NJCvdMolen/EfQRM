#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estnorm_score.py

Purpose:
    Estimate a normal regression model, using lambda function

Version:
    1       Following estnorm.ox, using 1d parameter vectors
    score   Using the score

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
    dS= np.fabs(vP[0])
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
def Generate(vP, iN):
    """
    Purpose:
      Generate regression data

    Inputs:
      vP        iK vector of parameters
      iN        integer, number of observations

    Return values:
      vY        iN vector of data
      mX        iN x iK matrix of regressors, constant + uniforms
    """
    (dS, vBeta)= GetPars(vP)

    iK= len(vBeta);

    mX= np.hstack([np.ones((iN, 1)), np.random.rand(iN, iK-1)])

    vY= mX@vBeta + dS * np.random.randn(iN)
    #print ("Y: ", vY.shape)

    return (vY, mX)

###########################################################
### dALL= AvgNLnLRegr(vP, vY, mX)
def AvgNLnLRegr(vP, vY, mX):
    """
    Purpose:
        Compute average negative loglikelihood of regression model

    Inputs:
        vP      iK+1 vector of parameters, with sigma and beta
        vY      iN vector of data
        mX      iN x iK matrix of regressors

    Return value:
        dLL     double, average loglikelihood
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
    dALL= np.mean(vLL, axis= 0)

    print (".", end="")             # Give sign of life

    return -dALL

###########################################################
### vSc= AvgNLnLRegr_Jac(vP, vY, mX)
def AvgNLnLRegr_Jac(vP, vY, mX):
    """
    Purpose:
        Compute score of average negative loglikelihood of regression model

    Inputs:
        vP      iK+1 vector of parameters, with sigma and beta
        vY      iN vector of data
        mX      iN x iK matrix of regressors

    Return value:
        vSc     iK+1 vector, score
    """
    (iN, iK)= mX.shape
    if (np.size(vP) != iK+1):         # Check if vP is as expected
        print ("Warning: wrong size vP= ", vP)

    (dSigma, vBeta)= GetPars(vP)
    vE= vY - mX @ vBeta

    vSc= np.zeros(iK+1)
    vSc[0]= (1 - np.mean(np.square(vE/dSigma), axis= 0))/dSigma
    vSc[1:]= -mX.T@vE/(iN*np.square(dSigma))

    print ("g", end="")             # Give sign of life

    return vSc

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

    dLL= -iN*AvgNLnLRegr(vP0, vY, mX)
    print ("\nInitial LL= ", dLL, "\nvP0=", vP0)

    # Check score
    vSc0= AvgNLnLRegr_Jac(vP0, vY, mX)
    vSc1= opt.approx_fprime(vP0, AvgNLnLRegr, 1e-5*np.fabs(vP0), vY, mX)
    vSc2= gradient_2sided(AvgNLnLRegr, vP0, vY, mX)
    print ("\nScores:\n", 
           pd.DataFrame(np.vstack([vSc0, vSc1, vSc2]), index=["Analytical", "grad_1sided", "grad_2sided"]))

    dErr= np.sqrt(np.mean((vSc0 - vSc1)**2, axis= 0))
    if (dErr > 1e-3):
        print ("Warning: Implementation of gradient gives error= ", dErr)

    # Optimize without jacobian
    # res= opt.minimize(AvgNLnLRegr, vP0, args=(vY, mX), method="BFGS")
    # Optimize with jacobian
    res= opt.minimize(AvgNLnLRegr, vP0, args=(vY, mX), method="BFGS",
            jac=AvgNLnLRegr_Jac)

    vP= res.x
    sMess= res.message
    dLL= -iN*res.fun
    print ("\nBFGS results in ", sMess, "\nPars: ", vP, "\nLL= ", dLL, ", f-eval= ", res.nfev)

    mS2= GetCovML(AvgNLnLRegr, vP, iN, vY, mX)
    vS= np.sqrt(np.abs(np.diag(mS2)))

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
    (vY, mX)= Generate(vP0, iN)

    (vP, vS, dLnPdf, sMess)= EstimateRegr(vY, mX)
    Output(np.vstack([vP0, vP, vS]), dLnPdf, sMess);

###########################################################
### start main
if __name__ == "__main__":
    main()
