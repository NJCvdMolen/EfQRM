import readcapm4
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.optimize as opt
import math
import matplotlib.pyplot as plt

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


def clean_data(df):
    """
    Cleans a dataframe
    :param df: a dataframe containing price data
    :return: a cleaned dataframe containing price data
    """

    return df[(df[[df.columns[2]]] != '.').all(axis=1)].set_index(df.columns[0])


def get_returns(df):
    """
    Calculates log returns
    :param df: a cleaned dataframe containing price data
    :return: a dataframe with log returns and rates (in percentages)
    """
    df.index = pd.to_datetime(df.index)
    df = df[df.index.year >= 2000]
    # save the column names
    names = df.columns
    # takes the log differences of the market proxy
    market_returns = np.diff(np.log(df[names[0]]))

    # divides the interest rate by 100 such that we have dicimals
    # this makes them on the same scale as the log returns
    rates = df[names[1]].divide(100, 'rows')[1:len(df.index)]

    # makes the stacks the market returns on top of the interest rates
    df_result = np.vstack((market_returns, rates))

    # loops over the stocks taking the log differences and adding them to the dataframe
    for i in names[2:len(names)]:
        stock_returns_i = np.diff(np.log(df[i]))
        df_result = np.vstack((df_result, stock_returns_i))

    # chages the stacked tabel to a dataframe
    # vstack gives an (6Xn)-matrix, thus if we transpose it we obtain a (nX6)-matrix
    df_result = pd.DataFrame(df_result).T

    # remove the FIRST index
    df_result.index = pd.to_datetime(df.index[1:len(df.index)])

    # gives the dataframe with daily log returns the same column names as the input dataframe
    df_result.columns = df.columns
    return df_result.loc[:, :]*100


def daily_excess_returns(df):
    # set interest rates to daily
    df[df.columns[1]] = df[df.columns[1]].divide(250, 'row')

    # substract daily interest rates from market log retruns
    df[df.columns[0]] = df[df.columns[0]] - df[df.columns[1]]

    # substract daily interest rates from stock log returns
    for i in df.columns[2:]:
        df[i] = df[i] - df[df.columns[1]]

        # remove the interest rate column
    df = df.drop(df.columns[1], 1)

    # drop rows with na values that are possibly created by the data manipulation
    return df.dropna(axis=0)


def init_data(filepath):
    """
    Initializes the DataFrame
    :param filepath: a string with the path to the source csv file
    :return: a data frame with excess returns
    """
    #readcapm4.main()
    clean_df = clean_data(pd.read_csv(filepath))
    returns_df = get_returns(clean_df)
    return daily_excess_returns(returns_df)


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

    bounds = [(0, None)]
    for i in range(iK):
        bounds.append((None, None))

    res= opt.minimize(AvgNLnLRegr, vP0, args=(), method="L-BFGS-B", bounds=bounds)
    # res= opt.minimize(AvgNLnLRegrXY, vP0, args=(vY, mX), method="BFGS")

    vP= res.x
    sMess= res.message
    dLL= -iN*res.fun
    print ("\nBFGS results in ", sMess, "\nPars: ", vP, "\nLL= ", dLL, ", f-eval= ", res.nfev)

    mS2= GetCovML(AvgNLnLRegr, vP, iN)
    vS= np.sqrt(np.diag(mS2))

    return (vP, vS, dLL, sMess, mS2)


###########################################################
### Output(mPPS, dLL, sMess)
def Output(mPPS, dLL, sMess,mS2):
    """
    Purpose:
      Provide output on screen
    """
    iK= mPPS.shape[1]-1
    print ("\n\nEstimation resulted in ", sMess)
    print ("Using ML with LL= ", dLL)

    print ("Parameter estimates:\n",
           pd.DataFrame(mPPS.T, index=GetParNames(iK), columns=["PHat", "s(P)"]))
    print('covariance matrix:')
    print(np.round(mS2, 6))


def main():
    #Magic 'Numbers'
    filepath = "data/capm.csv"

    #init
    e_returns_df = init_data(filepath)

    #We only need to regress on the IBM stock
    mX = sm.add_constant(e_returns_df["sp500"])
    vY = e_returns_df["ibm"]

    (vP, vS, dLnPdf, sMess, mS2) = EstimateRegr(vY, mX)
    Output(np.vstack([vP, vS]), dLnPdf, sMess, mS2);

    #plaatje
    fig = plt.figure()
    fig.set_size_inches(12, 8)

    plt.subplot(1, 2, 1)
    plt.scatter(e_returns_df["sp500"], e_returns_df["ibm"])
    plt.title('CAPM normal regression')
    plt.xlabel('SP-500 excess returns')
    plt.ylabel('IBM excess returns')
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = vP[1] + vP[2] * x_vals
    plt.plot(x_vals, y_vals, color='k', linestyle='-', linewidth=2)

    plt.subplot(1, 2, 2)
    residuals = vY - np.dot(sm.add_constant(mX), [vP[1], vP[2]])
    plt.hist(residuals, edgecolor='black')
    plt.title('Histogram Residuals')

    plt.show()


if __name__ == "__main__":
    main()
