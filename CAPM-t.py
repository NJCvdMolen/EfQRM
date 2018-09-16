import readcapm4
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.optimize as opt
import scipy.stats as st
import matplotlib.pyplot as plt
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
      Kevin Sheppard, adapted by Charles Bos, adapted by Niels van der Molen

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
    nu = False
    if vP.shape[0] ==4: #make sure degrees of freedom is correct
        nu = True

    f = fun(vP, *args)
    vh= _gh_stepsize(vP)
    vPh = vP + vh
    vh = vPh - vP

    mh = np.diag(vh)            # Build a diagonal matrix out of vh

    fp = np.zeros(iP)
    fm = np.zeros(iP)
    for i in range(iP):
        vPpT = vP + mh[i]
        vPmT = vP - mh[i]
        if nu: #make sure degrees of freedom is correct
            vPpT[1] = vP[1]
            vPmT[1] = vP[1]
        fp[i] = fun(vPpT, *args)
        fm[i] = fun(vPmT, *args)

    fpp = np.zeros((iP,iP))
    fmm = np.zeros((iP,iP))
    for i in range(iP):
        for j in range(i,iP):
            vPpT = vP + mh[i] + mh[j]
            vPmT = vP - mh[i] - mh[j]

            if nu: #make sure degrees of freedom is correct
                vPpT[1] = vP[1]
                vPmT[1] = vP[1]
            fpp[i,j] = fun(vPpT, *args)
            fpp[j,i] = fpp[i,j]
            fmm[i,j] = fun(vPmT, *args)
            fmm[j,i] = fmm[i,j]

    vh = vh.reshape((iP,1))
    mhh = vh @ vh.T             # mhh= h h', outer product of h-vector

    mH = np.zeros((iP,iP))
    for i in range(iP):
        for j in range(i,iP):
            mH[i,j] = (fpp[i,j] - fp[i] - fp[j] + f + f - fm[i] - fm[j] + fmm[i,j])/mhh[i,j]/2
            mH[j,i] = mH[i,j]

    print(mH)
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
    # save the column names
    names = df.columns
    # takes the log differences of the market proxy
    market_returns = np.diff(np.log(df[names[0]]))

    # divides the interest rate by 100 such that we have dicimals
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


def GetPars_norm(vP):
    """
    Purpose:
      Read out the parameters from the vector

    Inputs:
      vP        iK+1 vector with sigma and beta's

    Return value:
      dS        double, sigma
      ddf       degrees of freedom
      vBeta     iK vector, beta's
    """
    iK= np.size(vP)-1
    # Force vP to be a 1D matrix
    vP= vP.reshape(iK+1,)
    dS= vP[0]   # np.fabs(vP[0])
    vBeta= vP[1:]

    return dS, vBeta

def GetPars_student_t(vP):
    """
    Purpose:
      Read out the parameters from the vector

    Inputs:
      vP        iK+1 vector with sigma and beta's

    Return value:
      dS        double, sigma
      ddf       degrees of freedom
      vBeta     iK vector, beta's
    """
    iK= np.size(vP)-1
    # Force vP to be a 1D matrix
    vP= vP.reshape(iK+1,)
    dS= vP[0]   # np.fabs(vP[0])
    ddf = vP[1]
    vBeta= vP[2:]

    return dS, ddf, vBeta


def LnLRegr_norm(vP, vY, mX):
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

    (dSigma, vBeta)= GetPars_norm(vP)
    if (dSigma <= 0):
        print ("x", end="")
        return -math.inf

    vE= vY - mX @ vBeta

    vLL= -0.5*(np.log(2*np.pi) + 2*np.log(dSigma) + np.square(vE/dSigma))
    dLL= np.sum(vLL, axis= 0)

    print (".", end="")             # Give sign of life

    return dLL


def EstimateRegr_normal(mX, vY):
    """
    Optimize the regression model using

    Adapted from function created by Charles Bos
    :param df:
    :return:
    """

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

    (iN, iK) = mX.shape
    vP0 = np.ones(iK + 1)  # Get (bad...) starting values

    # vB= np.linalg.lstsq(mX, vY)[0]
    # vP0= np.vstack([[[1]], vB])

    dLL = LnLRegr_norm(vP0, vY, mX)
    print("Initial LL= ", dLL, "\nvP0=", vP0)

    # Create lambda function returning NEGATIVE AVERAGE LL, as function of vP only
    AvgNLnLRegr = lambda vP: -LnLRegr_norm(vP, vY, mX) / iN
    # Create function returning NEGATIVE average LL, as function of vP, vY, mX
    # AvgNLnLRegrXY= lambda vP, vY, mX: -LnLRegr(vP, vY, mX)/iN

    bounds = [(0, None)]
    for i in range(iK):
        bounds.append((None, None))

    res = opt.minimize(AvgNLnLRegr, vP0, args=(), method="BFGS")
    # res= opt.minimize(AvgNLnLRegrXY, vP0, args=(vY, mX), method="BFGS")

    vP = res.x
    sMess = res.message
    dLL = -iN * res.fun
    print("\nBFGS results in ", sMess, "\nPars: ", vP, "\nLL= ", dLL, ", f-eval= ", res.nfev)

    mS2 = GetCovML(AvgNLnLRegr, vP, iN)
    vS = np.sqrt(np.diag(mS2))
    print(mS2)
    exit()

    return vP, vS, dLL, sMess


def LnLRegr_student_t(vP, vY, mX):
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

    (dSigma, dDegFree, vBeta) = GetPars_student_t(vP)
    if (dSigma <= 0):
        print ("x1", end="")
        return -math.inf

    if (dDegFree < 3):
        print("x2", end="")
        return -math.inf

    vE= vY - mX @ vBeta

    vLL = st.t.logpdf(x=vE, df=dDegFree, loc=np.sqrt(dSigma))
    dLL= np.sum(vLL, axis=0)

    print(".", end="")             # Give sign of life

    return dLL


def EstimateRegr_student_t(mX, vY):
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
    vP0[1] = 3

    # vB= np.linalg.lstsq(mX, vY)[0]
    # vP0= np.vstack([[[1]], vB])

    dLL= LnLRegr_student_t(vP0, vY, mX)
    print("Initial LL= ", dLL, "\nvP0=", vP0)

    # Create lambda function returning NEGATIVE AVERAGE LL, as function of vP only
    AvgNLnLRegr_t= lambda vP: -LnLRegr_student_t(vP, vY, mX)/iN
    # Create function returning NEGATIVE average LL, as function of vP, vY, mX
    # AvgNLnLRegrXY= lambda vP, vY, mX: -LnLRegr(vP, vY, mX)/iN

    bounds = [(0, None), (3, None)]
    for i in range(iK):
        bounds.append((None, None))

    res= opt.minimize(AvgNLnLRegr_t, vP0, args=(), method="L-BFGS-B", bounds=bounds)
    # res= opt.minimize(AvgNLnLRegrXY, vP0, args=(vY, mX), method="BFGS")
    vP= res.x
    sMess= res.message
    dLL= -iN*res.fun
    print ("\nBFGS results in ", sMess, "\nPars: ", vP, "\nLL= ", dLL, ", f-eval= ", res.nfev)
    vP2 = get_pars_student_no_df(vP)
    mS2= GetCovML(AvgNLnLRegr_t, vP, iN)
    vS= np.sqrt(np.diag(mS2))


    return vP, vS, dLL, sMess

def get_pars_student_no_df(vP):
    """Returns parameters without degrees of freedom"""
    vPn = np.zeros(vP.shape[0]-1)
    vPn[0] = vP[0]
    vPn[1] = vP[2]
    vPn[2] = vP[3]
    return vPn

def GetParNames_norm(iK):
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

def GetParNames_student(iK):
    """
    Purpose:
      Construct names for the parameters from the vector

    Inputs:
      iK        integer, number of beta's

    Return value:
      asP       iK array, with strings "sigma", "b1", ...
    """
    asP= ["B"+str(i+1) for i in range(iK)]
    asP = ["Degrees of freedom"] + asP
    asP= ["Sigma"] + asP
    return asP

def Output_norm(mPPS, dLL, sMess):
    """
    Purpose:
      Provide output on screen
    """
    iK= mPPS.shape[1]-1
    print ("\n\nEstimation resulted in ", sMess)
    print ("Using ML with LL= ", dLL)

    print ("Parameter estimates:\n",
           pd.DataFrame(mPPS.T, index=GetParNames_norm(iK), columns=["PHat", "s(P)"]))

def Output_student(mPPS, dLL, sMess):
    """
    Purpose:
      Provide output on screen
    """

    iK= mPPS.shape[1]-2
    print ("\n\nEstimation resulted in ", sMess)
    print ("Using ML with LL= ", dLL)

    print ("Parameter estimates:\n",
           pd.DataFrame(mPPS.T, index=GetParNames_student(iK), columns=["PHat", "s(P)"]))


def optimize(df):
    """
    Performs the ML optimization
    :param: df a dataframe with sp500 and ibm data
    :return: returns results for both optimizations
    """
    mX = sm.add_constant(df["sp500"])
    vY = df["ibm"]
    print('Optimizing with normal')
    vPn, vSn, dLnPdfn, sMessn = EstimateRegr_normal(mX, vY)
    Output_norm(np.vstack([vPn, vSn]), dLnPdfn, sMessn)
    print('Optimizing with student t')
    vPt, vSt, dLnPdft, sMesst = EstimateRegr_student_t(mX, vY)
    Output_student(np.vstack([vPt, vSt]), dLnPdft, sMesst)

    return (vPn, vSn, dLnPdfn, sMessn), (vPt, vSt, dLnPdft, sMesst)

def plots(df, res):
    """
    Creates several plots for the report
    :param: df, dataframe with sp500 and ibm return data,
            res, the result of the optimization
    :return:
    """

    mX = sm.add_constant(df["sp500"])
    vY = df["ibm"]
    vP, vS, dLnPdf, sMess = res

    plt.subplot(1, 2, 1)
    plt.scatter(df["sp500"], df["ibm"])
    plt.title('CAPM normal regression')
    plt.xlabel('SP-500 excess returns')
    plt.ylabel('IBM excess returns')
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = vP[1] + vP[2] * x_vals
    plt.plot(x_vals, y_vals, color='k', linestyle='-', linewidth=2)

    plt.subplot(1, 2, 2)
    residuals = vY - np.dot(sm.add_constant(mX), [vP[1], vP[2]])

    mu, std = st.norm.fit(residuals)

    plt.hist(residuals, bins=25, density=True, alpha=0.6, edgecolor='black')
    xmin, xmax = plt.xlim()
    x_dist = np.linspace(xmin, xmax, 100)
    pdf = st.norm.pdf(x_dist, mu, std)
    plt.plot(x_dist, pdf, 'k')
    plt.title('Histogram Residuals and normal distribution')
    plt.show()


def regress_set(df):
    """
    Regress a data set, assuming the first column is a constant
    :param df:
    :return: a list of models for each regression
    """

    vX = sm.add_constant(df[df.columns[0]]).values

    model_list = []
    for col in df.columns[1:]:
        vY = df[col].values
        model_list.append(sm.OLS(vY, vX).fit())

    return model_list


def regression_plots(dataframes_year, lists_full, stock_names):
    """
    Processes the model list into plots
    :param model_list: a list of regression models
    :param stock_names: a list of stock names
    :return: None
    """

    beta0_list, beta1_list = lists_full
    beta0_yearlist, beta1_yearlist = dataframes_year

    fig = plt.figure()
    fig.set_size_inches(12, 5)
    plt.subplot(1, 2, 1)
    plt.plot(beta0_yearlist)
    plt.legend(stock_names)
    plt.title("Beta0")
    plt.subplot(1, 2, 2)
    plt.plot(beta1_yearlist)
    plt.legend(stock_names)
    plt.title("Beta1")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i in range(0, 4):
        plt.subplot(1, 2, 1)
        plt.axhline(y=beta0_list[i], color=colors[i], linestyle='--')
        plt.subplot(1, 2, 2)
        plt.axhline(y=beta1_list[i], color=colors[i], linestyle='--')

    plt.show()


def regress(df):
    """
    Performes several regressions
    :return: None
    """

    #regress on daily data
    model_full_list = regress_set(df)
    beta0_full = []
    beta1_full = []
    betap_full = []
    betas_full = []

    for model in model_full_list:
        beta0 = model.params[0]
        beta1 = model.params[1]
        betap_full.append(model.t_test('const = 0, x1 = 1').pvalue)
        betas_full.append(model.bse)
        beta0_full.append(beta0)
        beta1_full.append(beta1)

    full_lists = (beta0_full, beta1_full)

    model_year_list = []
    for year in df.index.year.unique():
        model_year_list.append(regress_set(df.loc[df.index.year == year]))

    beta0_year = []
    beta1_year = []
    betap_year = []
    betas_year = []

    for year in model_year_list:
        beta0_yeartemp = []
        beta1_yeartemp = []
        betap_yeartemp = []
        betas_yeartemp = []
        for model in year:
            betap_yeartemp.append(model.t_test('const = 0, x1 = 1').pvalue)
            betas_yeartemp.append(model.bse)
            beta0_yeartemp.append(model.params[0])
            beta1_yeartemp.append(model.params[1])

        beta0_year.append(beta0_yeartemp)
        beta1_year.append(beta1_yeartemp)
        betap_year.append(betap_yeartemp)
        betas_year.append(betas_yeartemp)

    df_beta0_year = pd.DataFrame(beta0_year)
    df_beta1_year = pd.DataFrame(beta1_year)
    df_beta0_year.index = df.index.year.unique()
    df_beta1_year.index = df.index.year.unique()

    cols = ['AIG', 'IBM', 'Ford', 'XOM']

    df_beta0_year.columns = cols
    df_beta1_year.columns = cols

    regression_plots((df_beta0_year, df_beta1_year), full_lists, cols)

    #make a table in copy ready format
    for j in range(0, 19):
        print(j, end="")
        for i in range(0, 4):
            hline = "\\\ "
            if i == 3:
                hline = hline + "\hline"
            print(" & ", cols[i], " & ", round(beta1_year[j][i],5), " & ", round(betas_year[j][i][1],5), " & ", round(betap_year[j][i][1],5), hline)




def main():
    #Magic 'Numbers'
    filepath = "data/capm.csv"

    #init
    e_returns_df = init_data(filepath)

    #output
    regress(e_returns_df)
    #res_n, res_t = optimize(e_returns_df)
    #plots(e_returns_df, res_n)


if __name__ == "__main__":
    main()
