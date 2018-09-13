import readcapm4
import numpy as np
import pandas as pd
import statsmodels.api as sm


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


def regress_set(df):
    """

    :param df:
    :return:
    """

    vX = sm.add_constant(df[df.columns[0]]).values

    for col in df.columns[1:]:
        vY = df[col].values
        model = sm.OLS(vY, vX).fit()


def regress(df):
    """
    Does regression stuff
    :return:
    """
    regress_set(df)

    for year in df.index.year.unique():
        regress_set(df.loc[df.index.year == year])


def optimize(df):
    """

    :param df:
    :return:
    """


def main():
    #Magic 'Numbers'
    filepath = "data/capm.csv"

    #init
    e_returns_df = init_data(filepath)
    regress(e_returns_df)


if __name__ == "__main__":
    main()
