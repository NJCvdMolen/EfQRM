#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
readcapm.py

Purpose:
    Read CAPM data

Version:
    2       Starting to get a working version
    2b      With hints from Elvan Toygarlar & Pim Burgers, much nicer
    3       Centering on IBM
    4       2018 version

Date:
    2017/9/5, 2018/9/3

@author: cbs310
"""
###########################################################
### Imports
import pandas as pd
# import matplotlib.pyplot as plt
# import scipy.optimize as opt

###########################################################
def ReadCAPM(asData, sPer):
    """
    Purpose:
        Read CAPM data

    Inputs:
        asData  list, with list of stocks, string with sp500, and string with
                TB indicator
        sPer    string, indicator for period

    Return value:
        df      dataframe, data
    """
    df= pd.DataFrame()
    # Get SP500 data
    df[asData[1]]= pd.read_csv("data/"+asData[1]+sPer+".csv", index_col="Date")["Adj Close"]

    # Add in TB
    df[asData[2]]= pd.read_csv("data/"+asData[2]+sPer+".csv", index_col="DATE", na_values=".")

    for sStock in asData[0]:
        df[sStock]= pd.read_csv("data/"+sStock+sPer+".csv", index_col="Date")["Adj Close"]

    df.index.name= "Date"
    df.index= pd.to_datetime(df.index)

    # For simplicity, drop rows with nans;
    #   this is more drastic than alternative program, which allowed for some missings
    df= df.dropna(axis=0, how='any')

    return df


def main():
    # Magic numbers
    asData= [["aig", "ibm", "ford", "xom"],
             "sp500", "dtb3"]      # Stocks, index and TB
    sPer= ""
    sBase= "capm"

    # Initialisation
    df= ReadCAPM(asData, sPer)

    # Output
    # Through trial and error: Format %.8g allows for 8 digits in total, some of which are placed before the dot...
    df.to_csv("data/"+sBase+sPer+".csv", float_format="%.8g")
    #print ("Published columns\n", df.columns, "\nto output ",
    #       "data/"+sBase+"_"+sPer+".csv",
    #       " and data/"+sBase+"_"+sPer+".xlsx")

###########################################################
### start main
if __name__ == "__main__":
    main()
