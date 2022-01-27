import pandas as pd
import numpy as np
import os

def concatenate_df():
    # Take the parent dirname for the raw data
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
    rawdir = os.path.join(parentdir,"raw_data")

    # init global DF
    globalDF = pd.read_csv(os.path.join(rawdir,"A01.csv"))
    globalDF["Turbine_id"] = 1

    # loop over the CSVs
    for i in range(2,26):
    # CSV file name
        if i <10:
            file="A0"+str(i)+".csv"
        else:
            file="A"+str(i)+".csv"
        df = pd.read_csv(os.path.join(rawdir,file))
        # add an ID for each turbine
        df["Turbine_id"] = i

        #concatenate df
        globalDF = pd.concat((globalDF,df),ignore_index=True)
    return globalDF

if __name__ =="__main__":
    df =concatenate_df()
    print(df.head())
