import pandas as pd

from projectwind.data import get_data
from projectwind.pipeline import get_pipeline


# 2 variables with the start and end of the data set
start_data ="2019-09-01 00:00:00"
end_data = "2021-07-01 00:00:00"
#Get the data with the get_data() function
def data_ml():
    data = get_data()
    #filter the first 4 month and the last 3 month of the dataset
    for index, df in data.items():
        data[index]=df[(df.index>=start_data) & (df.index<end_data)]
    return data

#add the missing rows function
def add_timestamps(data):
    results = {}
    for file in data.keys():
        df = data[file]
        ref_date_range = pd.date_range(start=start_data, end=end_data,freq='10T')
        ref_df = pd.DataFrame(index=ref_date_range)
        clean_data = df.reindex(ref_df.index)
        new_df = pd.merge(ref_df,clean_data,left_index=True, right_index=True,how='outer')
        results[file] = new_df
    return results

# This function gets the data and the pipeline, to then fit the data
def trainer():
    data =data_ml()
    data= add_timestamps(data)
    pipe =get_pipeline()
    ref_date_range = pd.date_range(start=start_data, end=end_data,freq='10T')
    for key, df in data.items():
        # Fit the date and set the index
        data[key] = pd.DataFrame(pipe.fit_transform(df), index=ref_date_range)
    return data

if __name__=='__main__':
    print(trainer()[0])
