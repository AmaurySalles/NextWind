import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import chain

from projectwind.data import get_WTG_data, get_WWO_API_data, get_MERRA2_data, get_ERA5_data

def make_datasets(num_datasets=25, forecast_data='MERRA2', frequency=None):
    """
    Fetches all data from a specified number of WTGs, alongside one of the forecast data (to be specified).
    Then runs through pre-processing (feature engineering & scaling) of data
    ------------
    Parameters
    num_datasets: 'int'
                  Number of WTG files to retrieve data from. Max (default) is 25
    forecast_data: 'str', 'list'
                    Specify which forecast data source to use. Options are: [
                    - 'WWO': World Weather Online API
                    - 'MERRA2': MERRA-2 historical meso-scale dataset
                    - 'ERA5': ERA-5 historical meso-scale dataset
    period: 'str'
            Specify the frequency of the output dataframe (e.g. '1H', '10T', etc.)
            Default 'None' outputs hourly indices
    ------------
    Returns:
    Tuple of train, validation and test (3) DataFrame
    """
    # Fetch WTG datasets
    data = get_WTG_data(num_datasets)
    
    # Fetch forecast data
    if forecast_data == 'WWO':
        weather = get_WWO_API_data(frequency='1H')
        weather = WWO_feature_engineering(weather)
    elif forecast_data == 'MERRA2':
        weather = get_MERRA2_data()
        weather = MERRA2_ERA5_feature_engineering(weather)
    elif forecast_data == 'ERA5':
        weather = get_ERA5_data()
        weather = MERRA2_ERA5_feature_engineering(weather)
    
    # Data pre-processing
    print('### Preparing train, val & test datasets ###')
    train_df, val_df, test_df = list(), list(), list()
    
    for WTG_data in data:

        # Feature engineering
        WTG_data = WTG_feature_engineering(WTG_data)

        # Join with weather data
        WTG_data = pd.concat([WTG_data, weather], axis=1)
        # Slice off additional API data timestamps
        WTG_data.dropna(axis=0, inplace=True)
     
        # Resampling to smooth out curves
        if frequency is not None:
            WTG_data = WTG_data.resample(frequency).mean()

        # Split datasets - taking last few months of each WTG into val & test sets
        n = len(WTG_data)
        train_df.append(WTG_data[0:int(n*0.7)])
        val_df.append(WTG_data[int(n*0.7):int(n*0.9)])
        test_df.append(WTG_data[int(n*0.9):])

    # Scale datasets
    train_df, val_df, test_df = min_max_scale_data(train_df, val_df, test_df)
    datasets = dict(train=train_df, val=val_df, test=test_df)

    return datasets

def std_scale_data(train_df, val_df, test_df):
    """
    Applies standard scaling to each column (according on training dataset) to all datasets.
    Ignores scaling target ('Power') and directional vectors.
    ------------
    Returns:
    Tuple of scaled train, validation and test (3) DataFrame
    """
    # Apply scaling to all three datasets
    pd.options.mode.chained_assignment = None
    column_names = train_df[0].columns
    for WTGs in range(len(train_df)):
        for col in column_names:
            if ('Target' in col) or ('Power' in col) or ('_' in col): # _ represents the X & Y direction vectors (already between 1 and -1)
                print(col)
                pass
            else:
                col_std =   train_df[WTGs][col].std()
                col_mean =  train_df[WTGs][col].mean()
                # Scale each columns of each dataset
                train_df[WTGs].loc[:,col] = train_df[WTGs][col].apply(lambda x: (x - col_mean) / col_std)
                val_df[WTGs].loc[:,col] = val_df[WTGs][col].apply(lambda x: (x - col_mean) / col_std)
                test_df[WTGs].loc[:,col] = test_df[WTGs][col].apply(lambda x: (x - col_mean) / col_std)
    pd.options.mode.chained_assignment = 'warn'
    return train_df, val_df, test_df

def min_max_scale_data(train_df, val_df, test_df):
    """
    Applies min-max scaling to each column (according on training dataset) to all datasets.
    Ignores scaling target ('Power') and directional vectors.
    ------------
    Returns:
    Tuple of scaled train, validation and test (3) DataFrame
    """
    # Find min / max of each category across all 25 WTGs (from train set only to avoid data leakage)
    scaling_data = pd.DataFrame(index=['min','max'], columns=train_df[0].columns, data=0)
    for WTG_data in train_df:
        for col in WTG_data:
            temp_min = np.min([scaling_data.loc['min', col], WTG_data[col].min(axis=0)])
            temp_max = np.max([scaling_data.loc['max', col], WTG_data[col].max(axis=0)])
            scaling_data.loc['min', col] = temp_min
            scaling_data.loc['max', col] = temp_max

    # Apply scaling to all three datasets
    pd.options.mode.chained_assignment = None
    column_names = train_df[0].columns
    for WTGs in range(len(train_df)):
        for col in column_names:
            if ('Target' in col) or ('Power' in col) or ('_' in col): # _ represents the X & Y direction vectors (already between 1 and -1)
                pass
            else:
                col_min = scaling_data.loc['min',col]
                col_max = scaling_data.loc['max',col]
                # Scale each columns of each dataset
                train_df[WTGs].loc[:,col] = train_df[WTGs][col].apply(lambda x: (x - col_min) / (col_max - col_min))
                val_df[WTGs].loc[:,col] = val_df[WTGs][col].apply(lambda x: (x - col_min) / (col_max - col_min))
                test_df[WTGs].loc[:,col] = test_df[WTGs][col].apply(lambda x: (x - col_min) / (col_max - col_min))
    pd.options.mode.chained_assignment = 'warn'
    return train_df, val_df, test_df

def WTG_feature_engineering(df):
    """
    Performs feature engineering on WTG data
    Involves transforming directional degrees (0 to 360) into directional vectors (X and Y).
    Removes the input features which have been transformed.
    ------------
    Returns:
    Input DataFrame with additional features
    """
    # Find wind direction (by correcting nacelle orientation with misalignment)
    df['Misalignment'] = df['Misalignment']* np.pi / 180 # Transform into radians
    df['Nacelle Orientation'] = df['Nacelle Orientation'] * np.pi / 180 # Transform into radians
    df['Wind Direction'] =  df['Nacelle Orientation'] - df['Misalignment']

    # Build vectors for nacelle orientation
    df['Nacelle_X'] = np.cos(df['Nacelle Orientation'])
    df['Nacelle_Y'] = np.sin(df['Nacelle Orientation'])

    # Build vectors from wind direction
    df['Wind_X'] = np.cos(df['Wind Direction']) 
    df['Wind_Y'] = np.sin(df['Wind Direction'])  

    # Remove superseeded columns, except wind speed
    df.drop(columns=['Misalignment','Nacelle Orientation', 'Wind Direction'], inplace=True)

    return df

def WWO_feature_engineering(df):
    """
    Performs feature engineering on WWO weather data
    Involves transforming directional degrees (0 to 360) into directional vectors (X and Y).
    Removes the input features which have been transformed.
    ------------
    Returns:
    Input DataFrame with additional features
    """
    df['windSpeed_API'] = df['windspeedKmph'] * 1000 / (60*60) # Transform into m/s
    df['windGust_API'] = df['WindGustKmph'] * 1000 / (60*60) # Transform into m/s
    df['windDirDegree'] = df['winddirDegree']* np.pi / 180 # Transform into radians

    # Build vectors from wind direction and wind speed
    df['Wind_API_X'] = df['windSpeed_API'] * np.cos(df['windDirDegree'])
    df['Wind_API_Y'] = df['windSpeed_API'] * np.sin(df['windDirDegree'])

    # Build vectors from wind direction and wind gust
    df['WindGust_API_X'] = df['windGust_API'] * np.cos(df['windDirDegree'])
    df['WindGust_API_Y'] = df['windGust_API'] * np.sin(df['windDirDegree'])

    # Remove superseeded columns, except wind speed
    df.drop(columns=['windDirDegree', 'windspeedKmph', 'WindGustKmph', 'winddirDegree'], inplace=True)

    return df

def MERRA2_ERA5_feature_engineering(df):
    """
    Performs feature engineering on WWO weather data
    Involves transforming directional degrees (0 to 360) into directional vectors (X and Y).
    Removes the input features which have been transformed.
    ------------
    Returns:
    Input DataFrame with additional features
    """
    
    # Build vectors from wind direction forecast
    df['Forecast_wind_direction'] = df['Forecast_wind_direction'] * np.pi / 180 # Transform into radians
    df['Forecast_X'] = np.cos(df['Forecast_wind_direction'])
    df['Forecast_Y'] = np.sin(df['Forecast_wind_direction'])
    
    # Remove superseeded column
    df.drop(columns=['Forecast_wind_direction'], inplace=True)
    
    return df

class SequenceGenerator():
    """
    Class used to generate sequences according to time window and columns specified.
    Firstly create the window, then create the train, validation and test sequences.
    ------------
    Parameters:
    input_width: 'int' 
                 Time-span of input features

    label_width: 'int' 
                 Time-span of label (and forecast) features

    shift: 'int' 
           Shift of the label window compared to input starting point (0)

    datasets: 'dict' 
            Dictionnary with keys `train`, `val` & `test`, each containing
            a lists of DataFrames (one per WTG loaded)

    classifications: 'bool'
                    If changed to True, target will be summed over the label_width and 
                    fitted into a bin according to default target ('power') quartiles.
                    Default: False.

    input_columns: 'str', 'list'
                    Specified column names to use as input features. 
                    Default `None` will use all dataset columns as inputs

    forecast_columns: 'str', 'list'
                      Specified column names to use as forecaste features. 
                      Default `None` will not use any forecasting features
                      
    label_columns: 'str', 'list' 
                    Specified column names to use as label. 
                    Default: `Power'.
    -------------
    Returns:
    Window of specified input.
    Allows retrieval of sequences through the class variables `train`, `val`, `test` 
    """
    
    def __init__(self, input_width, label_width, shift,
                 datasets=None, classification=False,
                 input_columns=None, forecast_columns=None, label_columns=['Power']):

        ### Work out sequence window parameters ###
        # Given parameters
        self.classification = classification
        self.input_width = input_width
        self.forecast_width = label_width
        self.label_width = label_width
        self.shift = shift
        
        # Work out window timestep slices
        self.total_window_size = input_width + shift
        # Inputs
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        # Forecast
        self.forecast_start = self.total_window_size - self.forecast_width
        self.forecast_slice = slice(self.forecast_start, None)
        self.forecast_indices = np.arange(self.total_window_size)[self.forecast_slice]
        # Label
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
         
        
        ### Work out which features belong in each categories (input/forecast/label) ###
        # If data example has not been specified, use default complete dataset parameters 
        # Else workout all features from the dataset
        if datasets is None:
            self.columns = self.input_columns = ['Power', 'Rotor Speed', 'Wind Speed', 'Blade Pitch', 
                                  'Nacelle_X', 'Nacelle_Y', 'Wind_X', 'Wind_Y', 
                                  'Forecast_wind_speed', 'Forecast_X','Forecast_Y']  
        else:
            self.columns = datasets['train'][0].columns
        self.column_indices = {name: i for i, name in enumerate(self.columns)}       

        # If input colums have been specified, find the input column indices. 
        # Else, select all default preproc columns.
        self.input_columns = input_columns
        if self.input_columns is None:
            self.input_columns = ['Power', 'Rotor Speed', 'Wind Speed', 'Blade Pitch', 
                                  'Nacelle_X', 'Nacelle_Y', 'Wind_X', 'Wind_Y', 
                                  'Forecast_wind_speed', 'Forecast_X','Forecast_Y']
        self.input_columns_indices = {name: i for i, name in enumerate(self.input_columns)}

        # # If another label colum than default ('Power') has been specified, find the label column index.
        self.label_columns = label_columns
        self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

        # If forecast colums have been specified, find the forecast column indices.
        # Else, no forecast to be created
        self.forecast_columns = forecast_columns
        if self.forecast_columns is not None:
            self.forecast_columns_indices = {name: i for i, name in enumerate(forecast_columns)}

        # Print window specifications
        print('### Window details ### \n',repr(self), '\n')

        ## Initialise sequences
        if datasets is not None:
            self.train, self.val, self.test = self.get_sequences(datasets)

    def __repr__(self):
        
        input_details = '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input column name(s): {self.input_columns}', 
            f'Input indices: {self.input_indices}'])

        label_details = '\n'.join([  
            f'Label column name(s): {self.label_columns}', 
            f'Label indices: {self.label_indices}'])

        if self.forecast_columns is not None:
            forecast_details = '\n'.join([f'Forecast column name(s): {self.forecast_columns}',            
                                          f'Forecast indices: {self.forecast_indices}'])
            _repr = input_details + '\n' + forecast_details + '\n' + label_details
        else:
            _repr = input_details + '\n' + label_details
        
        return _repr

    def get_sequences(self, datasets=None):
        # If not providing a dataset (train/val/test), try to load it from memory
        # Else, update train/val/test with dataset provided
        try:
            return self.load_sequences(datasets.keys())
        except FileNotFoundError:
            if datasets is not None:
                for set, data in datasets.items():
                    print(f'### Generating {set} sequences ###')
                    sequences = self.generate_sequences(data)
                    self.save_sequences(sequences, set)
                    if set == 'train':
                        self.train = sequences
                    elif set == 'val':
                        self.val = sequences
                    else:
                        self.test= sequences
                return self.train, self.val, self.test
            else:
                return print("Could not find any sequences to load - Provide dataset to generate sequences.")

    def generate_sequences(self, data):

        X_datasets, y_datasets, X_fc_datasets  = list(), list(), list()
        energy_datasets = list()
        sequences = dict()

        for WTG_data in data:

            # Find sequences according to window size of X (X_forecast) and y
            WTG_data = np.array(WTG_data, dtype=np.float32)
            WTG_sequences = tf.keras.utils.timeseries_dataset_from_array(data=WTG_data,
                                                                        targets=None,
                                                                        sequence_length=self.total_window_size,
                                                                        sampling_rate=1,
                                                                        sequence_stride=self.total_window_size,
                                                                        shuffle=False,
                                                                        batch_size=32)
            # Split X, (X_forecast) and y according to window size
            WTG_sequences = WTG_sequences.map(self.split_windows)

            # Transfer from tensor to numpy array to save under .NPY format
            if self.forecast_columns is None:
                X_datasets.append(chain.from_iterable([X.numpy() for X, y in WTG_sequences]))
                y_datasets.append(chain.from_iterable([y.numpy() for X, y in WTG_sequences]))
            else:
                X_datasets.append(chain.from_iterable([X.numpy() for X, X_fc, y in WTG_sequences]))
                X_fc_datasets.append(chain.from_iterable([X_fc.numpy() for X, X_fc, y in WTG_sequences]))
                y_datasets.append(chain.from_iterable([y.numpy() for X, X_fc, y in WTG_sequences]))
            
            # If a classification problem, sum target power values into a single energy value
            if self.classification == True:
                seq_energy = self.classify_target(y_datasets)
                energy_datasets.append(seq_energy)

        # Aggregate batches into one array (batch generator done through fit function)
        X_array = np.array(list(chain.from_iterable(X_datasets)))
        if self.classification == True:
            y_array = np.array(list(chain.from_iterable(energy_datasets)))
        else:
            y_array = np.array(list(chain.from_iterable(y_datasets)))
        
        # Shuffle sequences
        X_array = self.shuffle_sequences(X_array)
        y_array = self.shuffle_sequences(y_array)

        # Add to dict output
        sequences['X'] = X_array
        sequences['y'] = y_array
        
        # Perform same tasks for forecast & append to output
        if self.forecast_columns is not None:
            # Aggregate batches into one array (batch generator done through fit function)
            X_fc_array = np.array(list(chain.from_iterable(X_fc_datasets)))
            
            # Shuffle sequences
            X_fc_array = self.shuffle_sequences(X_fc_array)

            sequences['X_fc'] = X_fc_array
        
        return sequences

    def split_windows(self, sequences):
        
        # Splice correct timestamps
        inputs = sequences[:, self.input_slice, :]
        forecast = sequences[:, self.forecast_slice, :]
        labels = sequences[:, self.labels_slice, :]
        
        # If input, forecast & labels are specified, select requested columns
        if self.input_columns is not None:
            inputs = tf.stack([inputs[:,:, self.column_indices[name]] for name in self.input_columns], axis=-1)
        inputs.set_shape([None, self.input_width, None])
            
        if self.label_columns is not None:
            labels = tf.stack([labels[:,:, self.column_indices[name]] for name in self.label_columns], axis=-1)
        labels.set_shape([None, self.label_width, None])

        # Forecast
        if self.forecast_columns is None:
            return inputs, labels
        else:
            forecast = sequences[:, self.forecast_slice, :]
            forecast = tf.stack([forecast[:,:, self.column_indices[name]] for name in self.forecast_columns], axis=-1)
            forecast.set_shape([None, self.forecast_width, None])
            return inputs, forecast, labels

    def classify_target(self, target_list):
        for batch in target_list:
                seq_energy = []
                for seq in batch:
                    total = 0
                    for i in seq:
                        total += i[0]
                    seq_energy.append(total)
        return seq_energy

    def shuffle_sequences(self, data, seed=42):
        np.random.seed(seed)
        np.random.shuffle(data)
        return data
    
    def save_sequences(self, sequences, dataset_name):
         # Create sequence name
        if self.classification is True:
            sequence_name = 'Class_'
        else:
            sequence_name = 'Linear_'
        sequence_name += f"{self.input_width}in_{self.label_width}out"

        # Save sequences
        np.save(f'./projectwind/data/Sequences_{sequence_name}_X_{dataset_name}.npy', np.asanyarray(sequences['X'], dtype=float))
        np.save(f'./projectwind/data/Sequences_{sequence_name}_y_{dataset_name}.npy', np.asanyarray(sequences['y'], dtype=float))
        if self.forecast_columns is not None:
            np.save(f'./projectwind/data/Sequences_{sequence_name}_X_fc_{dataset_name}.npy', np.asanyarray(sequences['X_fc'], dtype=float))

        return print(f"### {dataset_name.capitalize()} sequences saved under './projectwind/data/Sequences__{sequence_name}_<X/X_fc/y>_{dataset_name}.npy")

    def load_sequences(self, dataset_names):
        
        for name in dataset_names:
            
            sequences = dict()
            
            # Create sequence name
            if self.classification is True:
                sequence_name = 'Class_'
            else:
                sequence_name = 'Linear_'
            sequence_name += f"{self.input_width}in_{self.label_width}out"
    
            # Load sequences
            sequences['X'] = np.load(f'./projectwind/data/Sequences_{sequence_name}_X_{name}.npy', allow_pickle=True)
            sequences['y'] = np.load(f'./projectwind/data/Sequences_{sequence_name}_y_{name}.npy', allow_pickle=True)
            if self.forecast_columns is not None:
                sequences['X_fc'] = np.load(f'./projectwind/data/Sequences_{sequence_name}_X_fc_{name}.npy', allow_pickle=True)
            
            print(f'### {name.capitalize()} {sequence_name} sequences loaded ###')

            # Update self
            if name == 'train':
                self.train = sequences
            elif name == 'val':
                self.val = sequences
            else:
                self.test= sequences

        return self.train, self.val, self.test


