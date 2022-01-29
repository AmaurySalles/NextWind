def subsample_sequence(results, day_length, number_of_subsamples, acceptable_level_of_missing_values = 0.1):
    """
    Given the initial dictionnary of dataframes `results`, return a list of dataframes of length `length`.
    """
    length = day_length*6*24
    last_possible = df.shape[0] - length
    random_start = np.random.randint(0, last_possible)

    subsamples = []

    while len(subsamples) < number_of_subsamples :
        random_file_number = np.random.randint(1, 26)

        if random_file_number < 10:
            file_name = f'A0{random_file_number}.csv'
        else :
            file_name = f'A{random_file_number}.csv'
        df_sample = results[file_name][random_start: random_start+int(length)]

        if (df_sample.isna().sum()/len(df_sample))[0] < acceptable_level_of_missing_values :
            subsamples.append(df_sample)
            # $CHALLENGIFY_END
    return subsamples

def split_subsample_sequence(results, day_length, number_of_sumbsamples, acceptable_level_of_missing_values = 0.1):
    '''Create one single random (X,y) pair'''
    # $CHALLENGIFY_BEGIN
    length = day_length*6*24
    subsamples = subsample_sequence(results, day_length, number_of_sumbsamples, acceptable_level_of_missing_values = 0.1)
    Y = []
    X = []
    for sample in subsamples:
        y_sample = sample[['Media de Potencia Activa 10M\n(kW)']].iloc[sample.shape[0]-72:]
        Y.append(y_sample)

        x_sample = sample[0:int(length) -72]
        X.append(x_sample)
    # $CHALLENGIFY_END
    return np.array(X), np.array(Y)
