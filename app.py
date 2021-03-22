#!/usr/bin/python
# -*- coding: UTF-8 -*-
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
    import pandas as pd
    import numpy as np
    from pandas import read_csv
    from datetime import datetime
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    from pandas import DataFrame
    from pandas import concat
    from sklearn.metrics import mean_squared_error
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from numpy import concatenate
    from keras.layers import LSTM
    from keras.layers import Bidirectional
    from math import sqrt

    import warnings                                  # do not disturbe mode
    warnings.filterwarnings('ignore')

    # Load packages
    import statsmodels.formula.api as smf            # statistics and econometrics
    import statsmodels.tsa.api as smt
    import statsmodels.api as sm
    import scipy.stats as scs

    from itertools import product                    # some useful functions
    from tqdm import tqdm_notebook

    # Importing everything from forecasting quality metrics
    from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
    from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error


    data_all = read_csv(args.training, encoding='utf-8', header=0, index_col=0)

    start = datetime(2020, 3, 21)
    end = datetime(2021, 3, 21)
    rng = pd.date_range(start, end)
    target_data = pd.Series(data_all.values[:, 0], index=rng)

    p, q, P, Q = 1, 1, 0, 1
    d = 1
    D = 1
    s = 7

    best_model=sm.tsa.statespace.SARIMAX(target_data, order=(p, d, q), 
                                            seasonal_order=(P, D, Q, s)).fit(disp=-1)
    print(best_model.summary())
    
    n_steps = 8
    forecast = best_model.predict(start = target_data.shape[0], end = target_data.shape[0]+n_steps)
    
    output = []
    for item in forecast[1:-1]:
        output.append(int(round(item*10)))

    date = ['20210323', '20210324', '20210325', '20210326', '20210327', '20210328', '20210329']
    dictionary = {'date':date, 'operating_reserve(MW)':output}
    df = pd.DataFrame(dictionary)
    df.to_csv(args.output, index = None)