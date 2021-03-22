#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 将数据转换成监督学习问题
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # 输入序列(t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # 预测序列(t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # 把所有放在一起
    agg = concat(cols, axis=1)
    agg.columns = names
    # 删除空值行
    if dropnan:
        agg.dropna(inplace=True)
    return agg


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