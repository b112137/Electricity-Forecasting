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

    data_all = read_csv(args.training, encoding='utf-8', header=0, index_col=0)

    values = data_all.values    
    values = values.astype('float32')
    reframed = series_to_supervised(values, 1, 8)
    reframed.drop(reframed.columns[   np.concatenate(( list(range(8,14)), list(range(15,21)), list(range(22,28)),\
                                                    list(range(29,35)), list(range(36,42)), list(range(43,49)),\
                                                    list(range(50,56)), list(range(57,63)) ))   ]   , axis=1, inplace=True)
    print(reframed.head())

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(reframed)
    values = scaled

    n_train_hours = 358
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    train_X, train_y = train[:, :-8], train[:,-8:]
    test_X, test_y = test[:, :-8], test[:, -8:]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    model = Sequential()
    model.add(LSTM(50, activation='relu',input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(8))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=300, batch_size=8,  verbose=2, shuffle=False)

    temp = scaler.transform([np.concatenate( ( list(data_all.iloc[-1]), list(range(0,8)) ) )])
    test_X = np.array([[  temp[0][:-8] ]])
    # 开始预测
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # 预测值反转缩放
    inv_yhat = concatenate((test_X[:, 0:], yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,-7:]
    print(inv_yhat)

    output = []
    for item in inv_yhat[0]:
        output.append(int(round(item*10)))

    date = ['20210323', '20210324', '20210325', '20210326', '20210327', '20210328', '20210329']
    dictionary = {'date':date, 'operating_reserve(MW)':output}
    df = pd.DataFrame(dictionary)
    df.to_csv(args.output, index = None)