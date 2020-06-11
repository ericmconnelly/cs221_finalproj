import warnings
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from sklearn.svm import SVR
from statsmodels.tsa.stattools import acf, pacf
from scipy.ndimage.interpolation import shift
from statsmodels.tsa.arima_model import ARIMA

import random

warnings.filterwarnings("ignore", category=DeprecationWarning)

PLOT_SHOW = True
PLOT_TYPE = False

NUM_TEST = 100
K = 50
NUM_ITERS = 10000

STOCKS = ['GSPC.csv', 'GOOG.csv', 'AAPL.csv', 'QCOM.csv', 'CMCSA.csv']

labels = ['Close', 'Open', 'High', 'Low']
likelihood_vect = np.empty([0, 1])
aic_vect = np.empty([0, 1])
bic_vect = np.empty([0, 1])

STATE_SPACE = range(2, 15)

def calc_mape(predicted_data, true_data):
    return np.divide(np.sum(np.divide(np.absolute(predicted_data - true_data), true_data), 0), true_data.shape[0])

def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')


def run_ets(series, H=100, alpha=0.5):
    result = [0, series[0]]
    for n in range(1, len(series)+H-1):
        if n >= len(series):
            result.append(alpha * series[-1] + (1 - alpha) * result[n])
        else:
            result.append(alpha * series[n] + (1 - alpha) * result[n])
    return result[len(series):len(series)+H]

for stock in STOCKS:
    dataset = read_csv('CMCSA.csv', header=0, parse_dates=True, squeeze=True, index_col='Date',
                  usecols=['Date', 'Close', 'Open', 'High', 'Low'], na_values=['nan'])

    dataframe = DataFrame(dataset.values)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    X = dataframe.values
    train_size = int(len(X) * 0.66)
    train, test = X[1:train_size], X[train_size:]

    lag_acf = acf(train, nlags=10)
    lag_pacf = pacf(train, nlags=10, method='ols')

    model = ARIMA(train, order=(2, 2, 2))
    results_AR = model.fit()

    ets_series = run_ets(train)

    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
    svr_rbf.fit(train)

    predicted_stock_data = np.empty([0, X.shape[1]])
    svr_stock_data = np.empty([0, X.shape[1]])
    arima_stock_data = np.empty([0, X.shape[1]])
    ets_stock_data = np.empty([0, X.shape[1]])
    baseline = np.empty([0, X.shape[1]])

    likelihood_vect = np.empty([0, 1])
    aic_vect = np.empty([0, 1])
    bic_vect = np.empty([0, 1])
    for states in STATE_SPACE:
        num_params = states ** 2 + states
        dirichlet_params_states = np.random.randint(1, 50, states)

        model = hmm.GaussianHMM(n_components=states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS)
        model.fit(test)
        if model.monitor_.iter == NUM_ITERS:
            sys.exit(1)
        likelihood_vect = np.vstack((likelihood_vect, model.score(dataset)))
        aic_vect = np.vstack((aic_vect, -2 * model.score(dataset) + 2 * num_params))
        bic_vect = np.vstack((bic_vect, -2 * model.score(dataset) + num_params * np.log(dataset.shape[0])))

    opt_states = np.argmin(bic_vect) + 2


    for idx in reversed(range(NUM_TEST)):
        train_dataset = X[idx + 1:, :]
        test_data = X[idx, :]
        num_examples = train_dataset.shape[0]
        if idx == NUM_TEST - 1:
            model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS,
                                    init_params='stmc')
        else:
            model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS,
                                    init_params='')
            model.transmat_ = transmat_retune_prior
            model.startprob_ = startprob_retune_prior
            model.means_ = means_retune_prior
            model.covars_ = covars_retune_prior

        model.fit(np.flipud(train_dataset))

        transmat_retune_prior = model.transmat_
        startprob_retune_prior = model.startprob_
        means_retune_prior = model.means_
        covars_retune_prior = model.covars_

        if model.monitor_.iter == NUM_ITERS:
            sys.exit(1)

        iters = 1
        past_likelihood = []
        curr_likelihood = model.score(np.flipud(train_dataset[0:K - 1, :]))
        while iters < num_examples / K - 1:
            past_likelihood = np.append(past_likelihood, model.score(np.flipud(train_dataset[iters:iters + K - 1, :])))
            iters = iters + 1
        likelihood_diff_idx = np.argmin(np.absolute(past_likelihood - curr_likelihood))
        predicted_change = train_dataset[likelihood_diff_idx, :] - train_dataset[likelihood_diff_idx + 1, :]
        predicted_stock_data = np.vstack((predicted_stock_data, X[idx + 1, :] + predicted_change))

        arima_predicted_change = results_AR.predict(start=idx, end=idx+1)
        ets_predicted_change = ets_series[idx] - ets_series[idx+1]
        svr_predicted_change = svr_rbf.predict(idx)

        svr_stock_data = np.vstack((svr_stock_data, X[idx + 1, :] + svr_predicted_change))
        arima_stock_data = np.vstack((arima_stock_data, X[idx + 1, :] + arima_predicted_change))
        ets_stock_data = np.vstack((ets_stock_data, X[idx + 1, :] + ets_predicted_change))


    np.savetxt('{}_forecast.csv'.format(stock), predicted_stock_data, delimiter=',', fmt='%.2f')

    mape = calc_mape(predicted_stock_data, np.flipud(X[range(100), :]))
    print('MAPE for the stock {} is '.format(stock), mape)

    if PLOT_TYPE:
        hdl_p = plt.plot(range(100), predicted_stock_data[:, 0])
        plt.title('Predicted stock prices')
        plt.legend(iter(hdl_p), ('Close'))
        plt.xlabel('Time steps')
        plt.ylabel('Price')
        plt.figure()
        hdl_a = plt.plot(range(100), np.flipud(X[range(100), 0]))
        plt.title('Actual stock prices')
        plt.legend(iter(hdl_p), ('Close'))
        plt.xlabel('Time steps')
        plt.ylabel('Price')
    else:
        for i in range(4):
            plt.figure()
            plt.plot(range(100), predicted_stock_data[:, i], 'k-', label='HMM Predicted ' + labels[i] + ' price')
            plt.plot(range(100), arima_stock_data[:, i], 'y--', label='ARIMA Predicted ' + labels[i] + ' price')
            plt.plot(range(100), ets_stock_data[:, i], 'g--', label='ETS Predicted ' + labels[i] + ' price')
            plt.plot(range(100), svr_stock_data[:, i], 'b--', label='SVR Predicted ' + labels[i] + ' price')
            plt.plot(range(100), np.flipud(X[range(100), i]), 'r--', label='Actual ' + labels[i] + ' price')
            plt.xlabel('Time steps')
            plt.ylabel('Price')
            plt.title(labels[i] + ' price' + ' for ' + stock[:-4])
            plt.grid(True)
            plt.legend(loc='upper left')

    plt.show()