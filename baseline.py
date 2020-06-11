from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error


def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')


series = read_csv('GSPC.csv', header=0, parse_dates=True, squeeze=True, date_parser=parser, index_col='Date',
                  usecols=['Date', 'Close'], na_values=['nan'], nrows=100)

# Create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
print(dataframe.head(5))

# split into train and test sets
X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:, 0], train[:, 1]
test_X, test_y = test[:, 0], test[:, 1]
#
# persistence model
def model_persistence(x):
    return x

# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)

# plot predictions and expected results

pyplot.plot(X[:, 0], 'r--', label='Actual GSPC price')
pyplot.plot([None for i in train_y] + [x for x in predictions], 'k-', label='Predicted GSPC price')
pyplot.xlabel("Date")
pyplot.ylabel("Price")
pyplot.legend()
pyplot.show()