import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm, model_selection, metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
# import get_stock_data as gs
# from finta import TA

from matplotlib import style
style.use('ggplot')
year = 252

# Input control data
ticker = 'AAPL'
label = "adjusted_close"
lookahead = 5
forecast_period = 100
train_period = 10 * year
test_period = 1 * year
verbose = 1
do_plot = 1
do_scale = 1

# Get Stock data
try:
	# df = gs.get_stock_daily_adjusted(ticker, outputsize='full')
	df = pd.read_csv(f'{ticker}_daily.csv', index_col='timestamp')
	df = df.sort_index(ascending=True)
except FileNotFoundError:
	print(f"Ticker data for '{ticker}' not found!")
	exit()

if verbose > 1:
	print(df.head())

# Add various technical indicators
col1 = label
col2 = "volume"
dfreg = df.loc[:, [col1, col2]]
dfreg['HL_PCT'] = (df['high'] - df['low']) / df['close'] * 100.0
dfreg['HC_PCT'] = (df['high'] - df['close']) / df['close'] * 100.0
dfreg['PCT_ci'] = (df['close'] - df['open']) / df['open'] * 100.0
dfreg['PCT_c1'] = df[col1].pct_change(periods=1) * 100.0
dfreg['PCT_c5'] = df[col1].pct_change(periods=5) * 100.0
dfreg['PCT_c20'] = df[col1].pct_change(periods=20) * 100.0
# dfreg['RSI-9'] = TA.RSI(df, period=9) / 100
# dfreg['RSI-20'] = TA.RSI(df, period=20) / 100
# dfreg['EMA-9'] = TA.EMA(df, column=col1, period=9)
# dfreg['EMA-20'] = TA.EMA(df, column=col1, period=20)
# dfreg['EMA-d'] = (TA.EMA(df, column=col1, period=9) - TA.EMA(df, column=col1, period=20)) / TA.EMA(df, column=col1, period=9) * 100.0

# Limit data to scope
dfreg = dfreg.iloc[-forecast_period-test_period-train_period:]
if verbose > 1:
	print(dfreg.shape)

# Separating the label here, we want to predict the "adjusted_close"
dfreg['label'] = dfreg[label].shift(-lookahead)
# dfreg['label'] = df[label].pct_change(periods=lookahead).shift(-lookahead)

# Drop missing value
dfreg.dropna(inplace=True)

# Make X
X = dfreg.drop(['label'], 1)
if verbose > 1:
	print(X.tail())

if verbose > 1:
	print(dfreg.head(6))
	print(dfreg.tail(6))

# Scale the X so that everyone can have the same distribution for linear regression
if do_scale:
	X = StandardScaler().fit_transform(X)
else:
	X = np.array(X)
	print("OBS: No pre-scaling of input data!!!")
if verbose > 2:
	print(X)

if verbose > 1:
	print(dfreg.head(6))
	print(dfreg.tail(6))

# quit() 

# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_period:]
X = X[:-forecast_period]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y_lately = y[-forecast_period:]
y = y[:-forecast_period]

if verbose > 1:
	print('Dimension of X', X.shape)
	print('Dimension of y', y.shape)

# Separation of training and testing of model by cross validation train test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# X_train = X[:-test_period]; y_train = y[:-test_period]; X_test = X[-test_period:]; y_test = y[-test_period:]

if verbose > 1:
	print('Dimension of X_train', X_train.shape)
	print('Dimension of y_train', y_train.shape)
	print('Dimension of X_test', X_test.shape)
	print('Dimension of y_test', y_test.shape)

models = {}

# Linear regression
name = 'LinearRegression'
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
models[name] = clf

# BayesianRidge
name = 'BayesianRidge'
clf = BayesianRidge()
clf.fit(X_train, y_train)
models[name] = clf

# Ridge
for a in [1e-3, 1e-2, 1e-1, 1, 10, 100]:
	name = f'Ridge-alph-{a:.0E}'
	clf = Ridge(alpha=a)
	clf.fit(X_train, y_train)
	models[name] = clf

# Lasso
for a in [1e-3, 1e-2, 1e-1, 1]:
	name = f'Lasso-alph-{a:.0E}'
	clf = Lasso(alpha=a)
	clf.fit(X_train, y_train)
	models[name] = clf

# SGDRegressor
name = 'SGDRegressor'
clf = SGDRegressor()
clf.fit(X_train, y_train)
models[name] = clf

# Quadratic Regression
for n in [2, 3]:
	name = f'QuadraticRegr-{n}'
	clf = make_pipeline(PolynomialFeatures(n), Ridge())
	clf.fit(X_train, y_train)
	models[name] = clf

# KNN Regression
for nn in [3, 5, 7, 9]:
	name = f'KNNRegression-{nn}'
	clf = KNeighborsRegressor(n_neighbors=nn)
	clf.fit(X_train, y_train)
	models[name] = clf

# Add SVR models
for knl in ['linear', 'rbf', 'sigmoid', 'poly']:
	for Cv in [1, 2, 3]:
		name = f'SVM-{knl}-C{Cv}'
		clf = svm.SVR(C=Cv, kernel=knl, gamma='scale')
		clf.fit(X_train, y_train)
		models[name] = clf

# Evaluate the models
print('-'*80)
last_score = -1000
best = ''
for name, clf in models.items():
	score = clf.score(X_test, y_test)
	# score = r2_score(y_lately, clf.predict(X_lately))
	# score = 1/mean_squared_error(y_lately, clf.predict(X_lately))
	if verbose > 0:
		y_pred = clf.predict(X_lately)
		cnf = clf.score(X_test, y_test)
		mse = mean_squared_error(y_lately, y_pred)
		var = r2_score(y_lately, y_pred)
		print(f'Confidence of {name} \t: {cnf:.6f},\tMSE: {mse:.2f},\tVAR: {var:.4f}')
		# print(clf.coef_)
	if score > last_score:
		last_score = score
		best = name

# Print result
print('-'*80)
print(f'Best performing model: {best}!!!')
print('-'*80)

# Calculate forecast
forecast_set = models[best].predict(X_lately)
dfc = pd.DataFrame(data=forecast_set, index=dfreg.index[-forecast_period:])
dfreg['Forecast'] = dfc

# Printing the forecast
if do_plot:
	# Plot the forecast
	dfreg['label'].tail(3*forecast_period).plot()
	dfreg['Forecast'].tail(3*forecast_period).plot()
	plt.legend(loc=4)
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title(f'{ticker} - Forecast')
	plt.show()
