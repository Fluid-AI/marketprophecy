# Predict the Market - Fluid AI Test

## Approach
I have used *tensorflows and keras* **LSTM** model to predict the NIFTY index for next 40 days.
The prediction is based on the closing value of the stock of each day.
To get independent variables, I have used last 50 days value of the given stock with the help of hyper-parameter tuning. (100 day for stock might be inappropriate as market is very volatile.)
Predicted 51st day's value with LSTM.
Used graphics to visualise the graphs(EDA).

## Technology Used
[Keras](https://keras.io/), [Tensorflow](https://www.tensorflow.org/), [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/), [matplotlib](https://matplotlib.org/), ['seaborn'](https://seaborn.pydata.org/)
