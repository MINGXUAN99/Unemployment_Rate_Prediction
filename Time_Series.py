import matplotlib.pyplot as plt
import pm

from _init_ import *
from sklearn.metrics import mean_squared_error


arima = pm.auto_arima(train_y, error_action='ignore', trace=True,
                      suppress_warnings=True, maxiter=5,
                      seasonal=True, m=12)


x_arima = np.arange(test_y.shape[0])
y_arima = arima.predict(n_periods=test_y.shape[0])
loss_arima = mean_squared_error(x_arima, y_arima)
plt.scatter(x_arima, test_y, marker='x')
plt.plot(x_arima, y_arima)
plt.title('Actual test samples vs. forecasts')
plt.show()
print(f"RMSE Loss of Arima is {np.sqrt(loss_arima)}")
