import matplotlib.pyplot as plt

from _init_ import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class lr_model():
    def __init__(self):
        self.model = LinearRegression()
    
    def train(self, x_train , y_train):
        self.model.fit(x_train, y_train)
        print("Linear Regression model training finished.")

    def predict(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        print('RMSE of test is =', np.sqrt(mean_squared_error(y_test,y_pred)))
        return y_pred

    def plot(self, y_test, y_pred):
        plt.scatter(range(len(y_test)), y_test, marker='x', c = 'red')
        plt.plot(range(len(y_pred)), y_pred)
        plt.title('Actual test samples vs. forecasts')
        plt.show()

lr = lr_model()
lr.train(train_x,train_y)
pred_y = lr.predict(test_x, test_y)
lr.plot(test_y, pred_y)


