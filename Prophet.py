from _init_ import *
from prophet import Prophet
from sklearn.metrics import mean_squared_error

ph_date = data.reset_index()['Date']
ph_train = pd.DataFrame({'ds':ph_date.iloc[0:len(train_y)], 'y':train_y})
ph_test = pd.DataFrame({'ds':ph_date.iloc[len(train_y):], 'y':test_y})
ph_model = Prophet()
ph_model.fit(ph_train)


forecast = ph_model.predict(ph_test)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
loss_ph = mean_squared_error(ph_test['y'], forecast['yhat'])
print(f"MSE Loss of Prophet is {np.sqrt(loss_ph)}")

fig1 = ph_model.plot(forecast)