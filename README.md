# Unemployment_Rate_Prediction

This work is based on course project in IE534 Fall 2022 in UIUC.

## Github Address

https://github.com/MINGXUAN99/Unemployment_Rate_Prediction

## Data 
### Source
The data for this project comes from the [Local Area Unemployment Statistics](https://data.ca.gov/dataset/local-area-unemployment-statistics-laus/resource/b4bc4656-7866-420f-8d87-4eda4c9996ed) (LAUS) dataset public in California Open Data Portal. 

## Methodologies
In the preprocessing step, I first converted the date datatype to Pandas timesteps, which is easier for time difference and other manipulations. With some descriptive data, I chose to work on the data where over 90% samples have the same value in specific columns (i.e. "Seasonally Adjusted" = "N", "Status" = "Final"). 

To make prediction, I first train a linear regression model as a benchmark

Then, I trained some time series model for higher prediction accuracy. Due to the feature of time series model, I further process the dataset with only "Date" and "Unemployment_Rate" columns left. Here I used ARIMA and Prophet model to forecast the future unemployment rate.

Finally, I also implemented LSTM model to make predictions. LSTM is a special type of RNN model and not restrcited to time series analysis. Although more general and usually more powerful, LSTM model required more care in tuning hyperparameters. I chose approriate number of LSTM cells to balance the tradeoff between low accuracy and overfitting.




