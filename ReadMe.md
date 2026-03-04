# 👾 Bitcoin's Volatility Prediction - Nowcasting & Forecasting 
#### **🚨 Try out the deployed model [here](https://huggingface.co/spaces/Lullooo/BTCVolatility-forecasting)**  
<br/>  

### 📄 MODEL's OUTLINE
The model nowcasts (same-day) and forecasts (next-day) BTC/USDT volatility based on BTC price data (OHLCV), the daily fear & greed index and Google trend data for the word "Bitcoin". In details, the model provides two different estimations:

- 🎯 **Point prediction**: through an XGBoost model, it yields a precise estimation for BTC volatility. This is how most forecasting models behave.
- 🔺🔻 **Interval predictions**: through a NGBoost model, it higlights the upper and lower bounds for BTC volatility with a 95% confidence interval by computing its probability distribution.

It is enough to input the date (YYYY-MM-DD) of interest, and the model 



### 🧐 WORKING EXAMPLE
GIF WHILE USING MODEL
