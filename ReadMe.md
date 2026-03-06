# 👾 Bitcoin's Volatility Prediction - Nowcasting & Forecasting 
#### **🚨 Try out the deployed model [here](https://huggingface.co/spaces/Lullooo/BTCVolatility-forecasting)**  
<br/>  

## 📄 MODEL's OUTLINE
A detailed project and model description can be found in the [project_description 📁](https://github.com/Lulloooo/BTCvolatility-nowcasting-Forecasting/blob/main/Project_description.txt)  
<br/>
The model nowcasts (same-day) and forecasts (next-day) 5-days EWMA BTC/USDT volatility based on BTC price data (OHLCV), the daily fear & greed index and Google trend data for the word "Bitcoin". In details, the model provides two different estimations:

- 🎯 **Point prediction**: through an XGBoost model, it yields a precise estimation for BTC volatility. This is how most forecasting models behave.
- 🔺🔻 **Interval predictions**: through a NGBoost model, it higlights the upper and lower bounds for BTC volatility with a 95% confidence interval by computing its probability distribution.

**⁉️ TARGET VARIABLE: WHY EWMA VOLATILITY?**   
   
As the goal is to predict BTC's future volatility and crypto market are extremely volatile, two key considerations arise:

- Recent market behavior tends to be more informative than older, more distant observations.
- Excessive noise within the volatility series can harm model performance. This is particularly true for ML models benefitting from smoother target signals.

These suggest that **a 5-days Exponentially Weighted Moving Average (EWMA) volatility** is a well-suited target variable, as it excels at *a)* smoothing volatility signal (i.e reducing noise); and *b)* prioritizing recent trends through decaying weights. Furthermore, a span = 5 gives approximately 33% weight to the most recent return, ensuring both responsiveness to market's shifts and memory of past movements.  
*Note.* Rolling volatility with different span's values (5-days, 21-days, 30-days) has been tested too. However, all resulted in significantly weaker performances with respect to EMWA volatility.  

## ⚙️ HOW IT WORKS
The model's functioning is quite easy: once the user inputs the date (YYYY-MM-DD format) of interest, the model:
- 1️⃣ Parses BTC OHLCV, fear and greed index and google trend raw data for that specific date.
- 2️⃣ Engineers the dataframe and assigns a cluster.
- 3️⃣ Feeds the data into three different models.
- 4️⃣ Returns point and interval predictions for the same (t) and next (t+1) day EWMA Volatility.  

![BTCModel-example ](https://github.com/user-attachments/assets/e5cd5d47-aab4-4c64-867d-069ab2c08d13)

## 🚀 DEPLOYMENT & RE-TRAINING
Model is deployed on Hugging Face 🤗 spaces.
DEPLOYMENT (HF) & AUTOMATIC RE-TRAINING 

