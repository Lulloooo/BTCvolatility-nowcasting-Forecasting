#gather fear e greed index (CFGI) -> measures crypto markets sentiment

#import needed packages
import requests
import pandas as pd

#define the function to gather fear&greed Index
def fear_greed_index():
    #API to get all historical data in JSON
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    #make an HTTP get req
    response = requests.get(url)
    #convert http in json and extract "data"
    data = response.json()["data"]
    
    #convert data (a list of dictionaries) into a df 
    df = pd.DataFrame(data)
    #change timestamp to datetime format in seconds
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit = "s")
    #set timestamp as the index
    df.set_index("timestamp", inplace = True)
    #select only the value column and change it into float
    df = df[["value"]].astype(float)
    #rename the column
    df.rename(columns = {"value" : "fear&greed_index"}, inplace = True)
    
    #return the final df sorted chronologically
    return df.sort_index()

#run the function only if the script is run directly
if __name__ == "__main__":
    df = fear_greed_index()
    df.to_csv('data/fear_greed_index_updated.csv')
    print("✅ Saved Fear & Greed Index data.")