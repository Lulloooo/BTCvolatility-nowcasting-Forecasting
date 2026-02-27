#script to build a google trend dataframe with a time frame of 7 years by merging 6months windows

#load the needed packages
import pandas as pd          
import os                    

#define the function to import and merge the datasets
def imp_datasets(periods=[1, 2], years=[17, 18, 19, 20, 21, 22, 23, 24, 25], path="data/google_trends_data"):
    # Initialize an empty list to store all DataFrames
    all_dfs = []

    # Loop over each year
    for year in years:
        #loop over period 
        for period in periods:
            # Build the full file path: e.g., data/google_trends_data/2017_1.csv
            filename = os.path.join(path, f"20{year}_{period}.csv")
            
            #set a try function to handle error (ex. df missing) without crasing the loop
            try:
                # Read the CSV file into a DataFrame 
                # Set the first column as the index and parse it as dates (for time series)
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                # Add the DataFrame to the list
                all_dfs.append(df)
                # Print confirmation message
                print(f"Loaded: {filename}")
            # If the file doesn't exist, skip it and print a warning
            except FileNotFoundError: #this Catches the specific case where the file doesn’t exist at the given path
                print(f"Skipped (file not found): {filename}")
            # Catch any other error and print it
            except Exception as e: #Catches any other unexpected error (e.g., file is corrupted, wrong format, etc.) and prints the error message e
                print(f"Error loading {filename}: {e}")

    # After loading all DataFrames, check if the list is not empty
    if all_dfs:
        # Concatenate all DataFrames into a single DataFrame
        final_df = pd.concat(all_dfs)
        # Sort the DataFrame by index (assumed to be dates)
        final_df = final_df.sort_index()
        # Drop any duplicated index entries, keeping the first occurrence
        final_df = final_df[~final_df.index.duplicated(keep='first')]
        # Print success message
        print("✅ All data concatenated.")
        
        # Return the final concatenated DataFrame
        return final_df
    else:
        # If no data was loaded, return an empty DataFrame
        print("⚠️ No data was loaded.")
        return pd.DataFrame()
    
    
# Run the function and save the result to a google_trends.csv
# Run the function and save the result to a google_trends.csv
if __name__ == "__main__":
    df_all = imp_datasets()
    # Rename the single column to a meaningful name
    df_all.columns = ['bit_trend']  
    # Rename the index to 'timestamp'
    df_all.index.name = 'timestamp'
    #save the df if the process was done right
    if not df_all.empty:
        df_all.to_csv("data/google_trends.csv")
        print("✅ Combined dataset saved to data/google_trends.csv")
    else:
        print("⚠️ Combined dataset is empty. Nothing was saved.")
