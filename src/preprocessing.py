import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df


def cleaning(df):
    """
    - Convert datetime columns to proper datetime format
    - Remove unrealistic trip durations (outliers) like 0s and 1s
    - Remove invalid passenger counts like 0 
    - Drop missing values

    
    """
    # convert datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

    # remove outliers
    df = df[df['trip_duration'].between(10, 10800)] 
    df = df[df['passenger_count'] > 0] 


    return df