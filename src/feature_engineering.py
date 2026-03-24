import numpy as np

# -----------------------------
# Time Features
# -----------------------------
def add_time_features(df):
    """
    Extract time-based features from pickup datetime.

    Features:
    ---------
    - pickup_hour: Hour of the day (0–23)
    - pickup_day: Day of the week (0=Monday, 6=Sunday)
    - is_weekend: Binary flag indicating weekends

    Why?
    ----
    Traffic patterns vary depending on time (rush hours, weekends),
    so these features help the model capture temporal effects.
    """
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day'] = df['pickup_datetime'].dt.dayofweek
    df['is_weekend'] = df['pickup_day'].isin([5, 6]).astype(int)
    return df


# -----------------------------
# Cyclical Encoding 
# -----------------------------
def add_cyclical_features(df):
    """
    Apply cyclical encoding to time-based features (hour, day of week).

    Why?
    ----
    Time features such as hours (0-23) and days (0-6) are cyclical in nature,
    meaning that their values wrap around (e.g., 23 is close to 0, Sunday is close to Monday).
    
    Using raw integer values can mislead models (especially linear models),
    as they assume a linear relationship between values.

    Solution:
    ---------
    We transform these features using sine and cosine functions to map them
    onto a circular representation. This preserves the cyclical relationships
    and allows the model to better capture temporal patterns.

    Example:
    --------
    hour = 23 and hour = 0 will have similar encoded values.
    """

    # hour
    df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)

    # day of week
    df['day_sin'] = np.sin(2 * np.pi * df['pickup_day'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['pickup_day'] / 7)

    return df


# -----------------------------
# Distance
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance between two points on Earth.

    Why?
    ----
    Euclidean distance is not accurate for geographic coordinates.
    Haversine formula accounts for Earth's curvature.

    Returns:
    --------
    distance in kilometers
    """
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def add_distance_feature(df):
    """
    Create distance feature between pickup and dropoff locations.

    Why?
    ----
    Distance is the most important factor affecting trip duration.
    """
    df['distance_km'] = haversine(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    return df



# -----------------------------
# Encoding
# -----------------------------
def encode_features(df):
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'Y': 1, 'N': 0})
    return df


# -----------------------------
# Main Pipeline
# -----------------------------
def prepare_features(df):
    df = add_time_features(df)
    df = add_cyclical_features(df)  
    df = add_distance_feature(df)
    df = encode_features(df)

    return df