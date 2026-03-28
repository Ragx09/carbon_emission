import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_preprocessing_pipeline():
    """
    Defines the transformation steps for numeric and categorical data.
    Aligns with Week 7 (Feature Engineering) and Week 9 (Scaling for SVM).
    """
    numeric_features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']
    categorical_features = ['Fuel Type', 'Transmission']

    # Numeric: Scaling is essential for KNN and SVM distance calculations
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Dense OHE keeps KNN/SVR paths stable (sparse + KNN can force huge dense blocks).
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical_transformer = Pipeline(steps=[("onehot", onehot)])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def load_and_split_data(filepath):
    filepath = Path(filepath)
    df = pd.read_csv(filepath)
    X = df.drop("CO2 Emissions(g/km)", axis=1)
    y = df["CO2 Emissions(g/km)"]
    
    # Standard 80/20 split
    return train_test_split(X, y, test_size=0.2, random_state=42)