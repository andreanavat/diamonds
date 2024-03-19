import pandas as pd
from scipy import stats
from scipy.stats import ksone
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import re
from joblib import dump, load
import joblib
import json
import functions_framework


# Function to create the response

def create_response(message: str, status_code: int):
    response_message = {
        "message": message
    }
    return json.dumps(response_message), status_code

mapeo = {
    'SI2': 'SI2', 'SI1': 'SI1', 'VS1': 'VS1', 'VS2': 'VS2', 'VVS2': 'VVS2', 'VVS1': 'VVS1',
    'I1': 'I1', 'IF': 'IF',
    'S?I1': 'SI1', 'SI!1': 'SI1', '&VS2': 'VS2', '&SI2': 'SI2', "S*'I1": 'SI1',
    'VS?1': 'VS1', "S*'I2": 'SI2', '#VS1': 'VS1', 'V&S2': 'VS2', 'V!S2': 'VS2',
    '!VS2': 'VS2', 'VS#2': 'VS2', "VVS*'2": 'VVS2', "*'SI2": 'SI2', 'VV?S1': 'VVS1',
    'S&I1': 'SI1', "*'SI1": 'SI1', 'SI?1': 'SI1', 'VV#S1': 'VVS1', 'V#S2': 'VS2',
    '#SI!1': 'SI1', 'S!I2': 'SI2'
}

list_corr = ['carat', 'y', 'z', 'x', 'table', 'clarity_SI2', 'clarity_VVS1',
 'cut_Ideal', 'cut_Premium', 'color_E', 'clarity_IF', 'clarity_VVS2', 'color_J',
 'color_D', 'cut_Fair', 'color_H', 'clarity_SI1', 'color_I', 'depth',
 'cut_Good', 'clarity_I1', 'cut_Very Good', 'clarity_VS1', 'color_G']




# Function to clean the values
def clean_value(value):
    # Remove special characters
    try:
        clean_value = re.sub(r"[^a-zA-Z\s]", '', value)
        # Correct common errors and standardize
        corrections = {
            'Very Goo': 'Very Good',
            'Very Go': 'Very Good',
            'V&ery Good': 'Very Good',
            'Very G#ood': 'Very Good',
            "Very *'Good": 'Very Good',
            'Very Go#od': 'Very Good',
            'Ide&al': 'Ideal',
            'Ide!al': 'Ideal',
            'Id!eal': 'Ideal',
            "Ide*'al": 'Ideal',
            '*Ideal': 'Ideal',
            'I#deal': 'Ideal',
            '&Ideal': 'Ideal',
            'Pre!mium': 'Premium',
            'Pr?emium': 'Premium',
            "P*'remium": 'Premium',
            'P?remium': 'Premium',
            '&Premium': 'Premium',
            'Go?od': 'Good',
            'G#ood': 'Good',
            '!Good': 'Good'
        }
        return corrections.get(clean_value, clean_value)
    except:
        return create_response(f"Error while cleaning the value: {value}", 500)

def validate_num_columns(df, num_columns):
    try:
        for column in num_columns:
            data_type = df[column].dtypes

            # Check if the data type is numeric
            if data_type != 'int' or data_type != 'float':
                print(f"The column {column} contains non-numeric data")
    except:
        return create_response(f"Error while validating the column: {column}", 500)


def categorical_column(df, num_columns):
    try:
        for column in num_columns:
            data_type = df[column].dtypes

            # Check if the data type is numeric
            if data_type != 'object':
                print(f"The column {column} contains non-text data")
    except:
        return create_response(f"Error while validating the column: {column}", 500)

def normalize(df):
    try:
        X_train,X_test=train_test_split(df,test_size=.2,random_state=413)
        scaler = MinMaxScaler()
        scaler.fit(X_train)

        index_train = X_train.index
        index_test = X_test.index

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=index_test)
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=index_train)
        df = pd.concat([X_train, X_test], axis=0)
        return df
    except:
        return create_response(f"Error while normalizing the data", 500)
    
def preprocessing(df, list_corr, num_columns_numeric, num_columns_categoric):
    try:
        df['cut'] = df['cut'].apply(clean_value)
        df['color'] = df['color'].replace(to_replace=r"[^EIJHFGD]", value='', regex=True)
        df['clarity'] = df['clarity'].replace(mapeo, regex=False)
        validate_num_columns(df, num_columns_numeric)
        categorical_column(df, num_columns_categoric)
        df = pd.get_dummies(df, columns=num_columns_categoric, prefix_sep='_')  
        
        for column in list_corr:
            if column not in df.columns:
                df[column] = 0  # Create the column and assign the value '0'
        
        df = df[list_corr]  # Makes sure that df only contains the columns in list_corr
        print(df)
        df = normalize(df)
        return df
    except Exception as e:
        print(e)
        return create_response(f"Error while preprocessing the data", 500)

def obtain_predictions(df_preprocessed):
    try:
        model_loaded = load('model_regresion_lineal.joblib')
        precio_scaler_loaded = joblib.load('price_scaler.joblib')
        print("loaded model")
        # Making predictions with the model
        y_pred = model_loaded.predict(df_preprocessed)
        print("obtained predictions")
        # Performing the inverse transformation of the 'price' predictions
        y_pred_inverse = precio_scaler_loaded.inverse_transform(y_pred.reshape(-1, 1))
        return y_pred_inverse
    except:
        return create_response(f"Error while obtaining the predictions", 500)

@functions_framework.http
def main(request):
    try: 
        request_json = request.get_json()
        carat = request_json['carat']
        cut = request_json['cut']
        color = request_json['color']
        clarity = request_json['clarity']
        depth = request_json['depth']
        table = request_json['table']
        x = request_json['x']
        y = request_json['y']
        z = request_json['z']
        stolendiamonds = pd.DataFrame({
            'carat': [float(request_json['carat'])],
            'cut': [str(request_json['cut'])],
            'color': [str(request_json['color'])],
            'clarity': [str(request_json['clarity'])],
            'depth': [float(request_json['depth'])],
            'table': [float(request_json['table'])],
            'x': [float(request_json['x'])],
            'y': [float(request_json['y'])],
            'z': [float(request_json['z'])]
        })
        print(stolendiamonds.dtypes)
        num_columns_numeric = stolendiamonds.select_dtypes(include=['int64', 'float64']).columns
        num_columns_categoric = stolendiamonds.select_dtypes(include=['object']).columns

        df_preprocessed = preprocessing(stolendiamonds, list_corr, num_columns_numeric, num_columns_categoric)
        #print(df_preprocessed)
        price_prediction = obtain_predictions(df_preprocessed)
        
        return price_prediction
    except:
        return create_response(f"Error while processing the data", 500)