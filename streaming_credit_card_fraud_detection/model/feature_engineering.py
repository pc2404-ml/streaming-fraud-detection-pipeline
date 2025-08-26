import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import glob

def feature_engineering(data_input_path,data_output_path):

    #Data Load
    file=glob.glob(data_input_path)[0]
    # Load your data
    df = pd.read_csv(file)

    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    # Create a copy for feature engineering
    df_features = df.copy()

    # Amount-Based Features
    print("\n CREATING AMOUNT-BASED FEATURES")
    print("-" * 40)

    # Amount categories
    print("\n CREATING AMOUNT-BASED FEATURES")
    print("-" * 40)
    df_features['amount_category'] = pd.cut(df_features['transaction_amount'],
                                      bins=[0, 50, 200, 500, 1000, 2000, float('inf')],
                                      labels=['very_low', 'low', 'medium', 'high', 'very_high', 'extreme'])
    print("Created: amount_category")

    # Time-Based Features
    # Hour categories
    df_features['is_night'] = df_features['transaction_hour'].apply(lambda x: 1 if x <= 6 or x >= 22 else 0)
    df_features['is_morning'] = df_features['transaction_hour'].apply(lambda x: 1 if 6 < x <= 12 else 0)
    df_features['is_afternoon'] = df_features['transaction_hour'].apply(lambda x: 1 if 12 < x <= 18 else 0)
    df_features['is_evening'] = df_features['transaction_hour'].apply(lambda x: 1 if 18 < x < 22 else 0)
    print("Created: is_night, is_morning, is_afternoon, is_evening")

    # Business hours
    df_features['is_business_hours'] = df_features['transaction_hour'].apply(lambda x: 1 if 9 <= x <= 17 else 0)
    df_features['is_late_night'] = df_features['transaction_hour'].apply(lambda x: 1 if x <= 5 or x >= 23 else 0)
    df_features['is_peak_hours'] = df_features['transaction_hour'].apply(lambda x: 1 if x in [12, 13, 18, 19, 20] else 0)
    print("Created: is_business_hours, is_late_night, is_peak_hours")

    # Location-Based Features

    print("\n CREATING LOCATION-BASED FEATURES")
    print("-" * 40)

    # International transaction flag
    df_features['is_international'] = (df_features['country'] != 'United States').astype(int)
    print("Created: is_international")

    # Country risk score (based on fraud rate by country)
    country_fraud_rate = df_features.groupby('country')['is_fraud'].mean()
    df_features['country_risk_score'] = df_features['country'].map(country_fraud_rate)
    print("Created: country_risk_score")

    # State features (if US)
    if 'state' in df_features.columns:
        state_fraud_rate = df_features.groupby('state')['is_fraud'].mean()
        df_features['state_risk_score'] = df_features['state'].map(state_fraud_rate)
        print("Created: state_risk_score")

    # Major cities flag (you can customize this list)
    major_cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
                    'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville']
    if 'city' in df_features.columns:
        df_features['is_major_city'] = df_features['city'].isin(major_cities).astype(int)
        print("Created: is_major_city")

    # Merchant-Based Features
    print("\nCREATING MERCHANT-BASED FEATURES")
    print("-" * 40)

    # Merchant risk score
    merchant_fraud_rate = df_features.groupby('merchant')['is_fraud'].mean()
    df_features['merchant_risk_score'] = df_features['merchant'].map(merchant_fraud_rate)
    print(" Created: merchant_risk_score")

    # Merchant transaction volume
    merchant_volume = df_features.groupby('merchant').size()
    df_features['merchant_volume'] = df_features['merchant'].map(merchant_volume)
    print(" Created: merchant_volume")

    # High-risk merchant categories
    high_risk_merchants = ['Unknown Store', 'Suspicious Site', 'Foreign ATM', 'Online Store']
    df_features['is_high_risk_merchant'] = df_features['merchant'].isin(high_risk_merchants).astype(int)
    print(" Created: is_high_risk_merchant")

    # Average amount per merchant
    merchant_avg_amount = df_features.groupby('merchant')['transaction_amount'].mean()
    df_features['merchant_avg_amount'] = df_features['merchant'].map(merchant_avg_amount)
    df_features['amount_vs_merchant_avg'] = df_features['transaction_amount'] / df_features['merchant_avg_amount']
    print(" Created: merchant_avg_amount, amount_vs_merchant_avg")

    # Card-Based Features

    print("\n CREATING CARD-BASED FEATURES")
    print("-" * 40)

    # Card provider risk
    card_fraud_rate = df_features.groupby('card_provider')['is_fraud'].mean()
    df_features['card_provider_risk'] = df_features['card_provider'].map(card_fraud_rate)
    print(" Created: card_provider_risk")

    # Premium card indicator
    premium_cards = ['American Express', 'Amex']
    df_features['is_premium_card'] = df_features['card_provider'].isin(premium_cards).astype(int)
    print(" Created: is_premium_card")

    # Interaction Features

    print("\nCREATING INTERACTION FEATURES")
    print("-" * 40)

    # Risk score combinations
    df_features['combined_risk_score'] = (df_features['merchant_risk_score'] +
                                          df_features['country_risk_score'] +
                                          df_features['card_provider_risk']) / 3
    print(" Created: combined_risk_score")

    # Location × Time interaction
    df_features['international_night'] = df_features['is_international'] * df_features['is_night']
    df_features['high_risk_merchant_night'] = df_features['is_high_risk_merchant'] * df_features['is_night']
    print(" Created: international_night, high_risk_merchant_night")

    # ENCODING CATEGORICAL VARIABLES

    print("\n ENCODING CATEGORICAL VARIABLES")
    print("-" * 40)

    # Label encoding for high-cardinality categorical variables
    label_encoders = {}
    categorical_cols = ['merchant', 'card_provider', 'country', 'state', 'city', 'email_domain', 'phone_area_code']

    for col in categorical_cols:
        if col in df_features.columns:
            le = LabelEncoder()
            df_features[f'{col}_encoded'] = le.fit_transform(df_features[col].astype(str))
            label_encoders[col] = le
            print(f" Encoded: {col}")

    # SCALING FEATURES

    print("\nSCALING NUMERICAL FEATURES")
    print("-" * 40)

    # Features to scale
    numerical_features = ['transaction_amount',
                          'merchant_volume', 'merchant_avg_amount', 'amount_vs_merchant_avg'
                          ]

    # Standard scaling
    scaler = StandardScaler()
    scaled_features = []

    for feature in numerical_features:
        if feature in df_features.columns:
            scaled_name = f'{feature}_scaled'
            df_features[scaled_name] = scaler.fit_transform(df_features[[feature]])
            scaled_features.append(scaled_name)
            print(f" Scaled: {feature}")

    # Saving Data

    df_features.to_csv(data_output_path, index=False)
    print(f" Saved to: {data_output_path}")


