import pandas as pd
import joblib
import json
import os
import tempfile
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from feature_engineering import feature_engineering


class CleanFraudPipeline:
    """Clean pipeline for fraud detection"""

    def __init__(self, model_path='model/'):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.label_encoders = None
        self._load_model_components()

    def _load_model_components(self):
        """Load trained model components"""
        try:
            # Load model info
            with open(f'{self.model_path}model_performance.json', 'r') as f:
                model_info = json.load(f)

            model_name = model_info['model_name'].lower().replace(" ", "_")
            self.model = joblib.load(f'{self.model_path}fraud_model_{model_name}.pkl')

            # Load feature names
            with open(f'{self.model_path}feature_names.txt', 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]

            # Load label encoders if they exist
            try:
                self.label_encoders = joblib.load(f'{self.model_path}remaining_label_encoders.pkl')
            except:
                self.label_encoders = {}

            print(f"Pipeline loaded: {model_info['model_name']}")
            print(f"Features: {len(self.feature_names)}")

        except Exception as e:
            print(f"Error loading model components: {e}")
            raise

    def predict_fraud(self, raw_transaction):
        """Process single transaction through fraud detection pipeline"""

        try:
            # Convert to DataFrame
            if isinstance(raw_transaction, dict):
                df_raw = pd.DataFrame([raw_transaction])
            else:
                df_raw = raw_transaction.copy()

            # Column mapping for different input formats
            column_mapping = {
                'amount': 'transaction_amount',
                'hour': 'transaction_hour',
                'merchant_name': 'merchant',
                'card_type': 'card_provider',
                'user_country': 'country',
                'user_state': 'state',
                'user_city': 'city',
                'user_email': 'email',
                'user_phone': 'phone'
            }

            # Apply column mapping
            for old_col, new_col in column_mapping.items():
                if old_col in df_raw.columns and new_col not in df_raw.columns:
                    df_raw[new_col] = df_raw[old_col]

            # Add missing required columns with defaults
            required_columns = {
                'transaction_amount': 0.0,
                'transaction_hour': datetime.now().hour,
                'merchant': 'Unknown Store',
                'card_provider': 'Unknown',
                'country': 'United States',
                'state': 'Unknown',
                'city': 'Unknown',
                'email': 'unknown@example.com',
                'phone': '+1-000-000-0000',
                'is_fraud': 0
            }

            for col, default_value in required_columns.items():
                if col not in df_raw.columns:
                    df_raw[col] = default_value

            # Create temp files for feature engineering
            temp_dir = tempfile.mkdtemp()
            temp_input_file = os.path.join(temp_dir, "raw_transaction.csv")
            temp_output_file = os.path.join(temp_dir, "engineered_features.csv")

            df_raw.to_csv(temp_input_file, index=False)

            # Run feature engineering
            feature_engineering(temp_input_file, temp_output_file)

            # Load engineered features
            df_features = pd.read_csv(temp_output_file)

            # Prepare for prediction
            if 'is_fraud' in df_features.columns:
                X = df_features.drop(['is_fraud'], axis=1)
            else:
                X = df_features.copy()

            # Apply label encoders
            for col, encoder in self.label_encoders.items():
                if col in X.columns:
                    try:
                        X[col] = encoder.transform(X[col].astype(str))
                    except ValueError:
                        X[col] = encoder.transform([encoder.classes_[0]] * len(X))

            # Add missing features
            for feature in self.feature_names:
                if feature not in X.columns:
                    X[feature] = 0.0

            # Select features in correct order
            X = X[self.feature_names]

            # Make prediction
            fraud_prob = self.model.predict_proba(X)[:, 1][0]
            is_fraud = self.model.predict(X)[0]

            # Business logic
            if fraud_prob > 0.8:
                risk_level = "HIGH"
                recommendation = "BLOCK"
            elif fraud_prob > 0.5:
                risk_level = "MEDIUM"
                recommendation = "REVIEW"
            else:
                risk_level = "LOW"
                recommendation = "APPROVE"

            # Clean up temp files
            try:
                os.remove(temp_input_file)
                os.remove(temp_output_file)
                os.rmdir(temp_dir)
            except:
                pass

            return {
                'is_fraud': bool(is_fraud),
                'fraud_probability': round(fraud_prob, 4),
                'fraud_percentage': round(fraud_prob * 100, 2),
                'risk_level': risk_level,
                'recommendation': recommendation,
                'transaction_details': {
                    'amount': float(raw_transaction.get('amount', raw_transaction.get('transaction_amount', 0))),
                    'merchant': raw_transaction.get('merchant', raw_transaction.get('merchant_name', 'Unknown')),
                    'hour': raw_transaction.get('hour', raw_transaction.get('transaction_hour', 0))
                },
                'processing_status': 'SUCCESS'
            }

        except Exception as e:
            return {
                'processing_status': 'ERROR',
                'error_message': str(e),
                'recommendation': 'MANUAL_REVIEW'
            }


def detect_fraud_batch(csv_file_path, output_file='model/batch_predictions.csv'):
    """Process CSV file and perform batch predictions"""

    print(f"Loading transactions from: {csv_file_path}")

    # Read CSV file
    df = pd.read_csv(csv_file_path)
    print(f"Loaded {len(df)} transactions for prediction")

    # Initialize pipeline
    pipeline = CleanFraudPipeline()

    # Process each transaction
    results = []

    for i, row in df.iterrows():
        transaction = row.to_dict()
        result = pipeline.predict_fraud(transaction)

        # Add transaction ID
        result['transaction_id'] = i
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(df)} transactions")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    # Show summary
    fraud_count = len([r for r in results if r.get('recommendation') == 'BLOCK'])
    approve_count = len([r for r in results if r.get('recommendation') == 'APPROVE'])
    review_count = len([r for r in results if r.get('recommendation') == 'REVIEW'])

    print(f"\nBatch Prediction Results:")
    print(f"  BLOCK (Fraud): {fraud_count}")
    print(f"  APPROVE: {approve_count}")
    print(f"  REVIEW: {review_count}")
    print(f"  Fraud Rate: {fraud_count / len(results) * 100:.1f}%")
    print(f"\nResults saved to: {output_file}")

    results_df.to_csv(output_file, index=False)
    print(f"Prediction file CSV created: {output_file}")
    return results_df




if __name__ == "__main__":
    import sys
    # Create sample and run prediction on it
    predict_file = 'model/predict_transactions.csv'
    output_file = 'model/batch_predictions.csv'
    detect_fraud_batch(predict_file)
    print('process completed')