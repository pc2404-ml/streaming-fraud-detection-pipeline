import os
from feature_engineering import feature_engineering
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import warnings
import json

warnings.filterwarnings('ignore')

print(" ML PIPELINE - SINGLE THREAD VERSION")

def main():
    # Get current working directory
    current_path = os.getcwd()
    print(f"Current working directory: {current_path}")

    data_input_path = current_path + "/output/training_data/part-*.csv"
    data_output_path = current_path + '/data/credit_card_fraud_fe_updated.csv'

    feature_engineering(data_input_path, data_output_path)

    print("\n STEP 1: LOADING FEATURE-ENGINEERED DATA")
    print("-" * 40)

    # Load your processed data
    df = pd.read_csv(data_output_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.shape[1]}")
    print(f"Fraud rate: {df['is_fraud'].mean() * 100:.2f}%")

    # Quick data quality check
    print(f"\nData Quality Check:")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    # Separate features and target
    X = df.drop(['is_fraud'], axis=1)
    y = df['is_fraud']

    print(f"Features (X): {X.shape}")
    print(f"Target (y): {y.shape}")
    print(f"Feature names: {list(X.columns[:5])}...")

    # DATA TYPE CHECK & FIX
    print("\n STEP 1.5: DATA TYPE VERIFICATION")
    print("-" * 35)

    non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()

    if non_numeric_cols:
        print(f"Found {len(non_numeric_cols)} non-numeric columns after feature engineering:")
        for col in non_numeric_cols:
            unique_vals = X[col].nunique()
            sample_vals = X[col].unique()[:3]
            print(f"   {col}: {unique_vals} unique values, samples: {sample_vals}")

        print("\n Applying Label Encoding to remaining categorical columns...")

        label_encoders = {}
        for col in non_numeric_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
            print(f"Encoded {col}")

        if label_encoders:
            os.makedirs('model', exist_ok=True)
            joblib.dump(label_encoders, 'model/remaining_label_encoders.pkl')
            print(f"Saved label encoders for {len(label_encoders)} columns")

    else:
        print(" All columns are already numeric - feature engineering worked perfectly!")

    print(f"\nFinal data types:")
    print(f"Numeric columns: {len(X.select_dtypes(include=['number']).columns)}")
    print(f"Non-numeric columns: {len(X.select_dtypes(include=['object']).columns)}")

    # STEP 2: DATA SPLITTING
    print("\n STEP 2: DATA SPLITTING")
    print("-" * 25)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"Training set:   {X_train.shape[0]:,} samples ({y_train.mean() * 100:.2f}% fraud)")
    print(f"Validation set: {X_val.shape[0]:,} samples ({y_val.mean() * 100:.2f}% fraud)")
    print(f"Test set:       {X_test.shape[0]:,} samples ({y_test.mean() * 100:.2f}% fraud)")

    # STEP 3: HANDLE CLASS IMBALANCE
    print("\n STEP 3: HANDLING CLASS IMBALANCE")
    print("-" * 35)

    print("Original class distribution:")
    print(f"Normal: {(y_train == 0).sum():,} ({(y_train == 0).mean() * 100:.1f}%)")
    print(f"Fraud:  {(y_train == 1).sum():,} ({(y_train == 1).mean() * 100:.1f}%)")

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print(f"\n After SMOTE balancing:")
    print(f"Normal: {(y_train_balanced == 0).sum():,} ({(y_train_balanced == 0).mean() * 100:.1f}%)")
    print(f"Fraud:  {(y_train_balanced == 1).sum():,} ({(y_train_balanced == 1).mean() * 100:.1f}%)")

    # STEP 4: MODEL TRAINING - ALL SINGLE THREADED
    print("\n STEP 4: MODEL TRAINING & COMPARISON")
    print("-" * 35)

    # Define models - ALL SINGLE THREADED
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=1),
        'SVM': SVC(random_state=42, probability=True, kernel='rbf')
    }

    model_results = {}

    print("Training and evaluating models (single-threaded)...")

    for name, model in models.items():
        print(f"\n Training {name}...")

        # Cross-validation - SINGLE THREADED
        cv_scores = cross_val_score(
            model, X_train_balanced, y_train_balanced,
            cv=3, scoring='roc_auc', n_jobs=1  # SINGLE THREAD
        )

        # Train model
        model.fit(X_train_balanced, y_train_balanced)

        # Validation predictions
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred_proba)

        model_results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'val_auc': val_auc
        }

        print(f"  CV AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"  Val AUC: {val_auc:.4f}")

    # Find best model
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['val_auc'])
    best_model = model_results[best_model_name]['model']
    best_val_auc = model_results[best_model_name]['val_auc']

    print(f"\n BEST MODEL: {best_model_name}")
    print(f"   Validation AUC: {best_val_auc:.4f}")

    # STEP 5: SIMPLIFIED HYPERPARAMETER TUNING - SINGLE THREADED

    print(f"\n STEP 5: HYPERPARAMETER TUNING (Single-threaded)")
    print("-" * 30)

    print(f"Tuning hyperparameters for {best_model_name}...")

    # Minimal parameter grids to speed up single-threaded execution
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20]
        },
        'XGBoost': {
            'n_estimators': [100],
            'max_depth': [6],
            'learning_rate': [0.1, 0.2]
        },
        'Logistic Regression': {
            'C': [0.1, 1, 10]
        },
        'SVM': {
            'C': [1, 10]
        }
    }

    if best_model_name in param_grids:
        # Create fresh model instance - SINGLE THREADED
        if best_model_name == 'Random Forest':
            base_model = RandomForestClassifier(random_state=42, n_jobs=1)
        elif best_model_name == 'XGBoost':
            base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=1)
        elif best_model_name == 'Logistic Regression':
            base_model = LogisticRegression(random_state=42, max_iter=1000)
        elif best_model_name == 'SVM':
            base_model = SVC(random_state=42, probability=True)

        # Grid search - SINGLE THREADED
        grid_search = GridSearchCV(
            base_model,
            param_grids[best_model_name],
            cv=3,
            scoring='roc_auc',
            n_jobs=1,  # SINGLE THREAD
            verbose=0  # Reduce output
        )

        grid_search.fit(X_train_balanced, y_train_balanced)
        best_model = grid_search.best_estimator_

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")

    # STEP 6: FINAL EVALUATION
    print(f"\n STEP 6: FINAL MODEL EVALUATION")
    print("-" * 30)

    test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_pred = best_model.predict(X_test)

    test_auc = roc_auc_score(y_test, test_pred_proba)

    print(f"FINAL RESULTS:")
    print(f"Model: {best_model_name}")
    print(f"Test AUC: {test_auc:.4f}")

    print(f"\n Classification Report:")
    print(classification_report(y_test, test_pred, target_names=['Normal', 'Fraud']))

    cm = confusion_matrix(y_test, test_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Normal  Fraud")
    print(f"Actual Normal   {cm[0, 0]:6d}  {cm[0, 1]:5d}")
    print(f"       Fraud    {cm[1, 0]:6d}  {cm[1, 1]:5d}")

    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0

    print(f"\nBusiness Impact:")
    print(f"Precision: {precision:.3f} (of flagged transactions, {precision * 100:.1f}% are actually fraud)")
    print(f"Recall: {recall:.3f} (catches {recall * 100:.1f}% of all fraud)")

    # STEP 7: SAVE MODEL AND ARTIFACTS
    print(f"\nSTEP 7: SAVING MODEL & ARTIFACTS")
    print("-" * 30)

    os.makedirs('model', exist_ok=True)

    model_filename = f'model/fraud_model_{best_model_name.lower().replace(" ", "_")}.pkl'
    joblib.dump(best_model, model_filename)
    print(f"Model saved: {model_filename}")

    smote_filename = 'model/fraud_smote_sampler.pkl'
    joblib.dump(smote, smote_filename)
    print(f"SMOTE sampler saved: {smote_filename}")

    feature_names = list(X.columns)
    with open('model/feature_names.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    print(f"Feature names saved: model/feature_names.txt")

    performance_summary = {
        'model_name': best_model_name,
        'test_auc': float(test_auc),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'confusion_matrix': cm.tolist(),
        'feature_count': len(feature_names),
        'training_samples': len(X_train_balanced),
        'test_samples': len(X_test),
        'fraud_rate_original': float(y.mean()),
        'best_parameters': best_model.get_params() if hasattr(best_model, 'get_params') else None
    }

    with open('model/model_performance.json', 'w') as f:
        json.dump(performance_summary, f, indent=2)
    print(f"Performance summary saved: model/model_performance.json")

    print(f"\nFRAUD DETECTION MODEL READY!")
    print("=" * 40)
    print("Generated Files:")
    print(f"   • {model_filename} - Trained model")
    print(f"   • {smote_filename} - Data balancer")
    if 'label_encoders' in locals():
        print(f"   • model/remaining_label_encoders.pkl - Label encoders")
    print(f"   • model/feature_names.txt - Feature list")
    print(f"   • model/model_performance.json - Results summary")




if __name__ == '__main__':
    main()