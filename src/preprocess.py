# src/preprocess.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(path, scaler_path=None, fit_scaler=True):
    df = pd.read_csv(path)

    # Drop unnecessary columns
    if 'salary' in df.columns:
        df.drop(['salary', 'salary_currency'], axis=1, inplace=True, errors='ignore')

    # Remove outliers in salary_in_usd (above 99th percentile)
    q_high = df["salary_in_usd"].quantile(0.99)
    df = df[df["salary_in_usd"] < q_high]

    # Encode categorical variables using One-Hot Encoding
    categorical_cols = ['experience_level', 'employment_type', 'job_title',
                        'employee_residence', 'company_location', 'company_size']
    df = pd.get_dummies(df, columns=categorical_cols)

    # Feature scaling
    scaler = StandardScaler()
    if fit_scaler:
        df[['work_year', 'remote_ratio']] = scaler.fit_transform(df[['work_year', 'remote_ratio']])

        # Ensure model directory exists before saving
        model_dir = 'model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
        joblib.dump(df.drop(['salary_in_usd'], axis=1).columns, os.path.join(model_dir, 'train_columns.pkl'))

    else:
        scaler = joblib.load(scaler_path)
        df[['work_year', 'remote_ratio']] = scaler.transform(df[['work_year', 'remote_ratio']])

    X = df.drop(['salary_in_usd'], axis=1)
    y = df['salary_in_usd']

    return X, y, scaler
