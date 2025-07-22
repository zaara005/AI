# src/train.py

import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from src.preprocess import preprocess_data

def train_and_save_model(data_path, model_dir, model_path):
    # Ensure model directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory: {model_dir}")

    X, y, scaler = preprocess_data(data_path, fit_scaler=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    print(f"Random Forest Regressor R² Score: {r2:.4f}")
    print(f"Random Forest Regressor RMSE: {rmse:.2f}")

    joblib.dump(model, model_path)
    print(f"Model saved successfully at {model_path}.")

if __name__ == "__main__":
    data_path = "data/DataScience_salaries_2025.csv"
    model_dir = "model"
    model_path = os.path.join(model_dir, "rf_model.pkl")
    train_and_save_model(data_path, model_dir, model_path)
