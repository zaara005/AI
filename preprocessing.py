#Data Preprocessing Script for DataScience_salaries_2025 dataset

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1. Load dataset
df = pd.read_csv('DataScience_salaries_2025.csv')

# Step 2. Drop unnecessary columns
df.drop(['salary', 'salary_currency'], axis=1, inplace=True)

# Step 3. Remove outliers in salary_in_usd (above 99th percentile)
q_high = df["salary_in_usd"].quantile(0.99)
df = df[df["salary_in_usd"] < q_high]

# Step 4. Encode categorical variables using One-Hot Encoding
categorical_cols = ['experience_level', 'employment_type', 'job_title',
                    'employee_residence', 'company_location', 'company_size']

df = pd.get_dummies(df, columns=categorical_cols)

# Step 5. Feature scaling (StandardScaler for numeric features)
scaler = StandardScaler()
df[['work_year', 'remote_ratio']] = scaler.fit_transform(df[['work_year', 'remote_ratio']])

# Step 6. Split into features and target
X = df.drop(['salary_in_usd'], axis=1)
y = df['salary_in_usd']

# Step 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Final shapes for confirmation
print("Training features shape:", X_train.shape)
print("Test features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Test target shape:", y_test.shape)
