import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load your data
data = pd.read_csv("D:\\Sustainathon\\startup data.csv")

# Preprocess the data (drop irrelevant columns, handle missing values, etc.)
columns_to_drop = ["id", "name", "zip_code", "city", "Unnamed: 0", "Unnamed: 6"]
data = data.drop(columns=columns_to_drop, errors="ignore")
data = data.fillna(data.median(numeric_only=True))  # Fill numeric columns with median
data = data.fillna("unknown")  # Fill categorical columns with "unknown"

# Encode categorical variables
categorical_columns = data.select_dtypes(include=["object"]).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Define features and target
target_column = "labels"  # Replace with the correct target column
X = data.drop(target_column, axis=1)
y = data[target_column]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for RandomForest
param_grid_rf = {
    'n_estimators': [100, 150],
    'max_depth': [10, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'bootstrap': [True]
}

grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)

# Train the RandomForest model with best parameters
best_model_rf = grid_search_rf.best_estimator_

# Pickle the trained model
with open('best_model_rf.pkl', 'wb') as rf_file:
    pickle.dump(best_model_rf, rf_file)

print("Model trained and saved successfully!")
