import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv("D:\\Sustainathon\\startup data.csv")  # Replace with your actual file path

# Drop irrelevant columns
columns_to_drop = ["id", "name", "zip_code", "city", "Unnamed: 0", "Unnamed: 6"]
data = data.drop(columns=columns_to_drop, errors="ignore")

# Handle missing values
data = data.fillna(data.median(numeric_only=True))  # Fill numeric columns with median
data = data.fillna("unknown")  # Fill categorical columns with "unknown"

# Identify categorical columns
categorical_columns = data.select_dtypes(include=["object"]).columns

# Encode categorical variables using LabelEncoder
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Select features and target
target_column = "labels"  # Replace with the correct target column
X = data.drop(target_column, axis=1)
y = data[target_column]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



# Use cross-validation to prevent overfitting
cv_scores_rf = cross_val_score(RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42), X_train, y_train, cv=5)
print("Random Forest Cross-Validation Scores: ", cv_scores_rf)
print("Average Cross-Validation Score for Random Forest: ", cv_scores_rf.mean())

cv_scores_gb = cross_val_score(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42), X_train, y_train, cv=5)
print("Gradient Boosting Cross-Validation Scores: ", cv_scores_gb)
print("Average Cross-Validation Score for Gradient Boosting: ", cv_scores_gb.mean())

# Hyperparameter tuning for RandomForest
param_grid_rf = {
    'n_estimators': [100, 150],
    'max_depth': [10, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'bootstrap': [True]
}

# GridSearchCV for RandomForest
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)

# Train the RandomForest model with best parameters
best_model_rf = grid_search_rf.best_estimator_
y_pred_rf = best_model_rf.predict(X_test)

# Evaluate the RandomForest model
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Hyperparameter tuning for Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 150],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

# GridSearchCV for Gradient Boosting
grid_search_gb = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42), param_grid=param_grid_gb, cv=5, n_jobs=-1, verbose=2)
grid_search_gb.fit(X_train, y_train)

# Train the GradientBoosting model with best parameters
best_model_gb = grid_search_gb.best_estimator_
y_pred_gb = best_model_gb.predict(X_test)

# Evaluate the GradientBoosting model
print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, y_pred_gb))

# Sample Test Data
sample_data = {
    "state_code": [3, 3, 1],  # Example values
    "latitude": [40.712776, 37.238916, 32.901049],
    "longitude": [-71.056820, -121.973718, -117.192656],
    "founded_at": [5, 8, 106],
    "closed_at": [201, 202, 202],
    "relationships": [0, 1, 1],
    "funding_rounds": [0, 3, 4],
    "funding_total_usd": [100000, 200000, 50000],
    "milestones": [0, 2, 3],
    "avg_participants": [1.0, 4.75, 4.0],
    "is_top500": [0, 1, 1],
}

sample_df = pd.DataFrame(sample_data)

# Ensure the sample data has the same columns as the training data
missing_columns = [col for col in X.columns if col not in sample_df.columns]
for col in missing_columns:
    sample_df[col] = 0  # Add missing columns with default values (0 or "unknown")

# Ensure column order matches the training data
sample_df = sample_df[X.columns]

# Scale the sample data
sample_scaled = scaler.transform(sample_df)

# Predict the success of startups using the trained model
predictions_rf = best_model_rf.predict(sample_scaled)
predictions_gb = best_model_gb.predict(sample_scaled)

# Output predictions
print("\nRandom Forest Predictions for Sample Data:")
for i, prediction in enumerate(predictions_rf):
    result = "Successful" if prediction == 1 else "Unsuccessful"
    print(f"Sample {i + 1}: {result}")

print("\nGradient Boosting Predictions for Sample Data:")
for i, prediction in enumerate(predictions_gb):
    result = "Successful" if prediction == 1 else "Unsuccessful"
    print(f"Sample {i + 1}: {result}")

# Evaluation metrics for both models
print("\nRandom Forest F1-Score: ", f1_score(y_test, y_pred_rf, average='weighted'))
print("\nRandom Forest Precision: ", precision_score(y_test, y_pred_rf))
print("\nRandom Forest Recall: ", recall_score(y_test, y_pred_rf))

print("\nGradient Boosting F1-Score: ", f1_score(y_test, y_pred_gb, average='weighted'))
print("\nGradient Boosting Precision: ", precision_score(y_test, y_pred_gb))
print("\nGradient Boosting Recall: ", recall_score(y_test, y_pred_gb))

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=[0, 1])
disp_rf.plot(cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()

# Confusion Matrix for Gradient Boosting
cm_gb = confusion_matrix(y_test, y_pred_gb)
disp_gb = ConfusionMatrixDisplay(confusion_matrix=cm_gb, display_labels=[0, 1])
disp_gb.plot(cmap="Blues")
plt.title("Gradient Boosting Confusion Matrix")
plt.show()

import pickle

# Save the trained Random Forest model
with open('model/random_forest_model.pkl', 'wb') as rf_file:
    pickle.dump(best_model_rf, rf_file)

# Save the trained Gradient Boosting model
with open('model/gradient_boosting_model.pkl', 'wb') as gb_file:
    pickle.dump(best_model_gb, gb_file)

# Save the scaler
with open('model/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)