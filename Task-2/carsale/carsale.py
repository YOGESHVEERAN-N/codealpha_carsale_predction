# ============================================================
# 🚗 CAR PRICE PREDICTION USING MACHINE LEARNING
# Dataset: car data.csv
# ============================================================

# =========================
# 1. IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================
# 2. LOAD DATASET
# =========================

df = pd.read_csv("car data.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Display first 5 rows
print("========== FIRST 5 ROWS ==========")
print(df.head())

# =========================
# 3. DATASET INFORMATION
# =========================

print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# =========================
# 4. FEATURE ENGINEERING
# =========================

# Create new feature: Car Age
current_year = 2025

df['Car_Age'] = current_year - df['Year']

# Drop unnecessary columns
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

print("\n========== UPDATED DATASET ==========")
print(df.head())

# =========================
# 5. ENCODE CATEGORICAL DATA
# =========================

le = LabelEncoder()

df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])
df['Selling_type'] = le.fit_transform(df['Selling_type'])
df['Transmission'] = le.fit_transform(df['Transmission'])

print("\n========== ENCODED DATASET ==========")
print(df.head())

# =========================
# 6. DATA VISUALIZATION
# =========================

# Correlation Heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()

# Selling Price Distribution
plt.figure(figsize=(8, 5))

sns.histplot(df['Selling_Price'], bins=30, kde=True)

plt.title("Selling Price Distribution")

plt.xlabel("Selling Price")

plt.show()

# =========================
# 7. SPLIT FEATURES & TARGET
# =========================

X = df.drop('Selling_Price', axis=1)

y = df['Selling_Price']

# =========================
# 8. TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING DATA SIZE:", X_train.shape)

print("TESTING DATA SIZE:", X_test.shape)

# =========================
# 9. TRAIN LINEAR REGRESSION MODEL
# =========================

model = LinearRegression()

model.fit(X_train, y_train)

print("\n========== MODEL TRAINED SUCCESSFULLY ==========")

# =========================
# 10. MAKE PREDICTIONS
# =========================

y_pred = model.predict(X_test)

print("\n========== PREDICTED VALUES ==========")

print(y_pred[:10])

# =========================
# 11. MODEL EVALUATION
# =========================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print("Mean Absolute Error (MAE):", mae)

print("Mean Squared Error (MSE):", mse)

print("Root Mean Squared Error (RMSE):", rmse)

print("R2 Score:", r2)

# =========================
# 12. ACTUAL vs PREDICTED GRAPH
# =========================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Car Prices")

plt.show()

# =========================
# 13. SAMPLE PREDICTION
# =========================

# Create sample input data
sample_data = pd.DataFrame({
    'Present_Price': [5.59],
    'Driven_kms': [27000],
    'Fuel_Type': [2],
    'Selling_type': [0],
    'Transmission': [1],
    'Owner': [0],
    'Car_Age': [11]
})

# Predict car price
prediction = model.predict(sample_data)

print("\n========== SAMPLE PREDICTION ==========")

print("Predicted Car Price:", prediction[0])

# ============================================================
# END OF PROJECT
# ============================================================