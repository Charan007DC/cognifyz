import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('Data/Dataset .csv')

use_cols = [
    'City', 'Cuisines', 'Average Cost for two', 'Currency', 
    'Has Table booking', 'Has Online delivery', 'Price range', 
    'Aggregate rating', 'Votes'
]
df = df[use_cols].copy()

print("Cleaning data...")
df['Average Cost for two'] = pd.to_numeric(df['Average Cost for two'], errors='coerce')
df['Price range'] = pd.to_numeric(df['Price range'], errors='coerce')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Aggregate rating'] = pd.to_numeric(df['Aggregate rating'], errors='coerce')

df = df.dropna(subset=['Aggregate rating'])

df['Average Cost for two'].fillna(df['Average Cost for two'].median(), inplace=True)
df['Price range'].fillna(df['Price range'].median(), inplace=True)
df['Votes'].fillna(df['Votes'].median(), inplace=True)

df['Has Table booking'] = df['Has Table booking'].map({'Yes': 1, 'No': 0})
df['Has Online delivery'] = df['Has Online delivery'].map({'Yes': 1, 'No': 0})

df['Cuisines'] = df['Cuisines'].fillna('Unknown')
df['cuisine_count'] = df['Cuisines'].str.split(',').apply(len)

label_encoders = {}
categorical_cols = ['City', 'Cuisines', 'Currency']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=['Aggregate rating'])
y = df['Aggregate rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training models...")
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R²': r2}
    print(f"{name} -> MSE: {mse:.4f}, R²: {r2:.4f}")

best_model = models['Random Forest']
importances = best_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 5 Most Influential Features:")
print(importance_df.head())

importance_df.to_csv('feature_importance.csv', index=False)
print("\nFeature importance saved to 'feature_importance.csv'")
print("Task 1 completed successfully!")