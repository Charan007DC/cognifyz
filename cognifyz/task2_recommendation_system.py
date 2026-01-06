import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print(" Starting Cognifyz Task 2: Recommendation System")
print("1. Loading dataset...")
full_df = pd.read_csv('Data/Dataset .csv')

features = [
    'City', 'Cuisines', 'Average Cost for two', 'Currency',
    'Has Table booking', 'Has Online delivery', 'Price range', 'Votes'
]
target = 'Aggregate rating'

df = full_df[features + [target] + ['Restaurant Name']].copy()

print("2. Cleaning data...")

df[target] = pd.to_numeric(df[target], errors='coerce')
df = df[df[target] > 0]  
df['Average Cost for two'] = pd.to_numeric(df['Average Cost for two'], errors='coerce')
df['Price range'] = pd.to_numeric(df['Price range'], errors='coerce')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

for col in ['Average Cost for two', 'Price range', 'Votes']:
    df[col].fillna(df[col].median(), inplace=True)

df['Has Table booking'] = df['Has Table booking'].map({'Yes': 1, 'No': 0}).fillna(0)
df['Has Online delivery'] = df['Has Online delivery'].map({'Yes': 1, 'No': 0}).fillna(0)

df['Cuisines'] = df['Cuisines'].fillna('Unknown')
df['cuisine_count'] = df['Cuisines'].str.split(',').apply(len)

le_city = LabelEncoder()
df['City'] = le_city.fit_transform(df['City'].astype(str))

le_cuisines = LabelEncoder()
df['Cuisines'] = le_cuisines.fit_transform(df['Cuisines'].astype(str))

le_curr = LabelEncoder()
df['Currency'] = le_curr.fit_transform(df['Currency'].astype(str))

print("3. Preparing features and training model...")

X = df.drop(columns=[target, 'Restaurant Name'])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n Model Performance:")
print(f"   - Mean Squared Error (MSE): {mse:.4f}")
print(f"   - R² Score: {r2:.4f}")

importances = model.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n🌟 Top 5 Most Important Features:")
print(feat_imp.head())

print("4. Building Recommendation System...")

from sklearn.metrics.pairwise import cosine_similarity

rec_features = ['City', 'Cuisines', 'Average Cost for two', 'Price range', 'cuisine_count']
rec_df = df[rec_features + ['Aggregate rating']].copy()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
rec_df[['Average Cost for two', 'Price range', 'cuisine_count']] = scaler.fit_transform(rec_df[['Average Cost for two', 'Price range', 'cuisine_count']])

similarity_matrix = cosine_similarity(rec_df.drop(columns=['Aggregate rating']))

def recommend_restaurants(restaurant_index, top_n=5):
    sim_scores = list(enumerate(similarity_matrix[restaurant_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude self
    restaurant_indices = [i[0] for i in sim_scores]
    return df.iloc[restaurant_indices][['Restaurant Name', 'Aggregate rating', 'City', 'Cuisines']]

print("\n Recommendation:")
print("Top 5 similar restaurants to the first one:")
print(recommend_restaurants(0))

# Save similarity matrix or something, but for now, just print
print("\n Task 2 completed successfully!")