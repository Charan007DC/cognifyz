import pandas as pd
import numpy as np
import folium
from folium import plugins
import warnings
warnings.filterwarnings('ignore')

print("Starting Cognifyz Task 4: Location-based Analysis")
print("1. Loading dataset...")
df = pd.read_csv('Data/Dataset .csv')

geo_cols = [
    'City', 'Locality', 'Longitude', 'Latitude',
    'Cuisines', 'Aggregate rating', 'Price range', 'Votes', 'Restaurant Name'
]
df_geo = df[geo_cols].copy()

df_geo['Latitude'] = pd.to_numeric(df_geo['Latitude'], errors='coerce')
df_geo['Longitude'] = pd.to_numeric(df_geo['Longitude'], errors='coerce')
df_geo['Aggregate rating'] = pd.to_numeric(df_geo['Aggregate rating'], errors='coerce')
df_geo['Price range'] = pd.to_numeric(df_geo['Price range'], errors='coerce')
df_geo['Votes'] = pd.to_numeric(df_geo['Votes'], errors='coerce')

df_geo = df_geo.dropna(subset=['Latitude', 'Longitude', 'Aggregate rating'])
df_geo = df_geo[df_geo['Aggregate rating'] > 0]

print(f"2. Cleaned data: {len(df_geo)} valid restaurant locations")

print("3. Generating interactive map...")

center_lat = df_geo['Latitude'].mean()
center_lon = df_geo['Longitude'].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5,
    tiles='CartoDB positron'
)

heat_data = [[row['Latitude'], row['Longitude']] for idx, row in df_geo.iterrows()]
plugins.HeatMap(heat_data, radius=12, blur=10, min_opacity=0.4).add_to(m)

top_rest = df_geo.nlargest(500, 'Aggregate rating')
for idx, row in top_rest.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=3,
        color='green' if row['Aggregate rating'] >= 4 else 'orange' if row['Aggregate rating'] >= 3 else 'red',
        fill=True,
        fill_opacity=0.6,
        popup=f"{row['Restaurant Name']}<br>Rating: {row['Aggregate rating']}<br>Locality: {row['Locality']}"
    ).add_to(m)

m.save('task4_restaurant_heatmap.html')
print(" Map saved as 'task4_restaurant_heatmap.html'")

print("4. Analyzing by city and locality...")

city_stats = df_geo.groupby('City').agg(
    restaurant_count=('City', 'size'),
    avg_rating=('Aggregate rating', 'mean'),
    avg_price_range=('Price range', 'mean'),
    total_votes=('Votes', 'sum')
).round(2).sort_values('restaurant_count', ascending=False)

df_geo['locality_clean'] = df_geo['Locality'].fillna('Unknown')
locality_stats = df_geo.groupby(['City', 'locality_clean']).agg(
    restaurant_count=('locality_clean', 'size'),
    avg_rating=('Aggregate rating', 'mean'),
    avg_price_range=('Price range', 'mean'),
    cuisines=('Cuisines', lambda x: ', '.join(set(', '.join(x.dropna()).split(', '))))
).round(2)

city_stats.to_csv('task4_city_analysis.csv')
locality_stats.to_csv('task4_locality_analysis.csv')

print("\n City-level insights (Top 10):")
print(city_stats.head(10))

print("\n5. Extracting cuisine trends per high-density area...")

top_cities = city_stats.head(5).index.tolist()
df_top = df_geo[df_geo['City'].isin(top_cities)].copy()

df_top = df_top.assign(cuisine_list=df_top['Cuisines'].str.split(', '))
df_exploded = df_top.explode('cuisine_list')
df_exploded['cuisine_list'] = df_exploded['cuisine_list'].str.strip()

cuisine_by_city = df_exploded.groupby(['City', 'cuisine_list']).size().reset_index(name='count')
cuisine_by_city = cuisine_by_city.sort_values(['City', 'count'], ascending=[True, False])

cuisine_by_city.to_csv('task4_cuisines_by_city.csv', index=False)
print(" Cuisine trends saved to 'task4_cuisines_by_city.csv'")

print("\n Key Findings:")
print(f"- Total restaurants analyzed: {len(df_geo)}")
print(f"- Top city by count: {city_stats.index[0]} ({city_stats.iloc[0]['restaurant_count']} restaurants)")
print(f"- Highest avg rating city: {city_stats['avg_rating'].idxmax()} ({city_stats['avg_rating'].max():.2f})")
print(f"- Most common locality: {locality_stats.index.get_level_values(1).value_counts().index[0]}")

print("\n Task 4 completed! Check the output files and HTML map.")