# Cognifyz Internship – Task 4: Location-based Analysis

> **"Where Data Meets Intelligence"**  
> Machine Learning Internship @ [Cognifyz Technologies](https://www.cognifyz.com)

This project performs a comprehensive **geospatial and statistical analysis** of restaurant distribution, ratings, cuisines, and pricing across cities and localities using the provided restaurant dataset.

---

## 🎯 Objective

- Visualize restaurant density across geographic regions using an interactive heatmap.
- Analyze restaurant concentration by **city** and **locality**.
- Compute key metrics: average ratings, price ranges, and cuisine trends per location.
- Extract actionable insights about regional dining preferences.

---

## 📂 Project Structure
cognifyz_task4/
├── Dataset .csv # Original dataset (provided by Cognifyz)
├── cognifyz_task4_location_analysis.py # Main analysis script
├── task4_restaurant_heatmap.html # Interactive heatmap (output)
├── task4_city_analysis.csv # City-level statistics
├── task4_locality_analysis.csv # Locality-level statistics
├── task4_cuisines_by_city.csv # Top cuisines per city
└── README.md # This file

## ⚙️ Requirements

- **Python 3.14.0** (as per your environment)
- **Operating System**: Windows (compatible with macOS/Linux too)
- **Libraries**:
  - `pandas`
  - `numpy`
  - `folium`
  - (Optional: `webbrowser` to auto-open map)

Install dependencies via:
```bash
pip install pandas numpy folium
