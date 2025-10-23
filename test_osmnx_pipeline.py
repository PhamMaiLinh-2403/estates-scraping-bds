import pandas as pd
import random
from src.cleaning import DataImputer  # adjust if module is under src/
import osmnx as ox

# ===============================
# 1. Load dataset
# ===============================
DATA_PATH = "C:\\Users\\ADMIN\\Downloads\\mightymerge.io__6mwa3qmi\\listing_details_1.csv"  # adjust if needed

print(f"Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# Ensure we have latitude/longitude columns
if "latitude" not in df.columns or "longitude" not in df.columns:
    raise ValueError("Dataset must have 'latitude' and 'longitude' columns!")

# Drop NaN coordinates
df_valid = df.dropna(subset=["latitude", "longitude"]).copy()
print(f"Total listings with valid coordinates: {len(df_valid)}")

if df_valid.empty:
    raise ValueError("No valid coordinates found in dataset.")

# Randomly sample a few rows to avoid API throttling
sample_rows = df_valid.sample(min(10, len(df_valid)), random_state=42)

# ===============================
# 2. Modify DataImputer for debugging
# ===============================
def debug_query_osmnx_for_roads(lat, lon, radius=300):
    print(f"\n--- OSMnx fetch start for ({lat}, {lon}) ---")
    G = DataImputer._get_cached_graph(lat, lon, radius)
    if G is None:
        print("❌ Graph fetch failed (likely network or Overpass issue).")
        return None

    try:
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        print(f"✅ Graph fetched with {len(edges)} edges.")
        print(f"Highway types found: {edges['highway'].unique()}")
        # Relaxed filter for testing
        major_roads = edges[edges['highway'].isin([
            'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
            'residential', 'unclassified', 'service'
        ])]
        print(f"Filtered to {len(major_roads)} major roads.")
        if major_roads.empty:
            print("⚠️ No major roads found in radius.")
        else:
            print("✅ Major roads available.")
        return major_roads
    except Exception as e:
        print(f"❌ Failed to convert graph to GeoDataFrame: {e}")
        return None

# Temporarily patch method for debugging
DataImputer._query_osmnx_for_roads = staticmethod(debug_query_osmnx_for_roads)

# ===============================
# 3. Test pipeline on sample rows
# ===============================
for idx, row in sample_rows.iterrows():
    lat, lon = float(row["latitude"]), float(row["longitude"])
    print(f"\n========== Testing listing {idx} ==========")
    print(f"Coordinates: lat={lat}, lon={lon}")

    try:
        result = DataImputer.fill_missing_distance_to_the_main_road({
            "latitude": lat,
            "longitude": lon,
            "Khoảng cách tới trục đường chính (m)": None
        })
        print(f"➡️  Final imputed distance: {result} m")
    except Exception as e:
        print(f"❌ Exception during imputation: {e}")

print("\n✅ Test completed.")
