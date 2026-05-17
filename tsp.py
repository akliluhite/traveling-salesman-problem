import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
import itertools

st.title("Traveling Salesman Problem (TSP) Visualizer")
st.write("Comparing algorithms across 5 European Countries.")

# Country coordinates
CITIES = {
    'Paris (France)': (48.8566, 2.3522),
    'Brussels (Belgium)': (50.8503, 4.3517),
    'Amsterdam (Netherlands)': (52.3676, 4.9041),
    'Berlin (Germany)': (52.5200, 13.4050),
    'Prague (Czechia)': (50.0755, 14.4378)
}

def distance(c1, c2):
    lat1, lon1 = np.radians(CITIES[c1])
    lat2, lon2 = np.radians(CITIES[c2])
    return 6371.0 * (2 * np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2)))

# Algorithm Selection
algo = st.selectbox("Choose TSP Algorithm", ["Nearest Neighbor (Greedy)", "Brute Force (Optimal)"])

route = ['Paris (France)']
if algo == "Nearest Neighbor (Greedy)":
    unvisited = list(CITIES.keys())
    unvisited.remove(route[0])
    while unvisited:
        closest = min(unvisited, key=lambda c: distance(route[-1], c))
        route.append(closest)
        unvisited.remove(closest)
    route.append(route[0])
else:
    best_dist = float('inf')
    for p in itertools.permutations(list(CITIES.keys())[1:]):
        r = ['Paris (France)'] + list(p) + ['Paris (France)']
        d = sum(distance(r[i], r[i+1]) for i in range(5))
        if d < best_dist:
            best_dist, route = d, r

# Calculate total route distance
total_dist = sum(distance(route[i], route[i+1]) for i in range(5))
st.metric(label="Total Journey Distance", value=f"{total_dist:.2f} km")

# Plot the map layout
df = pd.DataFrame([{"City": c, "Lat": CITIES[c][0], "Lon": CITIES[c][1]} for c in route])
fig = px.line_geo(df, lat="Lat", lon="Lon", hover_name="City", text="City", projection="natural earth")
fig.update_geos(center=dict(lat=51, lon=8), projection_scale=6)
st.plotly_chart(fig)
