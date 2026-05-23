import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Ultimate TSP Simulation Dashboard",
    layout="wide"
)

# -------------------------------------------------------------
# TITLE
# -------------------------------------------------------------
st.title("🧬 The Ultimate Traveling Salesman Dashboard")
st.write(
    "Compare computational complexity, accuracy benchmarks, "
    "and run real-time flight route simulations across Europe."
)

# -------------------------------------------------------------
# MASTER DATABASE OF CITIES
# -------------------------------------------------------------
MASTER_CITIES = {
    'Paris (France)': (48.8566, 2.3522),
    'Brussels (Belgium)': (50.8503, 4.3517),
    'Amsterdam (Netherlands)': (52.3676, 4.9041),
    'Berlin (Germany)': (52.5200, 13.4050),
    'Prague (Czechia)': (50.0755, 14.4378),
    'Rome (Italy)': (41.9028, 12.4964),
    'Madrid (Spain)': (40.4168, -3.7038),
    'Vienna (Austria)': (48.2082, 16.3738),
    'Bern (Switzerland)': (46.9480, 7.4474),
    'London (UK)': (51.5074, -0.1278),
    'Lisbon (Portugal)': (38.7223, -9.1393),
    'Dublin (Ireland)': (53.3498, -6.2603),
    'Copenhagen (Denmark)': (55.6761, 12.5683),
    'Warsaw (Poland)': (52.2297, 21.0122)
}

# -------------------------------------------------------------
# DISTANCE FUNCTIONS
# -------------------------------------------------------------
def distance(c1, c2, city_dict):
    """
    Haversine distance between two cities.
    """
    lat1, lon1 = np.radians(city_dict[c1])
    lat2, lon2 = np.radians(city_dict[c2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 6371.0 * (2 * np.arcsin(np.sqrt(a)))


def route_distance(route, city_dict):
    """
    Total route distance.
    """
    if not route or len(route) < 2:
        return 0

    total = 0

    for i in range(len(route) - 1):
        total += distance(route[i], route[i + 1], city_dict)

    return total


# -------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------
if "sim_running" not in st.session_state:
    st.session_state.sim_running = False

if "current_step" not in st.session_state:
    st.session_state.current_step = 1

if "previous_algo" not in st.session_state:
    st.session_state.previous_algo = ""


# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:

    st.header("🗺️ Network Workspace")

    selected_cities = st.multiselect(
        "Select Cities to Include in Route:",
        options=list(MASTER_CITIES.keys()),
        default=list(MASTER_CITIES.keys())[:7]
    )

    if len(selected_cities) < 3:
        st.error("Please select at least 3 cities.")
        st.stop()

    start_city = st.selectbox(
        "Select Starting Hub:",
        options=selected_cities
    )


# -------------------------------------------------------------
# ACTIVE CITY SET
# -------------------------------------------------------------
ACTIVE_CITIES = {
    city: MASTER_CITIES[city]
    for city in selected_cities
}

cities_list = list(ACTIVE_CITIES.keys())

other_cities = [
    city for city in cities_list
    if city != start_city
]


# -------------------------------------------------------------
# GREEDY SOLVER
# -------------------------------------------------------------
def solve_greedy():

    route = [start_city]

    unvisited = other_cities.copy()

    while unvisited:

        closest = min(
            unvisited,
            key=lambda city: distance(
                route[-1],
                city,
                ACTIVE_CITIES
            )
        )

        route.append(closest)

        unvisited.remove(closest)

    route.append(start_city)

    return route


# -------------------------------------------------------------
# DYNAMIC PROGRAMMING SOLVER
# -------------------------------------------------------------
def solve_dynamic():

    n = len(cities_list)

    if n > 15:
        return None

    mapping = {
        city: i
        for i, city in enumerate(cities_list)
    }

    inv_mapping = {
        i: city
        for i, city in enumerate(cities_list)
    }

    start_idx = mapping[start_city]

    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = distance(
                inv_mapping[i],
                inv_mapping[j],
                ACTIVE_CITIES
            )

    memo = {}

    def hk(mask, u):

        if mask == (1 << n) - 1:
            return dist_matrix[u][start_idx], [start_idx]

        if (mask, u) in memo:
            return memo[(mask, u)]

        best_cost = float('inf')
        best_path = []

        for v in range(n):

            if not (mask & (1 << v)):

                cost, path = hk(mask | (1 << v), v)

                total = dist_matrix[u][v] + cost

                if total < best_cost:
                    best_cost = total
                    best_path = [v] + path

        memo[(mask, u)] = (best_cost, best_path)

        return memo[(mask, u)]

    _, path_indices = hk(1 << start_idx, start_idx)

    final_route = [start_city]

    for idx in path_indices:
        final_route.append(inv_mapping[idx])

    return final_route


# -------------------------------------------------------------
# BACKTRACKING SOLVER
# -------------------------------------------------------------
def solve_backtracking():

    n = len(cities_list)

    if n > 10:
        return None

    best = {
        "dist": float('inf'),
        "path": []
    }

    def backtrack(curr, visited, path, curr_dist):

        if curr_dist >= best["dist"]:
            return

        if len(visited) == n:

            total = curr_dist + distance(
                curr,
                start_city,
                ACTIVE_CITIES
            )

            if total < best["dist"]:

                best["dist"] = total
                best["path"] = path + [start_city]

            return

        for nxt in other_cities:

            if nxt not in visited:

                visited.add(nxt)

                leg = distance(
                    curr,
                    nxt,
                    ACTIVE_CITIES
                )

                backtrack(
                    nxt,
                    visited,
                    path + [nxt],
                    curr_dist + leg
                )

                visited.remove(nxt)

    backtrack(
        start_city,
        {start_city},
        [start_city],
        0
    )

    return best["path"]


# -------------------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------------------
col_control, col_display = st.columns([1, 2])

# -------------------------------------------------------------
# CONTROLS
# -------------------------------------------------------------
with col_control:

    st.subheader("⚙️ Control Settings")

    algo = st.selectbox(
        "Choose Active Evaluation Strategy:",
        [
            "Greedy (Fast / Heuristic)",
            "Dynamic Programming (Held-Karp Exact)",
            "Backtracking (Exhaustive Search Exact)"
        ]
    )

    if algo != st.session_state.previous_algo:

        st.session_state.sim_running = False
        st.session_state.current_step = 1
        st.session_state.previous_algo = algo

    run_valid = True

    if (
        algo == "Dynamic Programming (Held-Karp Exact)"
        and len(selected_cities) > 15
    ):
        st.error("⚠️ Dynamic Programming max = 15 cities.")
        run_valid = False

    if (
        algo == "Backtracking (Exhaustive Search Exact)"
        and len(selected_cities) > 10
    ):
        st.error("⚠️ Backtracking max = 10 cities.")
        run_valid = False

    st.markdown("---")

    st.subheader("🎬 Simulation Control")

    speed = st.slider(
        "Step Interval (seconds)",
        0.05,
        2.0,
        0.3,
        0.05
    )

    trigger_sim = st.button(
        "▶️ Launch Live Simulation",
        disabled=not run_valid
    )


# -------------------------------------------------------------
# RUN SIMULATION
# -------------------------------------------------------------
if trigger_sim and run_valid:

    st.session_state.sim_running = True
    st.session_state.current_step = 1


# -------------------------------------------------------------
# SOLVE ROUTE
# -------------------------------------------------------------
route = []
total_dist = 0
execution_ms = 0

if run_valid:

    start = time.perf_counter()

    if algo == "Greedy (Fast / Heuristic)":
        route = solve_greedy()

    elif algo == "Dynamic Programming (Held-Karp Exact)":
        route = solve_dynamic()

    else:
        route = solve_backtracking()

    execution_ms = (time.perf_counter() - start) * 1000

    total_dist = route_distance(route, ACTIVE_CITIES)


# -------------------------------------------------------------
# DISPLAY PANEL
# -------------------------------------------------------------
with col_display:

    st.subheader("📊 Algorithmic Benchmarks")

    c1, c2 = st.columns(2)

    if route:

        c1.metric(
            "Calculated Loop Distance",
            f"{total_dist:.2f} km"
        )

        c2.metric(
            "Compute Processing Velocity",
            f"{execution_ms:.2f} ms"
        )

    else:

        c1.metric("Calculated Loop Distance", "N/A")
        c2.metric("Compute Processing Velocity", "N/A")

    st.markdown("---")

    st.subheader("🗺️ Live Route Map")

    # ---------------------------------------------------------
    # FIXED MAP CODE
    # ---------------------------------------------------------
    if route:

        latitudes = []
        longitudes = []
        labels = []

        for city in route:

            lat, lon = ACTIVE_CITIES[city]

            latitudes.append(lat)
            longitudes.append(lon)
            labels.append(city)

        df = pd.DataFrame({
            "city": labels,
            "lat": latitudes,
            "lon": longitudes
        })

        fig = go.Figure()

        # Route lines
        fig.add_trace(
            go.Scattermapbox(
                lat=latitudes,
                lon=longitudes,
                mode="lines+markers",
                marker=dict(
                    size=12
                ),
                text=labels,
                line=dict(
                    width=4
                ),
                hoverinfo="text"
            )
        )

        # Map layout
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=3.4,
            mapbox_center={
                "lat": 50,
                "lon": 10
            },
            height=700,
            margin={
                "r": 0,
                "t": 0,
                "l": 0,
                "b": 0
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -----------------------------------------------------
        # ROUTE TABLE
        # -----------------------------------------------------
        st.markdown("### 📍 Final Route")

        route_table = pd.DataFrame({
            "Step": list(range(1, len(route) + 1)),
            "City": route
        })

        st.dataframe(
            route_table,
            use_container_width=True
        )

    else:

        st.warning("No route generated.")
