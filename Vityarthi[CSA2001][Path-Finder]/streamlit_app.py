import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from collections import deque
import heapq
import random

# =====================================================
#  GRAPH DATA STRUCTURE
# =====================================================
class Graph:
    def __init__(self):
        self.nodes = {}   # name -> (x, y)
        self.edges = {}   # name -> list of (neighbor, base_cost)

    def add_node(self, name, x, y):
        self.nodes[name] = (x, y)
        if name not in self.edges:
            self.edges[name] = []

    def add_edge(self, u, v):
        if u not in self.nodes or v not in self.nodes:
            return
        dist = math.dist(self.nodes[u], self.nodes[v])
        self.edges.setdefault(u, []).append((v, dist))
        self.edges.setdefault(v, []).append((u, dist))


# =====================================================
#  SEARCH ALGORITHMS
# =====================================================
def reconstruct_path(came, start, goal):
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came.get(node)
    path.reverse()
    return path if path and path[0] == start else []


def bfs(graph: Graph, start: str, goal: str):
    queue = deque([start])
    visited = {start}
    came = {start: None}

    while queue:
        node = queue.popleft()
        if node == goal:
            return reconstruct_path(came, start, goal)
        for neigh, _ in graph.edges.get(node, []):
            if neigh not in visited:
                visited.add(neigh)
                came[neigh] = node
                queue.append(neigh)
    return []


def dfs(graph: Graph, start: str, goal: str):
    stack = [start]
    visited = {start}
    came = {start: None}

    while stack:
        node = stack.pop()
        if node == goal:
            return reconstruct_path(came, start, goal)
        for neigh, _ in graph.edges.get(node, []):
            if neigh not in visited:
                visited.add(neigh)
                came[neigh] = node
                stack.append(neigh)
    return []


def heuristic_distance(graph: Graph, a: str, b: str, mode: str = "Euclidean") -> float:
    (x1, y1) = graph.nodes[a]
    (x2, y2) = graph.nodes[b]
    dx, dy = abs(x1 - x2), abs(y1 - y2)
    if mode == "Manhattan":
        return dx + dy
    elif mode == "Chebyshev":
        return max(dx, dy)
    # default Euclidean
    return math.dist((x1, y1), (x2, y2))


def astar(graph: Graph, start: str, goal: str,
          heuristic_mode: str = "Euclidean",
          route_profile: str = "Shortest Distance",
          traffic: dict | None = None):
    open_set = [(0, start)]
    came = {start: None}
    gscore = {start: 0.0}

    if traffic is None:
        traffic = {n: 1 for n in graph.nodes}

    max_traffic = max(traffic.values()) if traffic else 1

    def edge_cost(u, v, base_cost):
        # adjust cost based on route profile + traffic
        t_val = traffic.get(u, 1) + traffic.get(v, 1)
        if route_profile == "Avoid Crowded":
            extra = t_val * 0.1
        elif route_profile == "Prefer Crowded":
            extra = (max_traffic * 2 - t_val) * 0.05
        else:  # Shortest Distance
            extra = 0.0
        return base_cost + extra

    while open_set:
        _, node = heapq.heappop(open_set)
        if node == goal:
            return reconstruct_path(came, start, goal)

        for neigh, base_cost in graph.edges.get(node, []):
            cost = edge_cost(node, neigh, base_cost)
            temp_g = gscore[node] + cost

            if temp_g < gscore.get(neigh, float("inf")):
                gscore[neigh] = temp_g
                came[neigh] = node
                h = heuristic_distance(graph, neigh, goal, heuristic_mode)
                f = temp_g + h
                heapq.heappush(open_set, (f, neigh))

    return []


# =====================================================
#  HELPERS
# =====================================================
def path_distance(graph: Graph, path: list[str]) -> float:
    total = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        for neigh, cost in graph.edges.get(u, []):
            if neigh == v:
                total += cost
                break
    return total


def path_directions(graph: Graph, path: list[str]) -> list[str]:
    """Returns human-readable directions for each step."""
    directions = []
    if len(path) < 2:
        return directions

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        (x1, y1) = graph.nodes[u]
        (x2, y2) = graph.nodes[v]
        dx, dy = x2 - x1, y2 - y1

        if abs(dx) < 0.1 and dy > 0:
            dir_text = "North"
        elif abs(dx) < 0.1 and dy < 0:
            dir_text = "South"
        elif abs(dy) < 0.1 and dx > 0:
            dir_text = "East"
        elif abs(dy) < 0.1 and dx < 0:
            dir_text = "West"
        elif dx > 0 and dy > 0:
            dir_text = "North-East"
        elif dx < 0 and dy > 0:
            dir_text = "North-West"
        elif dx > 0 and dy < 0:
            dir_text = "South-East"
        else:
            dir_text = "South-West"

        directions.append(f"{u} → {v} ({dir_text})")
    return directions


def degree(graph: Graph, node: str) -> int:
    return len(graph.edges.get(node, []))


def build_campus_graph() -> tuple[Graph, dict]:
    g = Graph()
    locations = {
        "Hostel": (0, 0),
        "Library": (1, 2),
        "Main Gate": (3, 1),
        "Cafeteria": (4, 3),
        "Academic Block": (6, 1)
    }
    for name, (x, y) in locations.items():
        g.add_node(name, x, y)

    edges = [
        ("Hostel", "Library"),
        ("Hostel", "Main Gate"),
        ("Library", "Cafeteria"),
        ("Main Gate", "Cafeteria"),
        ("Main Gate", "Academic Block"),
        ("Cafeteria", "Academic Block")
    ]
    for u, v in edges:
        g.add_edge(u, v)

    # simple "traffic" levels per node
    traffic = {
        "Hostel": 2,
        "Library": 3,
        "Main Gate": 4,
        "Cafeteria": 5,
        "Academic Block": 3
    }
    return g, traffic


def build_random_graph(n_nodes: int = 6, width: int = 10, height: int = 10) -> tuple[Graph, dict]:
    g = Graph()
    traffic = {}
    for i in range(n_nodes):
        name = f"Node{i+1}"
        x, y = random.uniform(0, width), random.uniform(0, height)
        g.add_node(name, x, y)
        traffic[name] = random.randint(1, 5)

    node_names = list(g.nodes.keys())
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < 0.4:  # 40% chance of edge
                g.add_edge(node_names[i], node_names[j])

    return g, traffic


# =====================================================
#  STREAMLIT UI
# =====================================================
st.set_page_config(page_title="AI Path Finder & Smart Route Analyzer", layout="wide")
st.title("🧭 AI Path Finder & Smart Route Analyzer")

st.write(
    "This app demonstrates **BFS, DFS, A***, clustering, and path history on a small map. "
    "You can switch between a campus demo and a random graph."
)
st.markdown("---")

# keep state
if "graph_mode" not in st.session_state:
    st.session_state.graph_mode = "Campus Demo"
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts

# =====================================================
#  SIDEBAR CONTROLS
# =====================================================
with st.sidebar:
    st.header("⚙ Settings")

    graph_mode = st.radio("Map Type", ["Campus Demo", "Random Graph"], index=0)
    st.session_state.graph_mode = graph_mode

    if graph_mode == "Random Graph":
        n_nodes = st.slider("Number of nodes", 4, 10, 6)
    else:
        n_nodes = 5  # ignored for campus

    route_profile = st.selectbox(
        "Route Profile",
        ["Shortest Distance", "Avoid Crowded", "Prefer Crowded"]
    )

    heuristic_mode = st.selectbox(
        "A* Heuristic",
        ["Euclidean", "Manhattan", "Chebyshev"]
    )

    show_bfs = st.checkbox("Show BFS path", True)
    show_dfs = st.checkbox("Show DFS path", True)
    show_astar = st.checkbox("Show A* path", True)

# =====================================================
#  BUILD SELECTED GRAPH
# =====================================================
if st.session_state.graph_mode == "Campus Demo":
    graph, traffic = build_campus_graph()
else:
    graph, traffic = build_random_graph(n_nodes=n_nodes)

node_names = list(graph.nodes.keys())

st.subheader("🗺 Choose Your Route")
col1, col2 = st.columns(2)

with col1:
    start = st.selectbox("Where are you **starting from**?", node_names, index=0)
with col2:
    goal = st.selectbox("Where do you want to **go**?", node_names, index=min(1, len(node_names)-1))

st.markdown("### 📍 Available Locations")
for name, (x, y) in graph.nodes.items():
    st.write(f"• **{name}** — Coordinates ({x:.1f}, {y:.1f}), Degree: {degree(graph, name)}")


# =====================================================
#  RUN PATHFINDING
# =====================================================
if st.button("Find Route"):
    paths = {}

    if show_bfs:
        paths["BFS"] = bfs(graph, start, goal)
    if show_dfs:
        paths["DFS"] = dfs(graph, start, goal)
    if show_astar:
        paths["A*"] = astar(graph, start, goal, heuristic_mode, route_profile, traffic)

    # Display textual results
    for algo_name, path in paths.items():
        if not path:
            st.warning(f"{algo_name}: No path found.")
        else:
            dist = path_distance(graph, path)
            st.success(f"{algo_name} path: {' → '.join(path)}  |  Distance = {dist:.2f}")
            # Add to history
            st.session_state.history.append({
                "Algorithm": algo_name,
                "Start": start,
                "Goal": goal,
                "Distance": round(dist, 2)
            })

            # Directions (only show for A* to avoid clutter, but you can change)
            if algo_name == "A*":
                st.markdown("**Step-by-step directions (A*):**")
                for step in path_directions(graph, path):
                    st.write("•", step)

    # =====================================================
    #  MAP VISUALIZATION WITH MULTI-PATHS
    # =====================================================
    st.subheader("🗺 Visual Map with Paths")

    fig, ax = plt.subplots()

    # base node sizes by degree
    max_deg = max((degree(graph, n) for n in graph.nodes), default=1)
    for name, (x, y) in graph.nodes.items():
        deg = degree(graph, name)
        size = 100 + (deg / max_deg) * 200  # bigger for higher degree
        ax.scatter(x, y, s=size, color="skyblue", edgecolor="black", zorder=3)
        ax.text(x + 0.1, y + 0.1, name, fontsize=11)

    # plot edges
    for u in graph.edges:
        for v, _ in graph.edges[u]:
            x1, y1 = graph.nodes[u]
            x2, y2 = graph.nodes[v]
            ax.plot([x1, x2], [y1, y2], color="gray", alpha=0.4, zorder=1)

    # draw each path in different color
    colors = {"BFS": "green", "DFS": "orange", "A*": "red"}
    for algo_name, path in paths.items():
        if len(path) > 1:
            xs = [graph.nodes[p][0] for p in path]
            ys = [graph.nodes[p][1] for p in path]
            ax.plot(xs, ys, color=colors.get(algo_name, "black"),
                    linewidth=3, label=f"{algo_name} path", zorder=4)

    ax.set_title("Map View")
    ax.legend()
    st.pyplot(fig)

    # =====================================================
    #  CLUSTERING VISUALIZATION (unchanged idea)
    # =====================================================
    st.subheader("🔵 K-Means Clustering of Locations")

    coords = np.array([graph.nodes[n] for n in graph.nodes])
    if len(coords) >= 2:
        n_clusters = min(3, len(coords))  # avoid k > n
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(coords)
        labels = kmeans.labels_
        fig2, ax2 = plt.subplots()
        for i, name in enumerate(graph.nodes.keys()):
            x, y = coords[i]
            ax2.scatter(x, y, s=120, c=f"C{labels[i]}", edgecolor="black")
            ax2.text(x + 0.1, y + 0.1, name, fontsize=11)
        ax2.set_title("Location Clusters")
        st.pyplot(fig2)
    else:
        st.info("Not enough locations for clustering.")

    # =====================================================
    #  SIMPLE LINEAR REGRESSION PREDICTION (as before)
    # =====================================================
    st.subheader("⏱ Simple Travel Time Prediction")

    train_dist = np.array([[1], [2], [3], [4], [5]])
    train_time = np.array([2, 4.1, 6.2, 8.1, 10.2])

    model = LinearRegression()
    model.fit(train_dist, train_time)

    # use A* distance if available, else BFS, else DFS
    chosen_dist = None
    for key in ["A*", "BFS", "DFS"]:
        if key in paths and paths[key]:
            chosen_dist = path_distance(graph, paths[key])
            break

    if chosen_dist is not None:
        predicted = model.predict(np.array([[chosen_dist]]))[0]
        st.success(f"Predicted travel time for distance {chosen_dist:.2f}: **{predicted:.2f} minutes**")
    else:
        st.info("No valid path found to estimate time.")


# =====================================================
#  PATH HISTORY
# =====================================================
st.markdown("---")
st.subheader("📜 Path History (This Session)")
if st.session_state.history:
    st.dataframe(st.session_state.history)
else:
    st.write("No paths searched yet.")
