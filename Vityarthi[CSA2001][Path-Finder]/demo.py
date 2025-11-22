# demo.py

from graph import Graph
from search_algorithms import bfs, dfs, astar
from clustering import cluster_locations
from regression_model import TravelTimeRegressor
import math


def build_sample_graph() -> Graph:
    """
    Build a small example graph with locations and edges.
    You can replace this with your own campus/map data.
    """
    g = Graph()

    # Add locations (name, x, y)
    g.add_location("A", 0, 0)
    g.add_location("B", 1, 2)
    g.add_location("C", 3, 1)
    g.add_location("D", 4, 3)
    g.add_location("E", 6, 1)

    # Add edges (like roads with distance as cost)
    def dist(u, v):
        (x1, y1) = g.get_location_coordinates(u)
        (x2, y2) = g.get_location_coordinates(v)
        return math.dist((x1, y1), (x2, y2))

    g.add_edge("A", "B", dist("A", "B"))
    g.add_edge("A", "C", dist("A", "C"))
    g.add_edge("B", "D", dist("B", "D"))
    g.add_edge("C", "D", dist("C", "D"))
    g.add_edge("C", "E", dist("C", "E"))
    g.add_edge("D", "E", dist("D", "E"))

    return g


def path_distance(g: Graph, path):
    """Sum of edge costs along the path."""
    if len(path) < 2:
        return 0.0

    total = 0.0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        # find cost of edge u->v
        for neigh, cost in g.get_neighbors(u):
            if neigh == v:
                total += cost
                break
    return total


if __name__ == "__main__":
    graph = build_sample_graph()
    start, goal = "A", "E"

    print("Graph:", graph)

    # 1. PATHFINDING
    bfs_path = bfs(graph, start, goal)
    dfs_path = dfs(graph, start, goal)
    astar_path = astar(graph, start, goal)

    print(f"\nBFS path {start}->{goal}: {bfs_path}  | distance = {path_distance(graph, bfs_path):.2f}")
    print(f"DFS path {start}->{goal}: {dfs_path}  | distance = {path_distance(graph, dfs_path):.2f}")
    print(f"A*  path {start}->{goal}: {astar_path} | distance = {path_distance(graph, astar_path):.2f}")

    # 2. CLUSTERING
    clusters = cluster_locations(graph, n_clusters=2)
    print("\nLocation clusters:")
    for cid, nodes in clusters.items():
        print(f"  Cluster {cid}: {nodes}")

    # 3. LINEAR REGRESSION (fake training data)
    # Example: (distance, time in minutes)
    training_data = [
        (1.0, 2.0),
        (2.0, 4.1),
        (3.0, 6.2),
        (4.0, 8.1),
        (5.0, 10.2),
    ]

    reg = TravelTimeRegressor()
    reg.train(training_data)

    distance = path_distance(graph, astar_path)
    predicted_time = reg.predict_time(distance)
    print(f"\nPredicted travel time for distance {distance:.2f} units: {predicted_time:.2f} minutes")
