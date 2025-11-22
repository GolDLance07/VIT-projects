# clustering.py

from typing import Dict, List
import numpy as np
from sklearn.cluster import KMeans

from graph import Graph


def cluster_locations(graph: Graph, n_clusters: int = 3) -> Dict[int, List[str]]:
    """
    Cluster locations into groups using K-Means on (x, y) coordinates.
    Returns: dict cluster_id -> list of node names
    """
    if len(graph.locations) == 0:
        return {}

    names = list(graph.locations.keys())
    coords = np.array([graph.get_location_coordinates(name) for name in names])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(coords)

    clusters: Dict[int, List[str]] = {}
    for name, label in zip(names, labels):
        clusters.setdefault(int(label), []).append(name)

    return clusters
