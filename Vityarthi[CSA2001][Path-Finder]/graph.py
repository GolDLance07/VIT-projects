# graph.py

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Location:
    """Represents a node in the map."""
    name: str
    x: float
    y: float


class Graph:
    def __init__(self):
        # adjacency list: node -> list of (neighbor, cost)
        self.adj_list: Dict[str, List[Tuple[str, float]]] = {}
        self.locations: Dict[str, Location] = {}

    def add_location(self, name: str, x: float, y: float):
        """Add a location (node) to the graph."""
        self.locations[name] = Location(name, x, y)
        if name not in self.adj_list:
            self.adj_list[name] = []

    def add_edge(self, src: str, dest: str, cost: float = 1.0, bidirectional: bool = True):
        """Add an edge between two locations."""
        if src not in self.adj_list:
            self.adj_list[src] = []
        if dest not in self.adj_list:
            self.adj_list[dest] = []

        self.adj_list[src].append((dest, cost))
        if bidirectional:
            self.adj_list[dest].append((src, cost))

    def get_neighbors(self, node: str) -> List[Tuple[str, float]]:
        return self.adj_list.get(node, [])

    def get_location_coordinates(self, name: str) -> Tuple[float, float]:
        loc = self.locations[name]
        return loc.x, loc.y

    def __repr__(self):
        return f"Graph(nodes={list(self.locations.keys())})"
