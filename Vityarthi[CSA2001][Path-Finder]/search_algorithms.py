# search_algorithms.py

from typing import Dict, List, Tuple, Optional
from collections import deque
import heapq
import math

from graph import Graph


def reconstruct_path(came_from: Dict[str, Optional[str]], start: str, goal: str) -> List[str]:
    """Reconstruct path from start to goal using parent pointers."""
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    path.reverse()

    if path and path[0] == start:
        return path
    return []  # no valid path


def bfs(graph: Graph, start: str, goal: str) -> List[str]:
    """Breadth-First Search for shortest path in unweighted graph."""
    queue = deque([start])
    visited = {start}
    came_from: Dict[str, Optional[str]] = {start: None}

    while queue:
        node = queue.popleft()
        if node == goal:
            return reconstruct_path(came_from, start, goal)

        for neighbor, _cost in graph.get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = node
                queue.append(neighbor)

    return []  # no path


def dfs(graph: Graph, start: str, goal: str) -> List[str]:
    """Depth-First Search (not guaranteed shortest)."""
    stack = [start]
    visited = {start}
    came_from: Dict[str, Optional[str]] = {start: None}

    while stack:
        node = stack.pop()
        if node == goal:
            return reconstruct_path(came_from, start, goal)

        for neighbor, _cost in graph.get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = node
                stack.append(neighbor)

    return []


def euclidean_heuristic(graph: Graph, node: str, goal: str) -> float:
    """Heuristic for A*: Euclidean distance between node and goal."""
    x1, y1 = graph.get_location_coordinates(node)
    x2, y2 = graph.get_location_coordinates(goal)
    return math.dist((x1, y1), (x2, y2))


def astar(graph: Graph, start: str, goal: str) -> List[str]:
    """A* Search: optimal path using cost + heuristic."""
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from: Dict[str, Optional[str]] = {start: None}
    g_score: Dict[str, float] = {start: 0.0}

    while open_set:
        _f, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, start, goal)

        for neighbor, cost in graph.get_neighbors(current):
            tentative_g = g_score[current] + cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + euclidean_heuristic(graph, neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return []
