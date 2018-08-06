import time

from enum import Enum
from shapely.geometry import Polygon, Point, LineString
from sklearn.neighbors import KDTree
from queue import PriorityQueue
import networkx as nx
import numpy as np


def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))

def extract_polygons(data, safety_distance=1):
    """
    Return a list of pairs of a polygon with its coordinates
    """

    polygons = []
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        obstacle = [
            north - d_north - safety_distance,
            north + d_north + safety_distance,
            east - d_east - safety_distance,
            east + d_east + safety_distance ]
        corners = [
            (obstacle[0], obstacle[2]),
            (obstacle[0], obstacle[3]),
            (obstacle[1], obstacle[3]),
            (obstacle[1], obstacle[2]) ]

        height = alt + d_alt + safety_distance

        p = Polygon(corners)
        # polygons.append((p, height))
        polygons.append((p, (north, east, height)))

    return polygons

def collides(polygons_kd_tree, polygons_kd_list, point, kd_k=3):
    """
    Determine whether the point collides with any obstacles
    """

    collides = False
    inds = polygons_kd_tree.query([(point[0], point[1])], k=kd_k, return_distance=False)
    for i in inds[0]:
        poly, (_, _, z) = polygons_kd_list[i]
        pt = Point(point[0], point[1])
        if poly.contains(pt) and z > point[2]:
            collides = True
            break

    return collides

def sample_points(data, target_altitude, safety_distance, num_samples=100):
    """
    Sample points randomly filtering out ones that collides with the obstacles
    """

    xmin = np.min(data[:, 0] - data[:, 3])
    xmax = np.max(data[:, 0] + data[:, 3])

    ymin = np.min(data[:, 1] - data[:, 4])
    ymax = np.max(data[:, 1] + data[:, 4])

    zmin = target_altitude + 1*safety_distance
    zmax = target_altitude + 2.5*safety_distance

    #print("sample points", xmin, xmax, ymin, ymax, zmin, zmax)
    xvals = np.random.uniform(xmin, xmax, num_samples)
    yvals = np.random.uniform(ymin, ymax, num_samples)
    zvals = np.random.uniform(zmin, zmax, num_samples)

    #samples = np.array(list(zip(xvals, yvals, zvals)))
    samples = zip(xvals, yvals, zvals)

    polygons_kd_list = extract_polygons(data)
    polygons_kd_points = [(p[0], p[1]) for _, p in polygons_kd_list]
    polygons_kd_tree = KDTree(polygons_kd_points)

    to_keep = []
    for point in samples:
        if not collides(polygons_kd_tree, polygons_kd_list, point, kd_k=20):
            to_keep.append(point)

    return to_keep

def point_2d(p): return (p[0], p[1])

def point_near(points, point):
    """
    Return closest `point` from a list of `points`
    """
    points_kd_tree = KDTree(points)
    inds = points_kd_tree.query([point], k=1, return_distance=False)
    return points[inds[0][0]]

def points_between(c1, c2, num):
    """
    Return points on a line between the `c1` and `c2`
    """
    p1 = point_2d(c1)
    p2 = point_2d(c2)
    pz = min(c1[2], c2[2])
    line = LineString([p1, p2])
    nums = np.linspace(0.0, 1.0, num)
    points = [line.interpolate(x, normalized=True) for x in nums]
    return [(p.x, p.y, pz) for p in points]

def can_connect(n1, n2, polygons):
    """
    Check if we can draw the line between points 'n1' and 'n2' that does not
    intersect with 'polygons'
    """
    p1 = point_2d(n1)
    p2 = point_2d(n2)
    pz = min(n1[2], n2[2])
    line = LineString([p1, p2])
    for p, (_, _, h) in polygons:
        if p.intersects(line) and pz < h:
            return False
    return True

def create_graph(nodes, polygons, k=5):
    """
    Create a graph from nodes, checking that edges does not collide with polygons
    """
    G = nx.Graph()

    nodes_tree = KDTree([point_2d(c) for c in nodes])
    polygons_tree = KDTree([point_2d(c) for _, c in polygons])

    for node in nodes:
        node_inds = nodes_tree.query([point_2d(node)], k=k, return_distance=False)[0]
        #p1_inds = polygons_tree.query([node], k=12*k, return_distance=False)[0]
        for ni in node_inds[1:]:
            node_i = nodes[ni]
            pts = points_between(node, node_i, num=21)
            poly_inds_list = polygons_tree.query([point_2d(p) for p in pts], k=30, return_distance=False)
            poly_inds = set().union(*poly_inds_list)
            polys = [polygons[i] for i in poly_inds]
            if can_connect(node, node_i, polys):
                G.add_edge(node, node_i, weight=heuristic(node, node_i))

    return G

def a_star_graph(graph, heuristic, start, goal):
    """
    Modified A* algorithm to work with NetworkX graphs.
    """

    path = []
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                new_cost = current_cost + cost + heuristic(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node))

                    branch[next_node] = (new_cost, current_node)

    path = []
    path_cost = 0
    if found:

        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])

        rev = path[::-1]
        rev.append(goal)
        return rev, path_cost
    else:
        print('PATH NOT FOUND!')
        return None, None
