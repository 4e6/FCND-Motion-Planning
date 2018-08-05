import time

from enum import Enum
from shapely.geometry import Polygon, Point, LineString
from sklearn.neighbors import KDTree
from queue import PriorityQueue
import networkx as nx
import numpy as np

def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    return valid_actions


def a_star(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost


def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))

def extract_polygons(data, safety_distance=1):

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

def collides0(polygons, point):
    """ Determine whether the point collides with any obstacles
    """
    for p, height in polygons:
        if p.contains(Point(point)) and height >= point[2]:
            return True
    return False

def collides(polygons_kd_tree, polygons_kd_list, point, kd_k=3):
    """ Determine whether the point collides with any obstacles
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
    xmin = np.min(data[:, 0] - data[:, 3])
    xmax = np.max(data[:, 0] + data[:, 3])

    ymin = np.min(data[:, 1] - data[:, 4])
    ymax = np.max(data[:, 1] + data[:, 4])

    zmin = target_altitude
    zmax = target_altitude + 4*safety_distance

    print("sample points", xmin, xmax, ymin, ymax, zmin, zmax)
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
        if not collides(polygons_kd_tree, polygons_kd_list, point, kd_k=10):
            to_keep.append(point)

    return to_keep

def point_near(points, point):
    points_kd_tree = KDTree([(p[0], p[1]) for p in points])
    inds = points_kd_tree.query([point], k=1, return_distance=False)
    return points[inds[0][0]]

def points_between(c1, c2, num):
    p1 = (c1[0], c1[1])
    p2 = (c2[0], c2[1])
    line = LineString([p1, p2])
    nums = np.linspace(0.0, 1.0, num)
    points = [line.interpolate(x, normalized=True) for x in nums]
    return [(p.x, p.y) for p in points]

def can_connect(n1, n2, polygons):
    p1 = (n1[0], n1[1])
    p2 = (n2[0], n2[1])
    line = LineString([p1, p2])
    for p, _ in polygons:
        if p.crosses(line):
            return False
    return True

def create_graph(nodes, polygons, k=5):
    G = nx.Graph()

    nodes_tree = KDTree([(n[0], n[1]) for n in nodes])
    polygons_coords = [(c[0], c[1]) for _, c in polygons]
    polygons_tree = KDTree(polygons_coords)

    for node in nodes:
        node_inds = nodes_tree.query([(node[0], node[1])], k=k, return_distance=False)[0]
        #p1_inds = polygons_tree.query([node], k=12*k, return_distance=False)[0]
        for ni in node_inds[1:]:
            node_i = nodes[ni]
            pts = points_between(node, node_i, num=11)
            poly_inds_list = polygons_tree.query(pts, k=10, return_distance=False)
            poly_inds = set().union(*poly_inds_list)
            polys = [polygons[i] for i in poly_inds]
            if can_connect(node, node_i, polys):
                G.add_edge(node, node_i, weight=heuristic(node, node_i))

    return G

def a_star_graph(graph, heuristic, start, goal):
    """Modified A* to work with NetworkX graphs."""

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

    return path[::-1], path_cost
