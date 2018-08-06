import argparse
import time
import msgpack
from enum import Enum, auto

import networkx as nx
import numpy as np

import planning_utils as pu
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local, local_to_global


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.1:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        print(self.target_position)
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING

        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE
        # Set the global home location from the first line of 'colliders.csv'
        print("setting position from colliders.csv ...")
        with open('colliders.csv') as f:
            first_line = f.readline()
            coords = [c for coord in first_line.split(',') for c in coord.split()]
            self.set_home_position(float(coords[3]), float(coords[1]), self.global_position[2])
        print('global home {0}, position {1}, local position {2}'.format(
            self.global_home,
            self.global_position,
            self.local_position))

        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        print("extracting polygons ...")
        t = time.time()
        polygons = pu.extract_polygons(data, SAFETY_DISTANCE)
        print("in {0} seconds".format(time.time() - t))

        print("sampling points ...")
        t = time.time()
        samples = pu.sample_points(data, TARGET_ALTITUDE, SAFETY_DISTANCE, num_samples=1200)
        # Define some points of interest, we'll use them as a goal locations
        point_local_position = (int(self.local_position[0]), int(self.local_position[1]), TARGET_ALTITUDE)
        # Harry Bridges Plaza
        point_middle_right = (400, 350, TARGET_ALTITUDE)
        # Top left dead end
        point_top_left = (550, -340, TARGET_ALTITUDE)
        # Bottom right square
        point_bottom_right = (-280, 400, TARGET_ALTITUDE)
        samples.append(point_local_position)
        samples.append(point_middle_right)
        samples.append(point_top_left)
        samples.append(point_bottom_right)
        print("{0} samples in {1} seconds".format(len(samples), time.time() - t))

        print("creating a graph ...")
        t = time.time()
        g = pu.create_graph(samples, polygons, k=25)
        print("{1} edges in {0} seconds".format(time.time() - t, len(g.edges)))

        # Since randomly sampled graph may have partitions (disconnected regions),
        # here we taking the component of the graph which has
        # the largest number connected nodes
        max_connected = list(max(nx.connected_components(g),key=len))
        #k = np.random.randint(len(max_connected))

        # Set goal as one of the points of interest
        start = pu.point_near(max_connected, point_local_position)
        goal = pu.point_near(max_connected, point_middle_right)
        # goal1 = pu.point_near(max_connected, point_top_left)
        # goal2 = pu.point_near(max_connected, point_bottom_right)

        # Run A* to find a path from start to goal
        print("searching for a path ...")
        t = time.time()
        path, _ = pu.a_star_graph(g, pu.heuristic, start, goal)
        # path1, _ = pu.a_star_graph(g, pu.heuristic, goal, goal1)
        # path2, _ = pu.a_star_graph(g, pu.heuristic, goal1, goal2)
        # path3, _ = pu.a_star_graph(g, pu.heuristic, goal2, start)
        print("in {0} seconds".format(time.time() - t))
        print("found {2} step path from {0} to {1}".format(start, goal, len(path)))

        # Convert path to waypoints
        waypoints = [[int(p[0]), int(p[1]), int(p[2]), 0] for p in path]
        # waypoints = [[int(p[0]), int(p[1]), int(p[2]), 0] for p in path + path1 + path2 + path3]
        # Set self.waypoints
        self.waypoints = waypoints
        #print(waypoints)
        # send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
