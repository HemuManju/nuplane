from collections import deque

import numpy as np
import networkx as nx

import osmnx as ox

from nuplane.utils.transform import (
    bezier_curve,
    vec_haversine_distance,
    linear_refine_implicit,
)

from experiments.taxiing.control.controllers import angle_diff


class PathPlanner(object):
    """Path planner based on the skeleton of the image.
    Generates a spline path
    """

    def __init__(self, env_map):
        self.map = env_map
        self.G = env_map.get_node_graph()
        self.route_deque = None
        self.distance_tol = 1.0
        return None

    def find_path(self, start, end, n_splits=1, smooth_path=False):
        """Finds a path between start and end using path graph

        Parameters
        ----------
        start : array
            A cartesian co-ordinate specifying the start position
        end : array
            A node ID specifying the end position
        n_splits : int, optional
            Number of splits in refining the path points, by default 1

        Returns
        -------
        path_points : array
            A refined path points in pybullet cartesian co-ordinate system
        """
        x = []
        y = []

        if not isinstance(start, (int, np.int64)):
            start_lat_lon = self.map.get_node_info(start)
            start = self.G.nearest_nodes(self.G, X=start_lat_lon[1], Y=start_lat_lon[0])
        if not isinstance(end, (int, np.int64)):
            end_lat_lon = self.map.get_node_info(end)
            end = self.G.nearest_nodes(self.G, X=end_lat_lon[1], Y=end_lat_lon[0])

        route = nx.shortest_path(self.G, start, end, weight='length')
        for node in route:
            x.append(self.G.nodes[node]['x'])
            y.append(self.G.nodes[node]['y'])

        lat_lon = np.array((x, y)).T

        if smooth_path:
            refined_points = linear_refine_implicit(lat_lon, n=n_splits)

            # Get the number of intersection per node
            streets_per_node = ox.stats.count_streets_per_node(self.G)
            weights = []
            for node in route:
                if streets_per_node[node] > 2:
                    weights.append(100)
                else:
                    weights.append(1)
            weights[0], weights[-1] = 2, 2
            # Curve fitting
            xvals, yvals = bezier_curve(lat_lon, weights, n_points=500)

            # For bezier, we need to flip the array along axis=0
            refined_points = np.flip(np.vstack((xvals, yvals)).T, axis=0)
        else:
            refined_points = lat_lon

        # Exchange x and y as they are reversed in xplane
        refined_points[:, [1, 0]] = refined_points[:, [0, 1]]

        return refined_points

    def _process_route(self, route):
        lat1, long1 = route[0:-1, 0], route[0:-1, 1]
        lat2, long2 = route[1:, 0], route[1:, 1]
        tol = vec_haversine_distance(lat1, long1, lat2, long2)
        return route

    def set_route(self, route):
        if self.route_deque is None:
            # Set up distance tolerance
            lat1, long1 = route[0:-1, 0], route[0:-1, 1]
            lat2, long2 = route[1:, 0], route[1:, 1]
            tol = vec_haversine_distance(lat1, long1, lat2, long2)
            self.distance_tol = min(5.0, np.min(tol))

            # Setup the route_deque
            self.route_deque = deque(self._process_route(route))
            self.route_deque.popleft()

            # Setup the next pose
            self.next_pos = self.route_deque[0]

    def update_route_points(self, observation):
        if observation is not None:
            # Update the path points
            self.distance = vec_haversine_distance(
                observation['latitude'],
                observation['longitude'],
                self.next_pos[0],
                self.next_pos[1],
            )

            if self.distance < self.distance_tol:
                self.route_deque.popleft()

            # Get heading error
            heading = self.get_heading(observation, self.next_pos)
            heading_error = angle_diff(
                heading * np.pi / 180, observation['true_psi'] * np.pi / 180
            )
            if abs(heading_error) > np.pi / 2:
                self.route_deque.popleft()

    def get_next_position(self, observation):
        # Start executing the action
        # TODO: Need to better next position search
        if len(self.route_deque) > 0:
            # Update route points
            self.update_route_points(observation)

            # Next pose
            try:
                self.next_pos = self.route_deque[0]
            except IndexError:
                self.next_pos = None

        return self.next_pos

    def get_heading(self, observation, next_pos=None):
        if next_pos is None:
            next_pos = self.get_next_position(observation)
        return ox.bearing.calculate_bearing(
            observation['latitude'], observation['longitude'], next_pos[0], next_pos[1]
        )

    def get_orientation(self):
        return ox.bearing.calculate_bearing(
            self.route_deque[0][0],
            self.route_deque[0][1],
            self.route_deque[1][0],
            self.route_deque[1][1],
        )

    def get_distance_to_next_pos(self):
        return self.distance

    def get_done_status(self):
        if len(self.route_deque) <= 1 or self.next_pos is None:
            return True
        else:
            return False
