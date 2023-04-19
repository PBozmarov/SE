# Dummy component placement environment implemented
# using OpenAI Gym and NumPy.

import gym  # type: ignore
import heapq
import numpy as np
import random
import sys
from typing import Dict, Tuple, Set, List, Any
from ray import tune
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog
import ray
from ray.rllib.utils.framework import try_import_tf
_, tf, _ = try_import_tf()
tfkl = tf.keras.layers


# import matplotlib  # type: ignore
# from matplotlib import pyplot as plt
# import matplotlib.colors  # type: ignore
# from matplotlib.colors import to_rgba  # type: ignore
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # type: ignore
# from matplotlib.figure import Figure  # type: ignore
# from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget  # type: ignore


class Pin(object):
    """A pin to be added to a component.
    Args:
        relative_x (int): the x position of the pin relative to the top-left
                        corner of the component
        relative_y (int): the y position of the pin relative to the top-left
                        corner of the component
        pin_id (int): the id of the pin
        component_id (int): the id of the component the pin belongs to
        net_id (int): the id of the net the pin is assigned to
    """

    def __init__(
        self,
        relative_x: int,
        relative_y: int,
        pin_id: int,
        component_id: int,
        net_id: int,
    ):
        self.relative_x = relative_x
        self.relative_y = relative_y
        self.absolute_x = -1
        self.absolute_y = -1
        self.pin_id = pin_id
        self.component_id = component_id
        self.net_id = net_id

    def update_pin_position(self, component_x: int, component_y: int):
        """Updates the absolute position of the pin based on the position of the
        top-left corner of the component that the pin is placed on.

        Args:
            component_x (int): the x position of the top-left corner of the
                            component that the pin belongs to
            component_y (int): the y position of the top-left corner of the
                            component that the pin belongs to
        """
        self.absolute_x = component_x + self.relative_x
        self.absolute_y = component_y + self.relative_y

    def calculate_feature(
        self, component_x: int, component_y: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the feature vector for the pin.

        Args:
            component_x (int): the x position of the top-left corner of the
                            component that the pin belongs to
            component_y (int): the y position of the top-left corner of the
                            component that the pin belongs to

        Returns:
            Tuple[np.ndarray, np.ndarray] : the numerical feature vector for the pin, the categorical
                        feature vector for the pin
        """
        if(not (component_x == -1 or component_y == -1)):
            self.update_pin_position(component_x, component_y)

        num_features = np.array(
            [
                self.relative_x,
                self.relative_y,
                self.absolute_x,
                self.absolute_y,
            ]
        )

        cat_features = np.array(
            [
                self.net_id,
                self.component_id,
            ]
        )

        return num_features, cat_features

    def __str__(self) -> str:
        """Returns a string representation of the pin."""
        return f"Pin at ({self.absolute_x, self.absolute_y}) for component {self.component_id} assigned to net {self.net_id}"

    def __repr__(self) -> str:
        """Returns a string representation of the pin."""
        return self.__str__()


class Component(object):
    """A component to be placed on the grid.

    Args:
        h (int): the height of the component
        w (int): the width of the component
        comp_id (int): the id of the component
        pins (List[Pin]): the pins to be added to the component
        placed (bool): whether the component has been placed on the grid. Defaults
                    to False.
    """

    def __init__(
        self, h: int, w: int, comp_id: int, pins: List[Pin], placed: bool = False
    ):
        self.h = h
        self.w = w
        self.area = h * w
        self.comp_id = comp_id
        self.placed = placed
        self.position = (-1, -1)
        self.pins = pins

    def place_component(self, orientation: int, x: int, y: int):
        """Places the component on the grid. Updates the relative and absolute
        positions of the pins based on the orientation of the component.

        Args:
            orientation (int): the orientation of the component. 0 is the original
                            orientation, 1 is 90 degrees clockwise, 2 is 180 degrees
                            clockwise, and 3 is 270 degrees clockwise
            x (int): the x position of the top-left corner of the component
            y (int): the y position of the top-left corner of the component
        """
        self.placed = True
        self.position = (x, y)

        # Update the positions of pins
        if orientation == 0:  # original orientation
            for pin in self.pins:
                pin.update_pin_position(x, y)
        elif orientation == 1:  # 90 degrees clockwise
            for pin in self.pins:
                pin.relative_x, pin.relative_y = (
                    pin.relative_y,
                    self.h - pin.relative_x - 1,
                )
                pin.update_pin_position(x, y)
        elif orientation == 2:  # 180 degrees clockwise
            for pin in self.pins:
                pin.relative_x, pin.relative_y = (
                    self.h - pin.relative_x - 1,
                    self.w - pin.relative_y - 1,
                )
                pin.update_pin_position(x, y)
        elif orientation == 3:  # 270 degrees clockwise
            for pin in self.pins:
                pin.relative_x, pin.relative_y = (
                    self.w - pin.relative_y - 1,
                    pin.relative_x,
                )
                pin.update_pin_position(x, y)

    def area_ratio(self, grid_area: int) -> float:
        """Returns the ratio of the component area to the grid area.

        Args:
            grid_area (int): total area of the grid

        Returns:
            float: the ratio of the component area to the grid area
        """
        return self.area / grid_area

    def calculate_feature(self, grid_area: int, max_num_pins_per_component: int) -> np.ndarray:
        """Calculates the feature vector for the component.

        Args:
            grid_area (int): the area of the grid
            max_num_pins_per_component (int): the maximum number of pins per component

        Returns:
            np.ndarray: the feature vector for the component
        """
        component_x, component_y = self.position
        component_area_ratio = self.area_ratio(grid_area)

        component_feature = np.array(
            [self.h, self.w, component_x, component_y, component_area_ratio],
        )

        # create an array of size max_num_pins_per_component with all values being -1
        all_pin_ids = np.full(max_num_pins_per_component, -1)
        # get a array of pin ids from the pins
        pin_ids = np.array([pin.pin_id for pin in self.pins])
        # replace the first len(pin_ids) values in all_pin_ids with the pin ids
        all_pin_ids[: len(pin_ids)] = pin_ids

        # extend the feature vector with the pin ids
        component_feature = np.append(component_feature, all_pin_ids)

        return component_feature

    def __str__(self) -> str:
        """Returns a string representation of the component."""
        return f"Component {self.comp_id}: {self.h} x {self.w}"

    def __repr__(self) -> str:
        """Returns a string representation of the component."""
        return self.__str__()


class DummyPlacementEnv(gym.Env):
    """A NumPy-based dummy environment for component placement.

    The environment is a grid of size height x width. The agent can place
    components on the grid, where the components are rectangular arrays with
    dimensions randomly generated between min_component_h x min_component_w.
    The environment is terminated when it is not possible to place any more
    components on the grid.

    - observation: a dictionary with keys: "grid", "action_mask",
    "all_components_feature", "all_pins_num_feature", "all_pins_cat_feature",
    "component_mask", "placement_mask", "pin_mask", "pin_placement_mask".
        - "grid" is a NumPy array of shape (height, width) with dtype
            np.float64. The array contains the current state of the grid.
            Each cell contains an integer 0 or 1, where 0 means the cell
            is empty and 1 means the cell is occupied.
        - "action_mask" is a NumPy array of shape (4, height, width) with
            dtype np.float64. The array contains a mask of valid actions.
            Each cell contains an integer 0 or 1, where 0 means the action
            is invalid and 1 means the action is valid.
        - "all_components_feature" is a NumPy array of shape
            (max_num_components, 5 + max_num_pins_per_component). 
            The array contains the features of all components and the ids of pins
            on the components. 
            Each row contains the features of a component.
            The features are (component_h, component_w,
            component_position_h, component_position_w, component_area_ratio,
            pin_ids).
        - "all_pins_num_feature" is a NumPy array of shape
            (max_num_components*max_num_pins_per_component, 4). The array contains
            the numerical features of all pins. Each row contains the features of a pin.
            The features are (relative_x, relative_y, absolute_x, absolute_y).
        - "all_pins_cat_feature" is a NumPy array of shape
            (max_num_components*max_num_pins_per_component, 2). The array contains
            the categorical features of all pins. Each row contains the features of a pin.
            The features are (net_id, component_id).
        - "component_mask" is a NumPy array of shape (max_num_components,).
            The array contains a mask of valid components. Each cell contains
            an integer 0 or 1, where 0 means the component is invalid and 1
            means the component is valid.
        - "placement_mask" is a NumPy array of shape (max_num_components,).
            The array contains a mask of placed components. Each cell contains
            an integer 0 or 1, where 0 means the component is not placed and 1
            means the component is placed.
        - "pin_mask" is a NumPy array of shape
            (max_num_components*max_num_pins_per_component,). The array contains
            a mask of valid pins. Each cell contains an integer 0 or 1, where
            0 means the pin is invalid and 1 means the pin is valid.
        - "pin_placement_mask" is a NumPy array of shape
            (max_num_components*max_num_pins_per_component,). The array contains
            a mask of placed pins. Each cell contains an integer 0 or 1, where
            0 means the pin is not placed and 1 means the pin is placed.

    - action: a tuple (orientation, x, y) where x and y are integers in the
        range [0, height) and [0, width) respectively, and represent the
        coordinates of the top-left corner of the component once it's placed.
        Orientation is an integer of value from [0, 1, 2, 3], where 0 means
        the component is placed in its original orientation (not rotated),
        1 means the component is rotated 90 degrees clockwise,
        2 means the component is rotated 180 degrees clockwise,
        and 3 means the component is rotated 270 degrees clockwise.

    - reward: integer 0 or 1
        The reward is 1 for each time step where a component is placed on the
        grid, and 0 otherwise.

    - episode termination:
        The episode is terminated when it is not possible to place any more
        components on the grid.
    """

    def __init__(  # noqa: max-complexity: 15
        self,
        height: int,
        width: int,
        net_distribution: int,
        pin_spread: int,
        min_component_w: int,
        max_component_w: int,
        min_component_h: int,
        max_component_h: int,
        max_num_components: int,
        min_num_components: int,
        min_num_nets: int,
        max_num_nets: int,
        max_num_pins_per_net: int,
        min_num_pins_per_net: int = 2,
        reward_type: str = "both",
        reward_beam_width: int = 2,
        weight_wirelength: float = 0.5,
    ):
        """Instantiates a new DummyPlacementEnv.

        Args:
            height (int): height of the grid
            width (int): width of the grid
            net_distribution (int): distribution of the nets
            pin_spread (int): spread of the pins
            min_component_w (int): minimum component width
            max_component_w (int): maximum component width
            min_component_h (int): minimum component height
            max_component_h (int): maximum component height
            max_num_components (int): maximum number of components
            to place on the grid
            min_num_components (int): minimum number of components
            to place on the grid
            min_num_nets (int): minimum number of nets of pins
            max_num_nets (int): maximum number of nets of pins
            max_num_pins_per_net (int): maximum number of pins per net
            min_num_pins_per_net (int): minimum number of pins per net.
            Defaults to 2
        """
        # Initialize the environment
        self.height = height
        self.width = width
        self.area = height * width

        # Clip comlexity to [0, 9]
        self.net_distribution = max(0, min(9, net_distribution))
        self.pin_spread = max(0, min(9, pin_spread))

        # Initialize component attributes
        self.min_component_w = min_component_w
        self.max_component_w = max_component_w
        self.min_component_h = min_component_h
        self.max_component_h = max_component_h
        self.max_num_components = max_num_components
        self.min_num_components = min_num_components
        self.min_num_nets = min_num_nets
        self.max_num_nets = max_num_nets
        self.max_num_pins_per_net = max_num_pins_per_net
        self.min_num_pins_per_net = min_num_pins_per_net
        self.max_num_pins_per_component = self.max_component_h * self.max_component_w
        self.action_space = gym.spaces.Tuple(
            [
                gym.spaces.Discrete(4),
                gym.spaces.Discrete(height),
                gym.spaces.Discrete(width),
            ]
        )
        self.pins: List[Pin] = []
        self.reward_type = reward_type
        self.reward_beam_width = reward_beam_width
        self.weight_wirelength = weight_wirelength
        self.max_num_intersections = self.calculate_upper_bound_intersections()
        self.max_wirelength = self.calculate_upper_bound_wirelength()

        # Initialize parameters for components of the reward
        self.reward_intersection = -1   # dummy value to be updated after reward is calculated
        self.reward_wirelength = -1

        # Initialize the observation space
        self.observation_space = gym.spaces.Dict(
            {
                "grid": gym.spaces.Box(
                    low=0, high=1, shape=(height, width), dtype=np.float64
                ),
                "pin_grid": gym.spaces.Box(
                    low=0, high=100, shape=(height, width, 2), dtype=np.float64
                ),
                "action_mask": gym.spaces.Box(
                    low=0, high=1, shape=(4, height, width), dtype=np.float64
                ),
                "all_components_feature": gym.spaces.Box(
                    low=-1,
                    high= max(self.max_num_pins_per_net * self.max_num_nets, self.height*self.width),
                    shape=(self.max_num_components, 5 + self.max_num_pins_per_component),
                    dtype=np.float64,
                ),
                "all_pins_num_feature": gym.spaces.Box(
                    low=-1,
                    high=max(height, width),
                    shape=(
                        self.max_num_components * self.max_num_pins_per_component+1,
                        4,
                    ),
                    dtype=np.float64,
                ),
                "all_pins_cat_feature": gym.spaces.Box(
                    low=-1,
                    high=max(self.max_num_components, self.max_num_nets),
                    shape=(
                        self.max_num_components * self.max_num_pins_per_component+1,
                        2,
                    ),
                    dtype=np.int32,
                ),
                "component_mask": gym.spaces.Box(
                    low=0, high=1, shape=(max_num_components,), dtype=np.float64
                ),
                "placement_mask": gym.spaces.Box(
                    low=-1, high=10, shape=(max_num_components,), dtype=np.float64
                ),
                "pin_mask": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_num_components * self.max_num_pins_per_component,),
                    dtype=np.float64,
                ),
                "pin_placement_mask": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_num_components * self.max_num_pins_per_component,),
                    dtype=np.float64,
                ),
                # "current_component_id": gym.spaces.Discrete(max_num_components, start=-1),
                # "max_num_nets": gym.spaces.Discrete(max_num_nets+1),
                # "current_component_id": gym.spaces.Box(
                #     low=-1, high=max_num_components, shape=(1,), dtype=np.int32
                # ),
                # "max_num_nets": gym.spaces.Box(
                #     low=0, high=max_num_nets, shape=(1,)
                # )
            }
        )

        # Initialize the environment state
        self.components: List[Component] = []
        self.grid: np.ndarray = np.zeros((height, width))
        self.pin_grid: np.ndarray = np.zeros((height, width, 2))
        self.actions: List[Tuple[int, int, int]] = []
        self.action_mask: np.ndarray = np.zeros((4, height, width))
        self.placement_mask = np.zeros(self.max_num_components)
        self.component_mask = np.zeros(self.max_num_components)
        self.pin_mask = np.zeros(
            self.max_num_components * self.max_num_pins_per_component
        )
        self.pin_placement_mask = np.zeros(
            self.max_num_components * self.max_num_pins_per_component
        )

        # Intialize all components feature which contains
        # (component_h, component_w, component_position_h, component_position_w,
        # area_ratio, list of pin_id's)
        # Initialize all pins feature which contains
        # (relative_x, relative_y, absolute_x, absolute_y, net_id, component_id)
        self.all_components_feature: np.ndarray = np.zeros(
            (self.max_num_components, 5 + self.max_num_pins_per_component),
            dtype=np.float64,
        )
        self.all_pins_num_feature: np.ndarray = np.zeros(
            (self.max_num_components*self.max_num_pins_per_component + 1, 4)
        )
        self.all_pins_cat_feature: np.ndarray = np.zeros(
            (self.max_num_components*self.max_num_pins_per_component + 1, 2)
        )
        self.all_pins_cat_feature[-1,:] = np.array([-1, -1])

        # Initialize the current component id
        self.current_component_id = -1

        # Validate the environment parameters
        if self.height < 0 or self.width < 0:
            raise ValueError("Grid size must be greater than 0.")
        if self.max_component_w > self.height or self.max_component_h > self.width:
            raise ValueError(
                "Component size must be less than or equal to the grid size."
            )
        if self.min_component_w < 1 or self.min_component_h < 1:
            raise ValueError("Component size must be greater than 0.")
        if max_num_components < 1 or max_num_components > self.area:
            raise ValueError(
                "Number of components must be greater than 0"
                + "and less than or equal to the grid area."
            )
        if self.reward_type not in ["beam", "centroid", "both"]:
            raise ValueError("Invalid type of routing for the reward")
        if (
            self.reward_beam_width < 2
            or self.reward_beam_width > self.max_num_pins_per_net
        ):
            raise ValueError(
                "The beam width must be an integer greater than 2 and at most the maximum number of pins per net."
            )
        if not isinstance(self.reward_beam_width, int):
            raise ValueError("The beam width must be an integer.")
        if not isinstance(self.weight_wirelength, float):
            raise ValueError("The weight of wirelength must be a float.")
        if self.weight_wirelength < 0:
            raise ValueError("The weight of wirelength must be greater than 0.")

    def lowest_num_intersections(
        self, routes: List[List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]]
    ) -> Tuple[int, int]:
        """Finds the lowest number of intersections in a list of routes.

        Args:
            routes (List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]): a list of
            routes where each route is a list of list of tuples where each list
            is a net and each tuple is a line segment defined by two points.

        Returns:
            Tuple[int, int]: a tuple of the lowest number of intersections and index of the route
        """
        num_intersections = []
        for route in routes:
            num_intersections.append(self.find_num_intersection(route))
        return min(num_intersections), num_intersections.index(min(num_intersections))

    def find_num_intersection(  # noqa: max-complexity: 15
        self, route: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]
    ) -> int:
        """Finds the number of intersections in a route.

        Args:
            route (List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]): a route
            where each route is a list of list of tuples where each list is a net
            and each tuple is a line segment defined by two points.

        Returns:
            int: number of intersections
        """
        num_intersections = 0
        for net in range(len(route)):
            for other_net in range(net + 1, len(route)):
                for line_segment_1 in route[net]:
                    for line_segment_2 in route[other_net]:
                        if self.is_intersect(line_segment_1, line_segment_2):
                            num_intersections += 1

        return num_intersections

    def is_intersect(
        self,
        line1: Tuple[Tuple[int, int], Tuple[int, int]],
        line2: Tuple[Tuple[int, int], Tuple[int, int]],
    ) -> bool:
        """Checks if two line segments intersect.

        Args:
            line1 (Tuple[Tuple[int, int], Tuple[int, int]]): a line segment defined
            by two points
            line2 (Tuple[Tuple[int, int], Tuple[int, int]]): a line segment defined
            by two points

        Returns:
            bool: True if the line segments intersect, False otherwise
        """
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]

        if (
            line1[0] == line2[0]
            or line1[0] == line2[1]
            or line1[1] == line2[0]
            or line1[1] == line2[1]
        ):
            return True

        # Calculate the determinant
        det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        # If the determinant is zero, there is no intersection
        if det == 0:
            return False

        # # Calculate the x and y coordinates of the intersection point
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

        # Check if the intersection point lies on both line segments
        if (
            min(x1, x2) <= x <= max(x1, x2)
            and min(x3, x4) <= x <= max(x3, x4)
            and min(y1, y2) <= y <= max(y1, y2)
            and min(y3, y4) <= y <= max(y3, y4)
        ):
            return True
        else:
            return False

    def find_wirelength(
        self, route: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]
    ) -> float:
        """Finds the total wirelength of a route.

        Args:
            route (List[Tuple[Tuple[int, int], Tuple[int, int]]]): a list of tuples
            where each tuple is a line segment defined by two points.

        Returns:
            float: total wirelength
        """
        wirelength = 0.0
        for net in route:
            for line in net:
                wirelength += self.euclidean_distance(line[0], line[1])
        return wirelength

    def calculate_upper_bound_wirelength(self) -> float:
        """Finds the upper bound on the total wirelength
        of a routing.

        To find the upper bound, we assume that the worst case scenario is
        when the pins for each net are evenly placed in opposite corners of the grid,
        with pins being able to be superimposed on each other.

        Returns:
            float: upper bound on wirelength
        """
        distance = self.euclidean_distance((0, 0), (self.height, self.width))
        total_distance = (
            0.5 * distance * (self.max_num_nets * self.max_num_pins_per_net)
        )
        return total_distance / (self.height + self.width)

    def calculate_upper_bound_intersections(self) -> float:
        """Finds the upper bound for the number of intersections
        in a routing.

        To find the upper bound, we assume that the worst case scenario is
        when we are using the centroid routing method and each
        net has the same centroid.

        Returns:
            float: upper bound on number of intersections
        """
        max_num_intersections = (
            0.5
            * (self.max_num_pins_per_net**2)
            * self.max_num_nets
            * (self.max_num_nets - 1)
        )
        return max_num_intersections

    def find_reward(self) -> float:
        """Finds the reward for an episode.

        The reward is calculated as the weighted negative sum of the wirelength and the
        number of intersections. The wirelength is normalized by the sum of the dimensions
        of the grid and the weight is given by self.weight_wirelength.

        To find the number of intersections, routing is done using the method
        specified by the self.reward_type attribute and then number of intersections for
        route is calculated. The self.reward_type parameter can be one of:
            - "beam": uses beam search to find the best route with self.reward_beam_width
            as the beam width and the starting point as the point returned by the
            self.pin_outlier method
            - "centroid": uses centroid routing to find the best route. The centroid
            method finds the centroid for each net and connects all the pins in that
            net to the centroid.
            - "both": uses both beam search and centroid routing and chooses the routing
            with the least number of intersections.

        For the case when not all the components have been placed in the episode,
        the reward is calculated as the negative number of components that have not
        been placed minus the upper bound on the wirelength and the upper bound on
        the number of intersections.

        Returns:
            float: reward for the current episode
        """

        placed_all = self.current_component.comp_id == -1
        normalizing_factor = self.height + self.width

        # Reward for when not all components have been placed

        if not placed_all:
            reward = (
                -1 * self.weight_wirelength * self.max_wirelength
                - self.max_num_intersections
            )

            # Update the different reward components for callbacks
            self.reward_intersection = self.max_num_intersections
            self.reward_wirelength = self.max_wirelength
            return reward

        # Calculate reward for when all components have been placed
        if self.reward_type == "beam":
            if self.reward_beam_width == 0:
                raise ValueError("Beam search is used but beam_k is 0.")
            else:
                # Find the routes using beam search
                route = self.route_pins_beam_search(self.reward_beam_width)
                num_intersections = self.find_num_intersection(route)

                # Find the wirelength and reward
                wirelength = self.find_wirelength(route) / normalizing_factor
                reward = -1 * (self.weight_wirelength * wirelength + num_intersections)

                # Update the different reward components for callbacks
                self.reward_intersection = num_intersections
                self.reward_wirelength = wirelength

        elif self.reward_type == "centroid":
            # Find the routes using centroid routing
            route = self.route_pins_centroid()
            num_intersections = self.find_num_intersection(route)

            # Find the wirelength and reward
            wirelength = self.find_wirelength(route) / normalizing_factor
            reward = -1 * (self.weight_wirelength*wirelength + 0.3*num_intersections) #num_intersections #(self.weight_wirelength * wirelength) + )

            # Update the different reward components for callbacks
            self.reward_intersection = num_intersections
            self.reward_wirelength = wirelength

        elif self.reward_type == "both":
            # Find the routes using beam search and centroid routing
            route_beam = self.route_pins_beam_search(self.reward_beam_width)
            route_centroid = self.route_pins_centroid()
            routes = [route_beam, route_centroid]

            # Find the min_num_intersections and wirelength
            min_num_intersections, index = self.lowest_num_intersections(routes)
            wirelength = self.find_wirelength(routes[index]) / normalizing_factor

            # Update the different reward components for callbacks
            self.reward_intersection = min_num_intersections
            self.reward_wirelength = wirelength

            # Find the reward depending on whether the episode is complete
            reward = -1 * (self.weight_wirelength * wirelength + min_num_intersections)
        return reward

    def generate_pins(self):
        """Generates a list of pins to be placed on the grid."""
        self.pins = []
        for i in range(self.total_num_pins):
            self.pins.append(Pin(-1, -1, i, -1, -1))

    def generate_components(self):
        """Generates a list of components to be placed on the grid."""
        components = []
        self.sample_num_components()
        for i in range(self.num_components):
            component_h = np.random.randint(
                self.min_component_h, self.max_component_h + 1
            )
            component_w = np.random.randint(
                self.min_component_w, self.max_component_w + 1
            )
            pins = []
            components.append(Component(component_h, component_w, i, pins))

        self.components = components
        self.total_area_covered_by_all_components = 0
        for component in self.components:
            self.total_area_covered_by_all_components += component.area

    def generate_instances(self):
        """Generates a placement instance in the following steps:

        1. Generates a list of components to be placed on the grid.
        2. Samples the number of nets.
        3. Samples the total number of pins.
        4. Generates a list of pins to be placed on the grid.
        5. Allocates pins to nets.
        6. Allocates pins to components.
        """
        self.generate_components()
        self.sample_num_nets()
        self.sample_total_num_pins()
        self.generate_pins()
        self.allocate_pins_to_nets()
        self.allocate_pins_to_components()

        components_pins = {component.comp_id: [] for component in self.components}
        for pin in self.pins:
            components_pins[pin.component_id].append(pin)

        for component in self.components:
            component.pins = components_pins[component.comp_id]
            self.place_pins_on_component(component)

    def sample_num_components(self):
        """Samples the number of components to be placed on the grid."""
        self.num_components = np.random.randint(
            self.min_num_components, self.max_num_components + 1
        )

    def sample_num_nets(self):
        """Samples the number of nets to be placed on the grid."""
        self.num_nets = np.random.randint(self.min_num_nets, self.max_num_nets + 1)
        self.num_nets = min(
            self.num_nets, int(self.total_area_covered_by_all_components / 2)
        )

    def sample_total_num_pins(self):
        """Samples the total number of pins to be placed on the grid."""
        total_num_pins = np.random.randint(
            self.min_num_pins_per_net * self.num_nets,
            self.max_num_pins_per_net * self.num_nets + 1,
        )

        self.total_num_pins = min(
            total_num_pins, self.total_area_covered_by_all_components
        )

    def allocate_pins_to_nets(self):
        """Allocates pins to nets."""
        probs_for_nets = np.random.normal(
            1 / self.num_nets, 1 / (self.net_distribution + 1), self.num_nets
        )
        # Apply softmax to probs_for_nets
        probs_for_nets = np.exp(probs_for_nets) / np.sum(np.exp(probs_for_nets))

        # An array of size num_nets, where each element is the number of
        # pins allocated to the corresponding net.
        net_allocation = np.random.multinomial(
            self.total_num_pins - 2 * self.num_nets, probs_for_nets
        )
        # Assign net ids to pins and create dictionary net_pins to store pins for each net
        pin_id = 0
        self.net_pins = {}

        for i in range(self.num_nets):
            for j in range(2):
                self.pins[j + pin_id].net_id = i
                self.pins[j + pin_id].absolute_x = -1
                self.pins[j + pin_id].absolute_y = -1
            pin_id += 2
            self.net_pins[i] = self.pins[pin_id - 2 : pin_id]

        for i in range(self.num_nets):
            for j in range(net_allocation[i]):
                self.pins[j + pin_id].net_id = i
                self.pins[j + pin_id].absolute_x = -1
                self.pins[j + pin_id].absolute_y = -1
            if net_allocation[i] > 0:
                pin_id += net_allocation[i]
                self.net_pins[i].extend(self.pins[pin_id - net_allocation[i] : pin_id])

    def allocate_pins_to_components(self):
        """Allocates pins to components."""
        # Number of components that each net will cover based on complexity (pin_spread)
        self.num_components_w_pins = min(
            int(((self.pin_spread) / 10) * self.num_components) + 1, self.num_components
        )
        # Create dictionary of components and number of available spaces for pins
        components_available_space = {}
        for component in self.components:
            components_available_space[component.comp_id] = self.components[
                component.comp_id
            ].area
        # Allocate pins to components
        for net in range(self.num_nets):
            components_available_space = self.allocate_pins_to_components_for_net(
                net, components_available_space, self.num_components_w_pins
            )
        # Update self.pins using self.net_pins
        self.pins = []
        for net in range(self.num_nets):
            self.pins.extend(self.net_pins[net])

    def allocate_pins_to_components_for_net(  # noqa: max-complexity: 15
        self,
        net: int,
        components_available_space: Dict[int, int],
        num_components_w_pins: int,
    ) -> Dict[int, int]:
        """Allocates pins to components for a given net.

        Args:
            net (int): The net to allocate pins to.
            components_available_space (Dict[int, int]): A dictionary of
                    components and the number of available spaces for pins.
            num_components_w_pins (int): The number of components to
                    allocate pins to.
        Returns:
            Dict[int, int]: A dictionary of components and the number of
                available spaces for pins.
        """
        num_pins_unassigned = len(self.net_pins[net])
        # Order components_available_space dictionary by number of available
        # spaces for pins
        components_available_space = dict(
            sorted(
                components_available_space.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        # Update num_components_w_pins to ensure that all pins are assigned
        total_components_space = 0
        num_components_w_pins -= 1
        while total_components_space < num_pins_unassigned:
            total_components_space = 0
            num_components_w_pins += 1
            components_to_assign_pins = list(components_available_space.keys())[
                :num_components_w_pins
            ]
            for component in components_to_assign_pins:
                component_space = components_available_space[component]
                total_components_space += component_space

        pin_id = 0
        while num_pins_unassigned > 0:
            # Take the first num_components_w_pins components and allocate pins to them
            components_to_assign_pins = list(components_available_space.keys())[
                :num_components_w_pins
            ]

            total_space_available = sum(
                components_available_space[component]
                for component in components_to_assign_pins
            )

            # Allocate pins in net to components proportionally to their available space
            num_pins_per_component = np.random.multinomial(
                num_pins_unassigned,
                np.array(
                    [
                        components_available_space[component] / total_space_available
                        for component in components_to_assign_pins
                    ]
                ),
            )
            for component, num_pins in zip(
                components_to_assign_pins, num_pins_per_component
            ):
                # If there are more pins to be assigned than available space,
                # assign all available space
                if components_available_space[component] < num_pins:
                    num_pins = components_available_space[component]
                components_available_space[component] -= num_pins

                # Assign component ids to pins
                for pin in range(num_pins):
                    self.net_pins[net][pin + pin_id].component_id = component
                pin_id += num_pins
                # update num_pins_unassigned
                num_pins_unassigned -= num_pins

        return components_available_space

    def get_net_pin_positions(self) -> Dict[int, List[Tuple[int, int]]]:
        """Returns a dictionary of net ids and their corresponding pin positions.
        Returns:
            Dict[int, List[Tuple[int, int]]]: A dictionary of net ids and their corresponding pin positions.
        """
        net_pin_positions = {}
        # Loop through pins for each net
        for net in self.net_pins.keys():
            # Create key value pair for net and pin positions.
            net_pin_positions[net] = [
                (pin.absolute_x, pin.absolute_y) for pin in self.net_pins[net]
            ]
        return net_pin_positions

    @staticmethod
    def get_centroid(points: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Returns the centroid of a set of points.

        Args:
            points (List[Tuple[int, int]]): A list of points.

        Returns:
            Tuple[int, int]: The centroid of the points.
        """
        points_arr = np.array(points)
        cx, cy = np.mean(points_arr, axis=0)
        return (cx, cy)

    def route_pins_centroid(
        self,
    ) -> List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """Routes pins to the centroid of the pins for each net.

        Returns:
            List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]: A list of net routes,
            where each net route is a list of tuples of line endpoints.
        """
        net_pin_positions = self.get_net_pin_positions()
        route_point_pairs = []

        # Loop through sets of pin positions for each net
        for positions in net_pin_positions.values():
            if len(positions) == 2:
                # If there are only two pins, route directly between them
                net_route = [(positions[0], positions[1])]

            else:
                # Get centroid
                centroid = self.get_centroid(positions)

                # Add each pair of line endpoints to net route
                net_route = [(position, centroid) for position in positions]
            route_point_pairs.append(net_route)

        return route_point_pairs

    def pin_outlier(self, pin_positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Returns the pin that is furthest away from the centroid of the centroid.

        Args:
            pin_positions (List[Tuple[int, int]]): A list of pin positions

        Returns:
            Tuple[int, int]: The pin that is furthest away from the centroid
        """
        centroid = self.get_centroid(pin_positions)
        distances = [
            np.linalg.norm(np.array(pin) - np.array(centroid)) for pin in pin_positions
        ]
        return pin_positions[np.argmax(distances)]

    @staticmethod
    def euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate the euclidean distance between two points.

        Args:
            point1 (Tuple[int, int]): The first point
            point2 (Tuple[int, int]): The second point

        Returns:
            float: The euclidean distance between the two points
        """
        point_1 = np.array(point1)
        point_2 = np.array(point2)
        return np.linalg.norm(point_1 - point_2)

    def beam_search(
        self,
        start_point: Tuple[int, int],
        points: List[Tuple[int, int]],
        beam_width: int = 2,
    ) -> List[Tuple[int, int]]:
        """Find shortest path that visits all points using beam search.

        In this method we start with a starting point and attempt to find
        the shortest path that visits all points. We do this by looking
        at the beam_width number of paths with the smallest total distance
        and expanding them. We continue to expand paths until we have visited
        all points.

        Args:
            start_point (Tuple[int, int]): The starting point
            points (List[Tuple[int, int]]): The points to visit (not including start_point)
            beam_width (int): The beam width. Defaults to 2

        Returns:
            List[Tuple[int, int]]: The shortest path that visits all points
        """
        # Initialize the priority queue with the starting point and an empty set of visited points
        points_to_visit = set(points)
        queue: List[Tuple[float, List[Tuple[int, int]], Set[Tuple[int, int]]]] = [
            (0, [start_point], set())
        ]

        while queue:
            new_queue: List[
                Tuple[float, List[Tuple[int, int]], Set[Tuple[int, int]]]
            ] = []
            # Loop through the beam width number of paths with the smallest total distance
            for _ in range(min(beam_width, len(queue))):
                # Get the path with the smallest total distance so far
                priority, path, visited = heapq.heappop(queue)
                current = path[-1]

                # If all points have been visited, return the path
                if visited == set(points_to_visit):
                    return path

                # Sort the neighbors by distance to the current point
                neighbors = sorted(
                    points_to_visit - visited,
                    key=lambda x: self.euclidean_distance(current, x),
                )
                priorities = [priority for _ in range(len(neighbors[:beam_width]))]

                # Push the beam width nearest neighbour paths to the queue.
                for ind, neighbor in enumerate(neighbors[:beam_width]):
                    new_path = path + [neighbor]
                    new_visited = visited | {neighbor}

                    # Add the new path to the queue, using the total distance as priority
                    priorities[ind] += self.euclidean_distance(neighbor, current)
                    heapq.heappush(new_queue, (priorities[ind], new_path, new_visited))
            queue = new_queue
        return path

    def route_pins_beam_search(
        self, beam_width: int = 2
    ) -> List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """Routes pins using beam search.

        In this method, we route each net using the beam search method described in
        self.beam_search.

        Args:
            beam_width (int): The width of the beam. Defaults to 2

        Returns:
            List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]: A list of net routes,
            where each net route is a list of tuples of line endpoints.
        """
        # Get start pin position
        pin_positions = self.get_net_pin_positions()
        route_point_pairs = []
        for net in pin_positions:
            # Get start pin
            start = self.pin_outlier(pin_positions[net])
            pin_positions[net].remove(start)

            # Get shortest path and line segments
            shortest_path = self.beam_search(start, pin_positions[net], beam_width)
            line_segments = [
                (shortest_path[i], shortest_path[i + 1])
                for i in range(len(shortest_path) - 1)
            ]
            route_point_pairs.append(line_segments)

        return route_point_pairs

    def get_all_relative_coordinates_in_component(self, component: Component):
        """Gets all relative coordinates (relative to the component's top-left corner)
        in a component.

        Args:
            component (Component): component

        Returns:
            List[Tuple[int, int]]: list of relative coordinates.
        """
        coordinates = []
        for x in range(component.h):
            for y in range(component.w):
                coordinates.append((x, y))
        return coordinates

    def place_pins_on_component(self, component: Component):
        """Place pins associated with component on component randomly, update
        each pin's relative_x, relative_y.

        Args:
            component (Component): a Component object.
        """
        component_relative_coordinates = self.get_all_relative_coordinates_in_component(
            component
        )

        for pin in component.pins:
            # randomly choose a relative coordinate in the component
            relative_coordinate = random.choice(component_relative_coordinates)
            # remove relative_coordinate from component_relative_coordinates
            component_relative_coordinates.remove(relative_coordinate)
            pin.relative_x = relative_coordinate[0]
            pin.relative_y = relative_coordinate[1]

    def update_placement_mask(self, component: Component):
        """Updates the placement_mask array.

        Args:
            component (Component): component
        """
        self.placement_mask[component.comp_id] = 0.0

    def update_component_mask(self, components: List[Component]):
        """Updates the component_mask array.

        Args:
            components (List[Component]): list of components
        """
        self.component_mask = np.zeros(self.max_num_components)
        for component in components:
            self.component_mask[component.comp_id] = 1.0

    def update_pin_placement_mask(self, component: Component):
        """Updates the pin_placement_mask array.

        Args:
            component (Component): component
        """
        for pin in component.pins:
            self.pin_placement_mask[pin.pin_id] = 1.0

    def update_pin_mask(self, pins: List[Pin]):
        """Updates the pin_mask array.

        Args:
            pins (List[Pin]): list of pins
        """
        self.pin_mask = np.zeros(
            self.max_num_components * self.max_num_pins_per_component
        )
        for pin in pins:
            self.pin_mask[pin.pin_id] = 1.0

    def update_all_components_feature(self, components: List[Component]):
        """Updates the all_components_feature given a list of components, upon
        resetting the environment.

        Args:
            components (List[Component]): list of components
        """
        for component in components:
            feature = component.calculate_feature(self.area, self.max_num_pins_per_component)

            self.all_components_feature[component.comp_id, :] = feature

    def update_all_pins_feature(self, components: List[Component]):
        """Updates the all_pins_feature given a list of components, upon
        resetting the environment.

        Args:
            components (List[Component]): list of components
        """
        for component in components:
            for pin in component.pins:
                pin_num_feature, pin_cat_feature = pin.calculate_feature(
                    component.position[0], component.position[1]
                )
                self.all_pins_num_feature[pin.pin_id, :] = pin_num_feature
                self.all_pins_cat_feature[pin.pin_id, :] = pin_cat_feature

    def reset(self, verbose: bool = False, *args: Any, **kwargs: Any) -> dict:
        """Resets the environment.

        At each reset, new components are generated by sampling from the
        maximum number of components and the possible component sizes.

        Args:
            verbose (bool): whether to print the grid and action mask after the
            environment is reset

        Returns:
            observation (dict): the initial observation
        """
        # Generate instances of components
        self.grid = np.zeros((self.height, self.width))
        self.pin_grid = np.zeros((self.height, self.width, 2))
        self.net_pins = {}
        self.generate_instances()

        # Initialize masks and all_components_features
        self.all_components_feature = np.zeros(
            (self.max_num_components, 5 + self.max_num_pins_per_component), 
            dtype=np.float64
        )
        self.all_pins_num_feature = np.zeros(
            (self.max_num_components * self.max_num_pins_per_component+1, 4)
        )
        self.all_pins_cat_feature = np.zeros(
            (self.max_num_components * self.max_num_pins_per_component+1, 2)
        )
        self.all_pins_cat_feature[-1,:] = -1
        self.current_component = self.components[0]
        self.current_component_id = self.current_component.comp_id
        self.action_mask = self.compute_action_mask(self.current_component)
        

        # Update masks and features
        self.placement_mask = np.ones(self.max_num_components)
        self.placement_mask[len(self.components) :] = -1.0
        self.placement_mask[self.current_component_id] = 10.0
        self.update_component_mask(self.components)
        self.update_all_components_feature(self.components)
        self.pin_placement_mask = np.zeros(
            self.max_num_components * self.max_num_pins_per_component
        )
        self.update_pin_mask(self.pins)
        self.update_all_pins_feature(self.components)

        # Show the grid and action mask
        if verbose:
            print("Grid:")
            print(self.grid, end="\n\n")
            print("Action mask:")
            print(self.action_mask, end="\n\n")

        state = {
            "grid": self.grid.copy(),
            "pin_grid": self.pin_grid.copy(),
            "action_mask": self.action_mask.copy(),
            "all_components_feature": self.all_components_feature.copy(),
            "component_mask": self.component_mask.copy(),
            "placement_mask": self.placement_mask.copy(),
            "all_pins_num_feature": self.all_pins_num_feature.copy(),
            "all_pins_cat_feature": self.all_pins_cat_feature.copy(),
            "pin_mask": self.pin_mask.copy(),
            "pin_placement_mask": self.pin_placement_mask.copy(),
            # "current_component_id": self.current_component_id,
            # "max_num_nets": self.max_num_nets,
        }

        return state

    def step(  # noqa: max-complexity: 15
        self, action: Tuple[int, int, int], verbose: bool = False
    ) -> Tuple[dict, float, bool, dict]:
        """Steps the environment.

        Args:
            action (Tuple[int, int, int]): the action to be taken
            verbose (bool): whether to print the grid and action mask after the
            environment is reset

        Returns:
            observation (dict): the observation after the action is taken
            reward (float): the reward for taking the action
            done (bool): whether the episode is done
            info (dict): additional information
        """

        orientation, x, y = action
        self.actions.append(action)
        valid_action = self.validate_action(orientation, x, y)

        if valid_action:
            # Place the component on the grid
            # print(action)
            self.update_grid(self.current_component, orientation, x, y)
            self.current_component.place_component(orientation, x, y)

            feature = self.current_component.calculate_feature(self.area, self.max_num_pins_per_component)
            # Update the all components feature
            self.all_components_feature[self.current_component.comp_id, :] = feature
            # Update all pins feature
            self.update_all_pins_feature(self.components)
            # self.draw_pins()
            # print('valid')
            self.draw_pins()


            # Update the placement mask
            self.update_placement_mask(self.current_component)
            self.update_pin_placement_mask(self.current_component)

            # Set current component to next component
            if self.current_component.comp_id + 1 < len(self.components):
                self.current_component = self.components[
                    self.current_component.comp_id + 1
                ]
                self.current_component_id = self.current_component.comp_id
                self.placement_mask[self.current_component.comp_id] = 10.0
            else: 
                self.current_component = Component(-1, -1, -1, [])
                self.current_component_id = -1

            # Compute action mask for next component
            if self.current_component.comp_id != -1:
                self.action_mask = self.compute_action_mask(self.current_component)
            else:
                self.action_mask = np.zeros((4, self.height, self.width))

            # Calculate the reward
            done = self.compute_if_done()
            if not done:
                reward = 0
                info = {}
            else:
                reward = self.find_reward()
                info = {"wirelength": self.reward_wirelength,
                        "num_intersections": self.reward_intersection}                                

            state = {
                "grid": self.grid.copy(),
                "pin_grid": self.pin_grid.copy(),
                "action_mask": self.action_mask.copy(),
                "all_components_feature": self.all_components_feature.copy(),
                "component_mask": self.component_mask.copy(),
                "placement_mask": self.placement_mask.copy(),
                "all_pins_num_feature": self.all_pins_num_feature.copy(),
                "all_pins_cat_feature": self.all_pins_cat_feature.copy(),
                "pin_mask": self.pin_mask.copy(),
                "pin_placement_mask": self.pin_placement_mask.copy(),
                # "current_component_id": self.current_component_id,
                # "max_num_nets": self.max_num_nets,
            }

            # Print the grid and action mask if verbose
            if verbose:
                print("Grid:")
                print(self.grid, end="\n\n")
                print("Action mask:")
                print(self.action_mask, end="\n")
                print("Reward:", reward, end="\n\n")

            return state, reward, done, info

        else:
            self.draw_pins()
            # print('invalid')
            state = {
                "grid": self.grid.copy(),
                "pin_grid": self.pin_grid.copy(),
                "action_mask": self.action_mask.copy(),
                "all_components_feature": self.all_components_feature.copy(),
                "component_mask": self.component_mask.copy(),
                "placement_mask": self.placement_mask.copy(),
                "all_pins_num_feature": self.all_pins_num_feature.copy(),
                "all_pins_cat_feature": self.all_pins_cat_feature.copy(),
                "pin_mask": self.pin_mask.copy(),
                "pin_placement_mask": self.pin_placement_mask.copy(),
                # "current_component_id": self.current_component_id,
                # "max_num_nets": self.max_num_nets,
            }
            reward = self.find_reward()
            info = {"wirelength": self.reward_wirelength,
                    "num_intersections": self.reward_intersection}
            return state, reward, True, info


    def draw_pins(self):
        # curr_grid = self.grid.copy()
        # # print(self.net_pins)
        

        # for net in self.net_pins.keys():
        #     for pin in self.net_pins[net]:
        #         if int(pin.absolute_x)==-1 or int(pin.absolute_y)==-1:
        #             continue
        #         curr_grid[int(pin.absolute_x), int(pin.absolute_y)] = net + 2
        # self.pin_grid = curr_grid
        # ## get one hot encoding in numpy
        # pin_grid = self.pin_grid.copy()
        # pin_grid = pin_grid.astype(int)
        # pin_grid_one_hot = np.eye(3)[pin_grid]
        
        # pin_grid = tf.cast(curr_grid, tf.int32)
        # # print(tf.get_static_value(pin_grid))
        # # # pin_grid_one_hot = tf.one_hot(pin_grid, 3)
        # # # tf.print(pin_grid_one_hot, summarize=-1)

        # # print(pin_grid_one_hot.shape)

        # # # pin_grid_one_hot = tf.ensure_shape(pin_grid_one_hot, [pin_grid_one_hot.shape[0], pin_grid_one_hot.shape[1], pin_grid_one_hot.shape[2]])
        # # print(self.pin_grid)
        # # print(pin_grid_one_hot)
        # # print(pin_grid_one_hot[:,:,1:])
        # self.pin_grid = pin_grid_one_hot[:,:,1:]
        # print(self.pin_grid)
        pass
        # print()

    def validate_action(self, orientation: int, x: int, y: int) -> bool:
        """Validates the action.

        The action is invalid if it would cause the component to go out of
        bounds or if any of cells that the component would occupy are already
        occupied. Note that the action mask both checks if the action is out
        of bounds and if any of the cells are already occupied.

        Args:
            component (Component): the component to place on the grid
            orientation (int): the orientation of the component (0 or 1),
                where 0 means the component is placed in its original (a x b)
                orientation and 1 means the component is placed in its rotated
                (b x a) orientation
            x (int): x coordinate of the action
            y (int): y coordinate of the action

        Returns:
            bool: whether the action is valid
        """
        try:
            return self.action_mask[orientation, x, y] == 1
        except IndexError:
            return False

    def update_grid(self, component: Component, orientation: int, x: int, y: int):
        """Updates the grid with the action.

        The action is to place a component on the cell (x, y),
        where the tuple represents the top-left corner of the component.

        Args:
            component (Component): the next component to place on the grid
            orientation (int): the orientation of the component (0, 1, 2, or 3),
                            where 0 means the component is placed in its original (a x b)
                            orientation,
                            1 means the component is rotated 90 degrees clockwise
                            from its original orientation (b x a),
                            2 means the component is rotated 180 degrees clockwise
                            from its original orientation (a x b),
                            3 means the component is rotated 270 degrees clockwise
                            from its original orientation (b x a)
            x (int): x coordinate of the action
            y (int): y coordinate of the action
        """
        # Set the placement height and width based on the orientation
        if orientation in [0, 2]:
            placement_height = component.h
            placement_width = component.w
        elif orientation in [1, 3]:
            placement_height = component.w
            placement_width = component.h
        else:
            raise Exception("Invalid orientation.")
        # Place the component on the grid
        self.grid[x : x + placement_height, y : y + placement_width] = 1

    def rows_cols_to_mask(
        self, component: Component, orientation: int
    ) -> Tuple[list, list]:
        """Gets the rows and cols to mask for a given orientation and
        component.

        Args:
            component (Component): the component to place on the grid
            orientation (int): the orientation of the component (0 or 1),
                where 0 means the component is placed in its original (a x b)
                orientation and 1 means the component is placed in its rotated
                (b x a) orientation

        Returns:
            rows_to_mask (list[int]): the rows to mask
            cols_to_mask (list[int]): the columns to mask
        """
        # Set the placement height and width based on the orientation
        if orientation == 0:
            placement_height = component.h
            placement_width = component.w
        elif orientation == 1:
            placement_height = component.w
            placement_width = component.h
        else:
            raise Exception("Invalid orientation.")
        rows_to_mask = [
            i
            for i in range(self.height - 1, self.height - placement_height, -1)
            if i >= 0
        ]
        cols_to_mask = [
            i for i in range(self.width - 1, self.width - placement_width, -1) if i >= 0
        ]
        return rows_to_mask, cols_to_mask

    def compute_action_mask_orientation(
        self, component: Component, orientation: int
    ) -> np.ndarray:
        """Compute the action mask for the component orientation.

        Args:
            component (Component): the component to place on the grid
            orientation (int): the orientation of the component (0 or 1)
            where 0 means the component is placed in its original (a x b)
            orientation and 1 means the component is placed in its rotated
            (b x a) orientation

        Returns:
            action_mask_orientation (np.ndarray): the component's action mask
            for specified orientation
        """
        # Assign placement height and width based on orientation
        if orientation == 0:
            placement_height = component.h
            placement_width = component.w
        elif orientation == 1:
            placement_height = component.w
            placement_width = component.h
        else:
            raise Exception("Invalid orientation.")

        # Initialise the action mask
        action_mask_orientation = np.ones((self.height, self.width))

        # Mask rows and columns near boundary.
        rows_to_mask, cols_to_mask = self.rows_cols_to_mask(component, orientation)
        action_mask_orientation[rows_to_mask, :] = 0
        action_mask_orientation[:, cols_to_mask] = 0

        # Loop through grid checking if component could be placed at each cell
        for x in range(self.height - placement_height + 1):
            for y in range(self.width - placement_width + 1):
                # Check if the component can be placed at the cell
                if not (
                    np.all(
                        self.grid[x : x + placement_height, y : y + placement_width]
                        == 0
                    )
                ):
                    action_mask_orientation[x, y] = 0
        return action_mask_orientation

    def compute_action_mask(self, component: Component):
        """Compute the action mask for the component.

        Args:
            component (Component): the component to place on the grid

        Returns:
            action_mask (np.ndarray): the component's action mask
        """
        # Initialise the action mask
        action_mask = np.zeros((4, self.height, self.width))

        # Compute the action mask for each orientation
        action_mask[0] = self.compute_action_mask_orientation(component, 0)
        action_mask[1] = self.compute_action_mask_orientation(component, 1)
        action_mask[2] = np.copy(action_mask[0])
        action_mask[3] = np.copy(action_mask[1])
        return action_mask

    def compute_if_done(self) -> bool:
        """Computes whether the episode is terminated.

        The episode is terminated when it is not possible to placed the
        next component on the grid or all components have been placed.

        Returns:
            bool: whether the episode is terminated
        """
        # Check if there are no more components to place
        if self.current_component.comp_id != -1:
            # Check if there are no more actions to take
            return bool(np.all(self.action_mask == 0))
        return True

    def __str__(self) -> str:
        """Returns a string representation of the grid."""
        return str(self.grid)

    def render(self, *args: Any, **kwargs: Any) -> None:
        """Plots the components on a grid with the given actions."""
        app = QApplication(sys.argv)
        main_window = QMainWindow()
        main_window.setWindowTitle("Interactive Grid Plot")
        fig = Figure()
        ax = fig.add_subplot(111)

        # Load the colors for the components
        all_colors = list(matplotlib.colors.CSS4_COLORS.keys())
        # Colors for the pins
        pin_colors = list(matplotlib.colors.CSS4_COLORS.keys())
        random.shuffle(pin_colors)
        # Colors for the net
        net_colors = []

        colors = [
            c
            for c in all_colors
            if "dark" not in c.lower()
            and "light" not in c.lower()
            and "white" not in c.lower()
            and "black" not in c.lower()
        ]
        colors.reverse()  # the reverse colors look nicer in the plot

        # create the figure
        _, ax = plt.subplots()

        # plot the grid
        for i in range(self.width):
            for j in range(self.height):
                ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, zorder=0))

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect("equal")

        # plot the components
        for i, (component, action) in enumerate(zip(self.components, self.actions)):
            # get the component
            orientation = action[0]
            if orientation in [0, 2]:
                placement_height = component.h
                placement_width = component.w
            elif orientation in [1, 3]:
                placement_height = component.w
                placement_width = component.h

            x, y = placement_height, placement_width
            rgba_color = to_rgba(colors[i % len(colors)], alpha=0.85)
            # add the component
            rect = plt.Rectangle(
                (action[2], self.height - action[1] - x),
                y,
                x,
                color=rgba_color,
                lw=2,
                zorder=2,
            )
            ax.add_patch(rect)

            # add shadow effect
            shadow = plt.Rectangle(
                (action[2] + 0.1, self.height - action[1] - x - 0.1),
                y - 0.2,
                x - 0.2,
                color="black",
                alpha=0.2,
                zorder=1,
            )
            ax.add_patch(shadow)

            # add index label
            ax.text(
                action[2] + y / 2,
                self.height - action[1] - x / 2,
                f"{i}",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=12,
                color="black",
                zorder=3,
            )
            # plot the pins
            for pin in component.pins:
                pin_color = pin_colors[pin.net_id]
                net_colors.append((pin_color, pin.net_id))

                pin_marker = plt.Circle(
                    (pin.absolute_y + 0.5, self.height - pin.absolute_x - 0.5),
                    radius=0.15,
                    color=pin_color,
                    zorder=5,
                )
                # Draw the black border circle
                pin_border = plt.Circle(
                    (pin.absolute_y + 0.5, self.height - pin.absolute_x - 0.5),
                    radius=0.17,  # Slightly larger radius for the border
                    color="black",
                    zorder=4,  # Lower zorder to place it behind the colored circle
                )
                ax.add_patch(pin_marker)
                ax.add_patch(pin_border)

        # remove the axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # sort the net_colors to make them unique
        net_colors = list(set(net_colors))
        net_colors = sorted(net_colors, key=lambda x: x[1])
        net_colors = [x for (x, y) in net_colors]

        # Create a map
        cmap = matplotlib.colors.ListedColormap(net_colors)

        # Plot the colorbar
        norm = matplotlib.colors.Normalize(vmin=0, vmax=len(net_colors) - 1)
        cb = plt.colorbar(
            matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
            ax=ax,
            orientation="vertical",
        )

        # Set the colorbar ticks and labels
        cb.set_ticks(np.arange(len(net_colors)))
        cb.set_ticklabels(np.arange(len(net_colors)))

        plt.show()

        canvas = FigureCanvas(fig)
        layout = QVBoxLayout()
        layout.addWidget(canvas)

        container = QWidget()
        container.setLayout(layout)
        main_window.setCentralWidget(container)

        main_window.show()
        sys.exit(app.exec_())
