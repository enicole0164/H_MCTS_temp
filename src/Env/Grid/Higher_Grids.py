import math
import numpy as np
import matplotlib.pyplot as plt

from ..utils import hierarchy_map
from .All_level_Grid_w_agent import check_both_power_of_RS


class HighLevelGrids:
    def __init__(
        self,
        grid_settings,
        highest_level=None,
        A_space={
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        },  # {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)} for root
        RS=2,
        l1_goal_reward=20,
        l1_subgoal_reward=4,
        action_cost=-1,
        random_seed=26,
        l1_barrier=True,
        num_barrier=10
    ):
        self.l1_rows = grid_settings[0]
        self.l1_cols = grid_settings[1]
        self.l1_width = grid_settings[2]
        self.l1_height = grid_settings[3]

        if highest_level is None:
            self.highest_level = check_both_power_of_RS(
                self.l1_rows, self.l1_cols, RS=2
            )
        else:
            self.highest_level = highest_level

        self.total_width = self.l1_cols * self.l1_width
        self.total_height = self.l1_rows * self.l1_height

        self.A_space = A_space
        self.RS = RS
        self.l1_goal_reward = l1_goal_reward
        self.l1_subgoal_reward = l1_subgoal_reward
        self.action_cost = action_cost

        np.random.seed(random_seed)

        if l1_barrier:
            self.num_barrier = num_barrier  # random.randint(3, 6)  # 15
            self.generate_barrier()
        else:
            self.num_barrier = 0
            self.barrier = []

        self.start_dict, self.goal_dict = self.generate_start_goal()

        # reward of goal and subgoal for each level; key: level, value: reward
        self.r_dict, self.sub_r_dict = self.set_rewards()
        self.cols = {
            l + 1: int(self.l1_cols / self.RS**l) for l in range(self.highest_level)
        }
        self.rows = {
            l + 1: int(self.l1_rows / self.RS**l) for l in range(self.highest_level)
        }

    def random_start_goal(self):
        start_x = np.random.randint(0, self.l1_cols)
        start_y = np.random.randint(0, self.l1_rows)
        goal_x = np.random.randint(0, self.l1_cols)
        goal_y = np.random.randint(0, self.l1_rows)

        while (start_x, start_y) == (goal_x, goal_y):
            start_x = np.random.randint(0, self.l1_cols)
            start_y = np.random.randint(0, self.l1_rows)
            goal_x = np.random.randint(0, self.l1_cols)
            goal_y = np.random.randint(0, self.l1_rows)
        
        return (start_x, start_y), (goal_x, goal_y)

    def generate_start_goal(self):
        positions = {"start": None, "goal": None}
        for pos in positions.keys():
            positions[pos], _ = self.random_start_goal()
            while positions[pos] in self.barrier:
                positions[pos], _ = self.random_start_goal()

        start_dict = {1: positions["start"]}
        goal_dict = {1: positions["goal"]}
        for level in range(2, self.highest_level + 1):
            for pos in positions.keys():
                positions[pos] = hierarchy_map(
                    level_current=1,
                    level_to_move=level,
                    x=positions[pos][0],
                    y=positions[pos][1],
                )
                if pos == "start":
                    start_dict[level] = positions[pos]
                else:
                    goal_dict[level] = positions[pos]

        return start_dict, goal_dict

    def set_rewards(self):
        r_dict = {}
        sub_r_dict = {}
        for l in range(1, self.highest_level + 1):
            r_dict[l] = self.l1_goal_reward / (self.RS ** (l - 1))
            sub_r_dict[l] = self.l1_subgoal_reward / (self.RS ** (l - 1))

        return r_dict, sub_r_dict

    def get_possible_Action(self, s):
        level, x, y = s
        possible_A = []

        # Check the neighboring cells in all directions
        directions = tuple(self.A_space)

        for dx, dy in directions:
            new_x = x + dx
            new_y = y + dy
            # Check if the new position is within the grid boundaries
            if 0 <= new_x < self.cols[level] and 0 <= new_y < self.rows[level]:
                if level == 1:
                    if not self.is_barrier(new_x, new_y):
                        possible_A.append((level, dx, dy))
                else:
                    possible_A.append((level, dx, dy))
                # possible_A.append((level, dx, dy))

        return possible_A  # list e.g. [(0, 1), (0, - 1), (1, 0), (- 1, 0)]

    # transition function T(s, a) -> s
    def step(self, s, a):  # s: (level, x, y), a: (level, dx, dy)
        level, x, y = s
        if (a[1], a[2]) not in self.A_space:
            raise Exception("Wrong action")

        next_x = x + a[1]
        next_y = y + a[2]

        # Clip the next position to ensure it stays within the grid boundaries
        next_x = np.clip(next_x, 0, self.cols[level] - 1)
        next_y = np.clip(next_y, 0, self.rows[level] - 1)

        if level == 1 and self.is_barrier(next_x, next_y):
            next_x, next_y = x, y

        return (level, next_x, next_y)  # , r

    def calculate_d2Goal(self, s):
        level, x, y = s
        goal_x, goal_y = self.goal_dict[level]
        return abs(x - goal_x) + abs(y - goal_y)  # Manhattan distance

    def check_goal_pos(self, s):
        level, x, y = s
        goal_x, goal_y = self.goal_dict[level]
        return (x, y) == (goal_x, goal_y)

    # check the state belongs to Root or not.
    def check_R_pos(self, s):
        level, x, y = s
        start_x, start_y = self.start_dict[level]
        return (x, y) == (start_x, start_y)

    def reward_goal(self, s):
        level, x, y = s
        goal_x, goal_y = self.goal_dict[level]
        return self.r_dict[level] if (x, y) == (goal_x, goal_y) else self.action_cost

    def reward_subgoal(self, node, subgoal_set):
        level, x, y = node.s
        subgoal_r_sum = 0
        for level_subgoal, subgoal_x, subgoal_y in subgoal_set:
            map_x, map_y = hierarchy_map(
                level_current=level,
                level_to_move=level_subgoal,
                x=x,
                y=y,
            )

            if (subgoal_x, subgoal_y) == (map_x, map_y) and level_subgoal > level:
                # node.subgoal_set.remove((level_subgoal, map_x, map_y))
                subgoal_r_sum += self.sub_r_dict[level_subgoal]

        return subgoal_r_sum

    # reward function
    def calculate_reward(self, node, subgoal_set):
        subgoal_r = self.reward_subgoal(node, subgoal_set)
        goal_r = self.reward_goal(node.s)
        # print(node, subgoal_set, subgoal_r, goal_r)
        return subgoal_r + goal_r

    def generate_barrier(self):
        regions = set()
        for _ in range(self.num_barrier):
            # Width of the barrier region
            region_width = np.random.randint(1, 2)  # self.cols // 2)
            # Height of the barrier region
            region_height = np.random.randint(1, 2)  # self.rows // 2)
            # X-coordinate of the top-left corner of the region
            region_x = np.random.randint(0, self.l1_cols - region_width)
            # Y-coordinate of the top-left corner of the region
            region_y = np.random.randint(0, self.l1_rows - region_height)
            for i in range(region_width):
                for j in range(region_height):
                    regions.add((region_x + i, region_y + j))

        self.barrier = regions

    def is_barrier(self, x, y):  # barrier for grid world
        if x < 0 or y < 0 or x >= self.l1_cols or y >= self.l1_rows:
            return True  # Outside of the grid is considered a barrier
        for region in self.barrier:
            region_x, region_y = region
            if (x, y) == (region_x, region_y):
                return True  # Inside the barrier region
        return False  # Not a barrier

    def plot_grid(self, level):
        fig, ax = plt.subplots()

        # Determine the grid size based on the specified level
        rows = self.rows[level]
        cols = self.cols[level]
        cell_width = self.total_width / cols
        cell_height = self.total_height / rows

        # Draw the grid lines with adjusted linewidth and grid size
        for i in range(rows + 1):
            y = i * cell_height
            plt.plot([0, self.total_width], [y, y], color="black", linewidth=0.5)

        for i in range(cols + 1):
            x = i * cell_width
            plt.plot([x, x], [0, self.total_height], color="black", linewidth=0.5)

        # Plot the start and goal points at the middle of the grid cell
        start_x, start_y = self.start_dict[level]
        goal_x, goal_y = self.goal_dict[level]
        start_x_middle = (start_x + 0.5) * cell_width
        start_y_middle = (start_y + 0.5) * cell_height
        goal_x_middle = (goal_x + 0.5) * cell_width
        goal_y_middle = (goal_y + 0.5) * cell_height
        plt.scatter(
            start_x_middle,
            start_y_middle,
            color="green",
            marker="o",
            s=25,
            label="Start",
        )
        plt.scatter(
            goal_x_middle, goal_y_middle, color="red", marker="o", s=25, label="goal"
        )

        if level == 1:
            for region in self.barrier:
                region_x, region_y = region  # barrier defined by coordinates
                rect = plt.Rectangle(
                    (region_x * cell_width, region_y * cell_height),
                    cell_width,
                    cell_height,
                    color="red",
                    alpha=0.3,
                )
                ax.add_patch(rect)

        plt.gca().set_aspect("equal", adjustable="box")

        # Set the tick marks to align with the grid
        ax.set_xticks(np.arange(0, self.total_width + cell_width, cell_width))
        ax.set_yticks(np.arange(0, self.total_height + cell_height, cell_height))

        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title(f"Grid (Level {level})")

        plt.show()
