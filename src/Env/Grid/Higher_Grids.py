import math
import numpy as np
import matplotlib.pyplot as plt

from ..utils import hierarchy_map
from .All_level_Grid_w_agent import check_both_power_of_RS


class HighLevelGrids:
    def __init__(
        self,
        grid_settings,
        H_level=None,
        A_space={
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        },  # {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)} for root node? root?
        RS=2,
        l1_goal_reward=20,
        l1_subgoal_reward=4,
        action_cost=(-1) * 4,
        random_seed=26,
        num_barrier=10,
    ):
        self.l1_rows = grid_settings[0]
        self.l1_cols = grid_settings[1]
        self.l1_width = grid_settings[2]
        self.l1_height = grid_settings[3]

        self.total_width = self.l1_cols * self.l1_width
        self.total_height = self.l1_rows * self.l1_height

        self.A_space = A_space
        self.RS = RS
        self.l1_goal_reward = l1_goal_reward
        self.l1_subgoal_reward = l1_subgoal_reward
        self.action_cost = action_cost

        np.random.seed(random_seed)  # Fix seed

        # Set highest level
        if H_level is None:
            self.H_level = check_both_power_of_RS(self.l1_rows, self.l1_cols, RS=RS)
        else:
            self.H_level = H_level
        
        self.levels = [i for i in range(1, H_level + 1)]

        # set level 1 barrier
        self.num_barrier = num_barrier  # random.randint(3, 6)
        self.generate_barrier()

        # Set start, goal point
        self.start_dict, self.goal_dict = self.generate_start_goal()

        # reward of goal and subgoal for each level
        self.set_rewards()  # key: level, value: reward
        
        # num of cols and rows for each level
        self.cols = {l: int(self.l1_cols / self.RS**(l-1)) for l in self.levels}
        self.rows = {l: int(self.l1_rows / self.RS**(l-1)) for l in self.levels}
        
        self.set_possible_Action_dict()

    def random_start_goal(self):
        while True:
            start_x, goal_x = np.random.randint(0, self.l1_cols, 2)
            start_y, goal_y = np.random.randint(0, self.l1_rows, 2)
            start = (start_x, start_y)
            goal = (goal_x, goal_y)
            distance = abs(start_x - goal_x) + abs(start_y - goal_y)
            
            # At least distance >= 2, start and goal does not belong barrier
            if distance > 2 and start not in self.barrier and goal not in self.barrier:  # distance > 1
                return start, goal

    def generate_start_goal(self):
        start, goal = self.random_start_goal()
        start_dict, goal_dict = {1: start}, {1: goal}

        for level in self.levels[1:]:
            start_dict[level] = hierarchy_map(level_curr=1, level2move=level, pos=start)
            goal_dict[level] = hierarchy_map(level_curr=1, level2move=level, pos=goal)

        return start_dict, goal_dict

    def set_rewards(self):
        self.r_dict = {l: self.l1_goal_reward / (self.RS ** (l - 1)) for l in self.levels}
        self.sub_r_dict = {l: self.l1_subgoal_reward / (self.RS ** (l - 1)) for l in self.levels}
        self.A_cost_dict = {l: self.action_cost * (self.RS ** (l - 1)) for l in self.levels}

    def set_possible_Action_dict(self):
        self.possible_Action_dict = dict()
        
        for l in self.levels:
            for i in range(self.cols[l]):
                for j in range(self.rows[l]):
                    s = (l, i, j)
                    if l == 1 and self.is_barrier(i, j):
                        continue
                    self.possible_Action_dict[s] = self.get_possible_Action(s)

    def get_possible_Action(self, s):
        level, x, y = s
        possible_A = set()
        
        # Check the neighboring cells in all directions
        for dx, dy in self.A_space:
            new_x = x + dx
            new_y = y + dy

            # Check if the new position is within the grid boundaries
            if 0 <= new_x < self.cols[level] and 0 <= new_y < self.rows[level]:
                if level != 1 or not self.is_barrier(new_x, new_y):
                    possible_A.add((level, dx, dy))

        return possible_A

    # transition function T(s, a) -> s
    def step(self, s, a):  # s: (level, x, y), a: (level, dx, dy)
        level, x, y = s
        level_a, dx, dy = a

        if (dx, dy) not in self.A_space:
            raise Exception("Invalid action")

        next_x, next_y = x + dx, y + dy

        # Clip the next position to ensure it stays within the grid boundaries
        next_x = np.clip(next_x, 0, self.cols[level] - 1)
        next_y = np.clip(next_y, 0, self.rows[level] - 1)

        # barrier is only detected at level 1
        if level == 1 and self.is_barrier(next_x, next_y):
            return (level, x, y)

        return (level, next_x, next_y)

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
        return self.r_dict[level] if (x, y) == (goal_x, goal_y) else self.A_cost_dict[level]

    def reward_subgoal(self, node, subgoal_set):
        level, x, y = node.s
        subgoal_r_sum = 0

        for level_subgoal, subgoal_x, subgoal_y in subgoal_set:
            if level_subgoal > level:
                map_x, map_y = hierarchy_map(
                    level_curr=level, level2move=level_subgoal, pos=(x, y)
                )
                if (subgoal_x, subgoal_y) == (map_x, map_y):
                    subgoal_r_sum += self.sub_r_dict[level_subgoal]

        return subgoal_r_sum

    # reward function
    def calculate_reward(self, node, subgoal_set):
        subgoal_r = self.reward_subgoal(node, subgoal_set)
        goal_r = self.reward_goal(node.s)
        
        # No cost when achieve subgoal
        if goal_r < 0 and subgoal_r != 0:
            goal_r = 0

        return subgoal_r + goal_r

    def generate_barrier(self):
        self.barrier = set()
        for _ in range(self.num_barrier):
            # Width, height of the barrier 
            barrier_width = np.random.randint(1, 2)  # self.cols // 2)
            barrier_height = np.random.randint(1, 2)  # self.rows // 2)
            
            # X, Y-coordinate of the bottom-left barrier
            barrier_x = np.random.randint(0, self.l1_cols - barrier_width)
            barrier_y = np.random.randint(0, self.l1_rows - barrier_height)
            for i in range(barrier_width):
                for j in range(barrier_height):
                    self.barrier.add((barrier_x + i, barrier_y + j))

    def is_barrier(self, x, y):  # barrier for grid world
        if x < 0 or y < 0 or x >= self.l1_cols or y >= self.l1_rows:
            return True  # Outside of the grid is considered a barrier
        for barrier in self.barrier:
            if (x, y) == barrier:
                return True  # Inside the barrier region
        return False  # Not a barrier

    def plot_grid(self, level):
        fig, ax = plt.subplots(figsize=(3, 3))

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
