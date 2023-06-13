import math
import numpy as np

from ..utils import hierarchy_map
from .All_level_Grid_w_agent import check_both_power_of_RS
from .l0_Grid_w_agent import lowest_Grid_w_agent


class high_level_Grids:
    def __init__(
        self,
        l1_rows: int,
        l1_cols: int,
        l1_cell_width,  # not have to be int, but recommended
        l1_cell_height,  # not have to be int, but recommended
        destination_radius: float = 1,
        barrier_find_segment: int = 101,
        highest_level=None,
        A_space={(1, 0), (-1, 0), (0, 1), (0, -1)},  # {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)} for root
        RS=2,
        l1_goal_reward=20,
        l1_subgoal_reward=4,
        action_cost=-1,
    ):
        if highest_level is None:
            self.highest_level = check_both_power_of_RS(l1_rows, l1_cols, RS=2)
        else:
            self.highest_level = highest_level

        self.l1_cell_width = l1_cell_width
        self.l1_cell_height = l1_cell_height
        self.A_space = A_space
        self.RS = RS
        self.l1_goal_reward = l1_goal_reward
        self.l1_subgoal_reward = l1_subgoal_reward
        self.action_cost = action_cost

        self.l0_grid = lowest_Grid_w_agent(
            l1_rows,
            l1_cols,
            l1_cell_width,
            l1_cell_height,
            level=0,
            destination_radius=destination_radius,
            barrier_find_segment=barrier_find_segment,
        )

        self.start_dict, self.dest_dict = self.set_start_dest_coordinates()
        self.goal_dict, self.subgoal_dict = self.set_rewards()
        self.cols = {l+1: l1_cols / self.RS**l for l in range(self.highest_level)}
        self.rows = {l+1: l1_rows / self.RS**l for l in range(self.highest_level)}


    def set_start_dest_coordinates(self):
        start_dict = {}
        dest_dict = {}
        for level in range(self.highest_level):
            l = level + 1
            start_x, start_y = hierarchy_map(
                level_current=0,
                level_to_move=l,
                x=self.l0_grid.start_x,
                y=self.l0_grid.start_y,
                cell_width=self.l1_cell_width,
                cell_height=self.l1_cell_height,
            )
            dest_x, dest_y = hierarchy_map(
                level_current=0,
                level_to_move=l,
                x=self.l0_grid.dest_x,
                y=self.l0_grid.dest_y,
                cell_width=self.l1_cell_width,
                cell_height=self.l1_cell_height,
            )

            start_dict[l] = (start_x, start_y)
            dest_dict[l] = (dest_x, dest_y)

        return start_dict, dest_dict

    def set_rewards(self):
        goal_dict = {}
        subgoal_dict = {}
        for level in range(self.highest_level):
            l = level + 1
            goal_dict[l] = self.l1_goal_reward / (self.RS ** level)
            subgoal_dict[l] = self.l1_subgoal_reward / (self.RS ** level)

        return goal_dict, subgoal_dict
    
    def get_possible_Action(self, state):
        level, x, y = state
        possible_A = []
        
        # Check the neighboring cells in all directions
        directions = tuple(self.A_space)
        
        for dx, dy in directions:
            new_x = x + dx
            new_y = y + dy
            
            # Check if the new position is within the grid boundaries
            if 0 <= new_x < self.cols[level] and 0 <= new_y < self.rows[level]:
                possible_A.append((dx, dy))
        
        return possible_A

    def step(self, state, action, subgoal_set):
        level, x, y = state
        if action not in self.A_space:
            raise Exception("Wrong action")

        next_x = x + action[0]
        next_y = y + action[1]

        # Clip the next position to ensure it stays within the grid boundaries
        next_x = np.clip(next_x, 0, self.cols[level] - 1)
        next_y = np.clip(next_y, 0, self.rows[level] - 1)

        r = self.calculate_reward(level, next_x, next_y, subgoal_set)
        done = self.check_termination_pos_at_Level(level, next_x, next_y)
        total_done = level == 1  # lowest at discrete action space's level

        return (level, next_x, next_y), r, done, total_done, self.calculate_d2Goal(level, next_x, next_y)

    def calculate_d2Goal(self, state):
        level, x, y = state
        dest_x, dest_y = self.dest_dict[level]
        return abs(x - dest_x) + abs(y - dest_y)  # Manhattan distance

    def check_termination_pos_at_Level(self, state):
        level, x, y = state
        dest_x, dest_y = self.dest_dict[level]
        return (x, y) == (dest_x, dest_y)

    def check_Root_pos_at_Level(self, state):
        level, x, y = state
        start_x, start_y = self.start_dict[level]
        return (x, y) == (start_x, start_y)

    def reward_goal(self, state):
        level, x, y = state
        dest_x, dest_y = self.dest_dict[level]
        return self.l1_goal_reward if (x, y) == (dest_x, dest_y) else self.action_cost

    def reward_subgoal(self, state, subgoal_set):
        level, x, y = state
        subgoal_r_sum = 0
        for level_subgoal, subgoal_x, subgoal_y in subgoal_set:
            map_x, map_y = hierarchy_map(
                level_current=level,
                level_to_move=level_subgoal,
                x=x,
                y=y,
                cell_height=self.l1_cell_height,
                cell_width=self.l1_cell_width,
            )

            if (subgoal_x, subgoal_y) == (map_x, map_y):
                subgoal_r_sum += self.subgoal_dict[level_subgoal]

        return subgoal_r_sum

    def calculate_reward(self, state, subgoal_set):
        subgoal_r = self.reward_subgoal(state, subgoal_set)
        goal_r = self.reward_goal(state)
        return subgoal_r + goal_r
