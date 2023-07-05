import random
import math
import matplotlib.pyplot as plt
import numpy as np

# from .Agent import lowest_Agent
from src.Env.utils import hierarchy_map_cont


class Continuous_Grid:
    def __init__(
        self,
        grid_settings,
        H_level: int = 2,
        goal_radius: float = 0.95,
        barrier_find_segment: int = 1001,
        random_seed: int=30,
        A_space={
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                }, 
        RS: int=2,
        l1_goal_reward=10,
        l1_subgoal_reward=4,
        action_cost=(-1) * 2,
        cont_action_radius: int=1,
        num_barrier: int = 10
    ):
        self.H_level = H_level
        
        self.l1_rows = grid_settings[0]
        self.l1_cols = grid_settings[1]
        self.l1_width = grid_settings[2]
        self.l1_height = grid_settings[3]
        self.levels = [i for i in range(0, self.H_level + 1)]
        
        self.total_width = self.l1_cols * self.l1_width
        self.total_height = self.l1_rows * self.l1_height
        
        self.A_space = A_space
        self.RS = RS
        self.l1_goal_reward = l1_goal_reward
        self.l1_subgoal_reward = l1_subgoal_reward
        self.action_cost = action_cost
        self.num_barrier = num_barrier

        self.cols = {l: int(self.l1_cols / self.RS**(l-1)) for l in self.levels}
        self.rows = {l: int(self.l1_rows / self.RS**(l-1)) for l in self.levels}
        
        self.levels = [i for i in range(0, self.H_level + 1)]
        self.cont_action_radius = cont_action_radius
        
        np.random.seed(random_seed)

        self.barrier = self.generate_barrier()
        self.start_dict, self.goal_dict = self.generate_start_goal()

        # Reward of goal and subgoal for each level
        self.set_rewards()

        self.is_terminated = False
        self.radius = goal_radius
        self.barrier_find_segment = barrier_find_segment

    def __str__(self):
        return (
            f"level: {0} \n"
            f"Grid: {self.l1_rows} rows, {self.l1_cols} columns, \n"
            f"total size: {self.total_width}x{self.total_height}, \n"
            f"start: ({self.start_dict[0][0]}, {self.start_dict[0][1]}), destination: ({self.goal_dict[0][0]}, {self.goal_dict[0][1]}), \n"
            f"is_terminated: {self.is_terminated}"
        )

    def get_possible_Action(self, s):
        level, x, y = s
        if level == 0:
            r = np.random.uniform(0, self.cont_action_radius)
            theta = np.random.uniform(0, 2 * math.pi)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            action = (level, x, y)
        else:  # discrete level
            possible_A = []
            for dx, dy in self.A_space:
                new_x = x + dx
                new_y = y + dy

                # Check if the new position is within the grid boundaries
                if 0 <= new_x < self.cols[level] and 0 <= new_y < self.rows[level]:
                    if level != 1 or not self.is_barrier(new_x, new_y):
                        # possible_A.add((level, dx, dy))
                        possible_A.append((level, dx, dy))
                        
            action = random.choice(possible_A)

        return action
            
    def random_start_goal(self):
        while True:
            start_x = np.random.uniform(0, self.total_width)
            start_y = np.random.uniform(0, self.total_height)
            goal_x = np.random.uniform(0, self.total_width)
            goal_y = np.random.uniform(0, self.total_height)
            distance = math.sqrt((start_x - goal_x) ** 2 + (start_y - goal_y) ** 2)
            if (
                distance > 3
                and not self.is_barrier(start_x, start_y)
                and not self.is_barrier(goal_x, goal_y)
            ):
                return (start_x, start_y), (goal_x, goal_y)
            
    def generate_start_goal(self):
        start, goal = self.random_start_goal()
        start_dict, goal_dict = {0: start}, {0: goal}
        for level in self.levels[1:]:
            start_dict[level] = hierarchy_map_cont(level_curr=0, level2move=level, pos=start)
            goal_dict[level] = hierarchy_map_cont(level_curr=0, level2move=level, pos=goal)

        return start_dict, goal_dict

    def generate_barrier(self):
        # Number of barrier regions (randomly chosen)
        regions = []
        for _ in range(self.num_barrier):
            # Width of the barrier region
            region_width = np.random.randint(1, 2)  # self.cols // 2)
            # Height of the barrier region
            region_height = np.random.randint(1,  2) # self.rows // 2)
            # X-coordinate of the top-left corner of the region
            region_x = np.random.randint(0, self.l1_cols - region_width)
            # Y-coordinate of the top-left corner of the region
            region_y = np.random.randint(0, self.l1_rows - region_height)
            regions.append(
                (
                    region_x * self.l1_width,
                    region_y * self.l1_height,
                    region_width * self.l1_width,
                    region_height * self.l1_height,
                )
            )
        return regions

    def is_barrier(self, x, y):
        if x < 0 or y < 0 or x >= self.total_width or y >= self.total_height:
            return True  # Outside of the grid is considered a barrier
        for region in self.barrier:
            region_x, region_y, region_width, region_height = region
            if (
                region_x <= x < region_x + region_width
                and region_y <= y < region_y + region_height
            ):
                return True  # Inside the barrier region
        return False  # Not a barrier
    
    def set_rewards(self):
        self.r_dict = {l: self.l1_goal_reward / (self.RS ** (l - 1)) for l in self.levels}
        self.sub_r_dict = {l: self.l1_subgoal_reward / (self.RS ** (l - 1)) for l in self.levels}
        self.A_cost_dict = {l: self.action_cost * (self.RS ** (l - 1)) for l in self.levels}

    def step(self, s, a):
        level, x, y = s
        level_a, dx, dy = a
        
        map_x, map_y = hierarchy_map_cont(level_curr=level,
                                     level2move=level_a,
                                     pos=(x, y))
        
        if level_a != 0:
            next_x, next_y = map_x + dx, map_y + dy
        else:  # level_a = 0
            
            next_x = np.clip(x + dx, 0, self.total_width)
            next_y = np.clip(y + dy, 0, self.total_height)

            next_x, next_y = self.find_farthest_point(
                x, y, next_x, next_y
            )

        new_s = (level_a, next_x, next_y)
        self.check_termination(new_s)

        return new_s
        
    # x1, y1: prev, x2, y2: next
    def find_farthest_point(self, prev_x, prev_y, next_x, next_y):
        farthest_x, farthest_y = prev_x, prev_y

        # Iterate along the line segment and find the farthest point
        for t in np.linspace(0, 1, self.barrier_find_segment):
            x = prev_x + (next_x - prev_x) * t
            y = prev_y + (next_y - prev_y) * t

            if not self.is_barrier(x, y):  # x, y does NOT belongs to barrier
                farthest_x, farthest_y = x, y
            else:  # x, y belongs to barrier
                return farthest_x, farthest_y

        return farthest_x, farthest_y

    def check_termination(self, s):  # for level 1
        level, x, y = s
        distance = math.sqrt(
            (x - self.goal_dict[0][0]) ** 2 + (y - self.goal_dict[0][1]) ** 2
        )

        # Check if the distance is within the specified radius (r)
        if distance <= self.radius and level == 0:
            # print('Problem Solved')
            self.is_terminated = True
            return True
            
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
        # plevel, px, py = ps
        # start_x, start_y = self.start_dict[level]
        goal_x, goal_y = self.goal_dict[level]
        
        if level > 0:
            return self.r_dict[level] if (x, y) == (goal_x, goal_y) else self.A_cost_dict[level]
        else:
            return - self.calculate_d2Goal(s)
            # return - 10 * (self.calculate_d2Goal(s)/self.calculate_d2Goal((level, start_x, start_y))) - 0 * (abs(x-px) + abs(y-py))

    def reward_subgoal(self, node):
        subgoal_r_sum = 0

        for s in node.achieved_subgoal:
            subgoal_r_sum += self.sub_r_dict[s[0]]

        return subgoal_r_sum

    # reward function
    def calculate_reward(self, node):
        subgoal_r = self.reward_subgoal(node)
        goal_r = self.reward_goal(node.s)
        
        # No cost when achieve subgoal
        if goal_r < 0 and subgoal_r != 0:
            goal_r = 0

        return subgoal_r + goal_r

    def plot_grid(self, level=0, traj= []):
        fig, ax = plt.subplots()

        # Draw the grid lines with adjusted linewidth
        for i in range(self.l1_rows + 1):
            y = i * self.l1_height
            plt.plot([0, self.total_width], [y, y], color="black", linewidth=0.5)

        for i in range(self.l1_cols + 1):
            x = i * self.l1_width
            plt.plot([x, x], [0, self.total_height], color="black", linewidth=0.5)

        # Plot the start and destination points with larger size
        plt.scatter(
            self.start_dict[level][0], self.start_dict[level][1], color="green", marker="o", s=25, label="Start"
        )
        plt.scatter(
            self.goal_dict[level][0], self.goal_dict[level][1], color="red", marker="o", s=25, label="Goal"
        )

        # Plot barrier regions with the same color
        if level <= 1:
            for region in self.barrier:
                region_x, region_y, region_width, region_height = region
                rect = plt.Rectangle(
                    (region_x, region_y),
                    region_width,
                    region_height,
                    color="red",
                    alpha=0.3,
                )
                ax.add_patch(rect)

        # Plot the agent's path with a different color and marker
        if not traj:
            pass
        else:
            l, traj_x, traj_y = zip(*traj)
            plt.scatter(traj_x[1:], traj_y[1:], color="skyblue", marker="o", s=25, zorder=2)
            plt.plot(
                traj_x, traj_y, color="skyblue", linewidth=2, label="Agent's path", zorder=1
            )

            # Plot arrows for the agent's direction with larger size
            for i in range(1, len(traj_x)):
                dx = traj_x[i] - traj_x[i - 1]
                dy = traj_y[i] - traj_y[i - 1]
                plt.arrow(
                    traj_x[i - 1],
                    traj_y[i - 1],
                    dx,
                    dy,
                    color="blue",
                    width=0.05,
                    head_width=0.5,
                    length_includes_head=True,
                )

        plt.gca().set_aspect("equal", adjustable="box")

        # Set the tick marks to align with the grid
        ax.set_xticks(np.arange(0, self.total_width + self.l1_width, self.l1_width))
        ax.set_yticks(
            np.arange(0, self.total_height + self.l1_height, self.l1_height)
        )

        # Move the legend outside the figure
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title(f"Grid (Level {level})")

        # Show the plot
        plt.show()