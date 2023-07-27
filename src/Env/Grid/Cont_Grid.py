import random
import math
import matplotlib.pyplot as plt
import numpy as np
import graphviz

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
        num_barrier: int = 10,
        goal_n_start_distance: int = 3,
    ):
        self.H_level = H_level
        
        self.l1_rows = grid_settings[0]
        self.l1_cols = grid_settings[1]
        self.l1_width = grid_settings[2]
        self.l1_height = grid_settings[3]
        self.levels = [i for i in range(0, self.H_level + 1)]
        self.goal_n_start_distance = goal_n_start_distance
        
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
            # Sample Action
            r = np.random.uniform(0, self.cont_action_radius)
            theta = np.random.uniform(0, 2 * math.pi)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            action = (level, x, y)
        else:  
            # In Discrete level
            possible_A = []
            for dx, dy in self.A_space:
                new_x = x + dx
                new_y = y + dy

                # Check if the new position is within the grid boundaries
                if 0 <= new_x < self.cols[level] and 0 <= new_y < self.rows[level]:
                    # if level != 1 or not self.is_barrier(new_x, new_y):
                        # possible_A.add((level, dx, dy))
                    possible_A.append((level, dx, dy))
                        
            action = random.choice(possible_A)
        return action
    
    def get_possible_Action_for_expand(self, s):
        level, x, y = s
        if level == 0:
            assert(False)
        else:  
            # In Discrete level
            possible_A = []
            for dx, dy in self.A_space:
                new_x = x + dx
                new_y = y + dy

                # Check if the new position is within the grid boundaries
                if 0 <= new_x < self.cols[level] and 0 <= new_y < self.rows[level]:
                    # if level != 1 or not self.is_barrier(new_x, new_y):
                        # possible_A.add((level, dx, dy))
                    possible_A.append((level, dx, dy))           
        return possible_A
            
    def random_start_goal(self):
        while True:
            start_x = np.random.uniform(0, self.total_width)
            start_y = np.random.uniform(0, self.total_height)
            goal_x = np.random.uniform(0, self.total_width)
            goal_y = np.random.uniform(0, self.total_height)
            distance = math.sqrt((start_x - goal_x) ** 2 + (start_y - goal_y) ** 2)
            if (
                distance > self.goal_n_start_distance
                # distance > 3
                and not self.is_barrier(start_x, start_y)
                and not self.is_barrier(goal_x, goal_y)
            ):
                # print(distance)
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
                region_x < x < region_x + region_width 
                and region_y < y < region_y + region_height
            ): # Boundaries are not barrier
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
            
            # next_x = np.clip(x + dx, 0, self.total_width)
            # next_y = np.clip(y + dy, 0, self.total_height)

            next_x, next_y = self.find_farthest_point(
                x, y, dx, dy
            )

        new_s = (level_a, next_x, next_y)
        self.check_termination(new_s)

        return new_s
        
    # x1, y1: prev, x2, y2: next
    def find_farthest_point(self, prev_x, prev_y, dx, dy):
        small_const= 0.000001
        farthest_x, farthest_y = prev_x, prev_y

        theta = math.atan2(dy, dx)
        r = math.sqrt(dx**2 + dy**2)

        line = ((prev_x, prev_y), (prev_x + dx, prev_y + dy))
        colliding_point = []
        # Check if any barrier point is in the dx/dy box
        for region in self.barrier:
            region_x, region_y, region_width, region_height = region
            l_l = ((region_x, region_y), (region_x, region_y + region_height))
            r_l = ((region_x + region_width, region_y), (region_x + region_width, region_y + region_height))
            d_l = ((region_x, region_y), (region_x + region_width, region_y))
            u_l = ((region_x, region_y + region_height), (region_x + region_width, region_y + region_height))
            
            intersect_l, intersect_l_point = self.line_intersection(line, l_l)
            intersect_r, intersect_r_point = self.line_intersection(line, r_l)
            intersect_d, intersect_d_point = self.line_intersection(line, d_l)
            intersect_u, intersect_u_point = self.line_intersection(line, u_l)

            if intersect_l:
                (px, py) = intersect_l_point
                colliding_point.append((px - small_const, py))
            if intersect_r:
                (px, py) = intersect_r_point
                colliding_point.append((px + small_const, py))
            if intersect_u:
                (px, py) = intersect_u_point
                colliding_point.append((px, py + small_const))
            if intersect_d:
                (px, py) = intersect_d_point
                colliding_point.append((px, py - small_const))

        # Consider the boundary of grid
        left_wall = ((0, 0), (0, self.total_height))
        right_wall = ((self.total_width, 0), (self.total_width, self.total_height))
        down_wall = ((0, 0), (self.total_width, 0))
        up_wall = ((0, self.total_height), (self.total_width, self.total_height))
        
        intersect_lw, intersect_lw_point = self.line_intersection(line, left_wall)
        intersect_rw, intersect_rw_point = self.line_intersection(line, right_wall)
        intersect_uw, intersect_uw_point = self.line_intersection(line, up_wall)
        intersect_dw, intersect_dw_point = self.line_intersection(line, down_wall)

        if intersect_lw:
            (px, py) = intersect_lw_point
            colliding_point.append((px + small_const, py))
        elif intersect_rw:
            (px, py) = intersect_rw_point
            colliding_point.append((px - small_const, py))
        elif intersect_uw:
            (px, py) = intersect_uw_point
            colliding_point.append((px, py - small_const))
        elif intersect_dw:
            (px, py) = intersect_dw_point
            colliding_point.append((px, py + small_const))
        
        if len(colliding_point) == 0:
            farthest_x += dx
            farthest_y += dy
        else:
            farthest_x, farthest_y = self.closest_point((prev_x, prev_y), colliding_point)

        return farthest_x, farthest_y

    def line_intersection(self, l1 ,l2):
        p1, p2 = l1
        p3, p4 = l2
        def on_segment(p, q, r):
            if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
                return True
            return False

        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # Collinear
            elif val > 0:
                return 1  # Clockwise orientation
            else:
                return 2  # Counterclockwise orientation

        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)

        # General case for intersection
        if o1 != o2 and o3 != o4:
            intersecting_x = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[0] - p4[0]) - (p1[0] - p2[0]) * (p3[0] * p4[1] - p3[1] * p4[0])) / ((p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]))
            intersecting_y = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] * p4[1] - p3[1] * p4[0])) / ((p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]))
            intersecting_point = (intersecting_x, intersecting_y)
            return True, intersecting_point

        # Special cases for collinear points
        if o1 == 0 and on_segment(p1, p3, p2):
            return True, p3
        if o2 == 0 and on_segment(p1, p4, p2):
            return True, p4
        if o3 == 0 and on_segment(p3, p1, p4):
            return True, p1
        if o4 == 0 and on_segment(p3, p2, p4):
            return True, p2

        return False, None


    def closest_point(self, p0, list_p):
        closest_distance = math.inf
        closest_point = None

        for point in list_p:
            distance = math.sqrt((point[0] - p0[0])**2 + (point[1] - p0[1])**2)
            if distance < closest_distance:
                closest_distance = distance
                closest_point = point

        return closest_point

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
    
    def reward_goal(self, s, ps, weight=1):
        level, x, y = s
        plevel, px, py = ps
        return - self.calculate_d2Goal(s) - math.sqrt(abs(x-px)**2 + abs(y-py)**2) * weight
            # Save Good
            # return - self.calculate_d2Goal(s) - (math.sqrt(abs(x-px)**2 + abs(y-py)**2)/10)

    def reward_subgoal(self, node):
        subgoal_r_sum = 0

        for s in node.achieved_subgoal:
            subgoal_r_sum += self.sub_r_dict[s[0]]

        return subgoal_r_sum

    # reward function
    def calculate_reward(self, node, weight=1):
        subgoal_r = self.reward_subgoal(node)
        goal_r = self.reward_goal(node.s, node.parent.s, weight)
        
        # No cost when achieve subgoal
        if goal_r < 0 and subgoal_r != 0:
            goal_r = 0

        return subgoal_r + goal_r

    def plot_grid(self, level=0, traj=[], save_path=None):
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

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            # Show the plot
            # pass
            plt.show()

    def draw_graph(self, node, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(node.s[1], node.s[2], 'c.')
        for child in node.children.values():
            if child.s[0] != 0:
                continue
            ax.plot([node.s[1], child.s[1]], [node.s[2], child.s[2]], 'c-')
            self.draw_graph(child, ax)

    def plot_grid_tree(self, root, level=0, save_path=None):
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
        if root.s[0] != 0:
            return
        else:
            self.draw_graph(root, ax)

        plt.gca().set_aspect("equal", adjustable="box")

        # Set the tick marks to align with the grid
        ax.set_xticks(np.arange(0, self.total_width + self.l1_width, self.l1_width))
        ax.set_yticks(
            np.arange(0, self.total_height + self.l1_height, self.l1_height)
        )

        # Move the legend outside the figure
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title(f"Grid (Level {level})")

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            # Show the plot
            # pass
            plt.show()

    def visualize_tree(self, root, depth=0, save_path=None):
        dot = graphviz.Digraph()
        dot.node(str(root.s)+"@"+str(depth)+"root")

        def add_nodes_edges(node, depth):
            for child in node.children.values():
                dot.node(str(child.s)+"@"+str(depth+1)+"parent:"+str(child.parent))
                if child.parent.isRoot:
                    dot.edge(str(root.s)+"@"+str(depth)+"root", str(child.s)+"@"+str(depth)+"parent:"+str(child.parent))
                else:
                    dot.edge(str(node.s)+"@"+str(depth)+"parent:"+str(node.parent), str(child.s)+"@"+str(depth)+"parent:"+str(child.parent))
                add_nodes_edges(child, depth+1)

        add_nodes_edges(root, 0)
        dot.render(view=True, format='png')