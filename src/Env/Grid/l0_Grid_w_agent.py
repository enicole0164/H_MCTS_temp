import random
import math
import matplotlib.pyplot as plt
import numpy as np

from .Agent import lowest_Agent


class lowest_Grid_w_agent:
    def __init__(
        self,
        rows: int,
        cols: int,
        cell_width,  # not have to be int, but recommended
        cell_height,  # not have to be int, but recommended
        level: int = 0,
        destination_radius: float = 2,
        barrier_find_segment: int = 101,
    ):
        self.level = level
        self.rows = rows
        self.cols = cols
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.total_width = cols * cell_width
        self.total_height = rows * cell_height

        self.barrier_regions = self.generate_barrier_regions()
        (
            self.start_x,
            self.start_y,
            self.dest_x,
            self.dest_y,
        ) = self.generate_start_dest()

        self.agent = lowest_Agent(self.start_x, self.start_y, self.level)
        self.is_terminated = False
        self.radius = destination_radius
        self.barrier_find_segment = barrier_find_segment

    def __str__(self):
        return (
            f"level: {self.level} \n"
            f"Grid: {self.rows} rows, {self.cols} columns, \n"
            f"total size: {self.total_width}x{self.total_height}, \n"
            f"start: ({self.start_x}, {self.start_y}), destination: ({self.dest_x}, {self.dest_y}), \n"
            f"is_terminated: {self.is_terminated}"
        )

    def generate_start_dest(self):
        while True:
            start_x = np.random.uniform(0, self.total_width)
            start_y = np.random.uniform(0, self.total_height)
            dest_x = np.random.uniform(0, self.total_width)
            dest_y = np.random.uniform(0, self.total_height)
            distance = math.sqrt((start_x - dest_x) ** 2 + (start_y - dest_y) ** 2)
            if (
                distance > 1
                and not self.is_barrier(start_x, start_y)
                and not self.is_barrier(dest_x, dest_y)
            ):
                return start_x, start_y, dest_x, dest_y

    def generate_barrier_regions(self):
        # Number of barrier regions (randomly chosen)
        num_regions = 4  # random.randint(3, 6)
        regions = []
        for _ in range(num_regions):
            # Width of the barrier region
            region_width = random.randint(1, self.cols // 2)
            # Height of the barrier region
            region_height = random.randint(1, self.rows // 2)
            # X-coordinate of the top-left corner of the region
            region_x = random.randint(0, self.cols - region_width)
            # Y-coordinate of the top-left corner of the region
            region_y = random.randint(0, self.rows - region_height)
            regions.append(
                (
                    region_x * self.cell_width,
                    region_y * self.cell_height,
                    region_width * self.cell_width,
                    region_height * self.cell_height,
                )
            )
        return regions

    def is_barrier(self, x, y):
        if x < 0 or y < 0 or x >= self.total_width or y >= self.total_height:
            return True  # Outside of the grid is considered a barrier
        for region in self.barrier_regions:
            region_x, region_y, region_width, region_height = region
            if (
                region_x <= x < region_x + region_width
                and region_y <= y < region_y + region_height
            ):
                return True  # Inside the barrier region
        return False  # Not a barrier

    def move_agent(self, dx, dy):
        next_x = np.clip(self.agent.x + dx, 0, self.total_width)
        next_y = np.clip(self.agent.y + dy, 0, self.total_height)

        prev_x, prev_y = self.agent.x, self.agent.y
        farthest_x, farthest_y = self.find_farthest_point(
            prev_x, prev_y, next_x, next_y
        )
        self.agent.move(farthest_x, farthest_y)

        self.check_termination()

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

    def check_termination(self):
        # Calculate the distance between the agent's recent location and the destination
        recent_x, recent_y = self.agent.trajectory[-1]
        distance = math.sqrt(
            (recent_x - self.dest_x) ** 2 + (recent_y - self.dest_y) ** 2
        )

        # Check if the distance is within the specified radius (r)
        if distance <= self.radius:
            self.is_terminated = True
            self.agent.is_Arrived = True

    def plot_grid(self):
        fig, ax = plt.subplots()

        # Draw the grid lines with adjusted linewidth
        for i in range(self.rows + 1):
            y = i * self.cell_height
            plt.plot([0, self.total_width], [y, y], color="black", linewidth=0.5)

        for i in range(self.cols + 1):
            x = i * self.cell_width
            plt.plot([x, x], [0, self.total_height], color="black", linewidth=0.5)

        # Plot the start and destination points with larger size
        plt.scatter(
            self.start_x, self.start_y, color="green", marker="o", s=25, label="Start"
        )
        plt.scatter(
            self.dest_x, self.dest_y, color="red", marker="o", s=25, label="Destination"
        )

        # Plot barrier regions with the same color
        for region in self.barrier_regions:
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
        traj_x, traj_y = zip(*self.agent.trajectory)
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
        ax.set_xticks(np.arange(0, self.total_width + self.cell_width, self.cell_width))
        ax.set_yticks(
            np.arange(0, self.total_height + self.cell_height, self.cell_height)
        )

        # Move the legend outside the figure
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title(f"Grid (Level {self.level})")

        # Show the plot
        plt.show()