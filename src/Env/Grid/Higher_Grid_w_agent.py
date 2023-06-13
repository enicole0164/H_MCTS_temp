import matplotlib.pyplot as plt
import numpy as np

from .Agent import highlevel_Agent


class high_Grid_w_agent:
    def __init__(
        self,
        rows: int,
        cols: int,
        cell_width,  # not have to be int, but recommended
        cell_height,  # not have to be int, but recommended
        start_x,
        start_y,
        dest_x,
        dest_y,
        level: int = 1,  # start from level 1, 2, ..., n
        A_space={(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)},  # discrete action_space
    ):
        self.level = level

        if float(rows).is_integer() and float(cols).is_integer():
            pass
        else:
            raise Exception(
                "wrong type of rows {}, cols {}".format(self.rows, self.cols)
            )
        self.rows = int(rows)
        self.cols = int(cols)
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.total_width = cols * cell_width
        self.total_height = rows * cell_height
        self.A_space = A_space
        self.is_terminated = False

        self.set_start_dest(start_x, start_y, dest_x, dest_y)

        self.agent = highlevel_Agent(
            level=self.level, start_x=self.start_x, start_y=self.start_y
        )

    def __str__(self):
        return (
            f"level: {self.level} \n"
            f"Grid: {self.rows} rows, {self.cols} columns, \n"
            f"total size: {self.total_width}x{self.total_height}, \n"
            f"start: ({self.start_x}, {self.start_y}), destination: ({self.dest_x}, {self.dest_y}), \n"
            f"is_terminated: {self.is_terminated}"
        )

    def set_start_dest(self, start_x, start_y, dest_x, dest_y):
        start_cell = (int(start_x / self.cell_width), int(start_y / self.cell_height))
        dest_cell = (int(dest_x / self.cell_width), int(dest_y / self.cell_height))

        self.start_x, self.start_y = start_cell
        self.dest_x, self.dest_y = dest_cell

    def move_agent(self, dx, dy):
        if (dx, dy) not in self.A_space:
            raise Exception("Wrong action")
        next_x = self.agent.x + dx
        next_y = self.agent.y + dy

        # Clip the next position to ensure it stays within the grid boundaries
        next_x = np.clip(next_x, 0, self.cols - 1)
        next_y = np.clip(next_y, 0, self.rows - 1)

        self.agent.move(next_x, next_y)
        self.check_termination_pos(next_x, next_y, move_agent=True)

    def check_termination_pos(self, x, y, move_agent=False):
        # Calculate the distance between the agent's recent location and the destination
        # recent_x, recent_y = self.agent.trajectory[-1]

        # Check if the distance is within the specified radius (r)
        if (x, y) == (self.dest_x, self.dest_y) and move_agent:
            self.is_terminated = True
            self.agent.is_Arrived = True

    def check_Root_pos(self, x, y):
        return (self.start_x, self.start_y) == (x, y)
    
    def get_possible_A(self, x, y):  # only possible at higher (discrete action space)
        possible_A = []
        
        # Check the neighboring cells in all directions
        directions = tuple(self.A_space)
        
        for dx, dy in directions:
            new_x = x + dx
            new_y = y + dy
            
            # Check if the new position is within the grid boundaries
            if 0 <= new_x < self.cols and 0 <= new_y < self.rows:
                possible_A.append((dx, dy))
        
        return possible_A
        

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
            self.start_x * self.cell_width + self.cell_width / 2,
            self.start_y * self.cell_height + self.cell_height / 2,
            color="green",
            marker="o",
            s=25,
            label="Start",
        )
        plt.scatter(
            self.dest_x * self.cell_width + self.cell_width / 2,
            self.dest_y * self.cell_height + self.cell_height / 2,
            color="red",
            marker="o",
            s=25,
            label="Destination",
        )

        # Plot the agent's path with a different color and marker
        traj_x, traj_y = zip(*self.agent.trajectory)
        plt.scatter(
            np.array(traj_x[1:]) * self.cell_width + self.cell_width / 2,
            np.array(traj_y[1:]) * self.cell_height + self.cell_height / 2,
            color="skyblue",
            marker="o",
            s=25,
            zorder=2,
        )
        plt.plot(
            np.array(traj_x) * self.cell_width + self.cell_width / 2,
            np.array(traj_y) * self.cell_height + self.cell_height / 2,
            color="skyblue",
            linewidth=2,
            label="Agent's path",
            zorder=1,
        )

        # Plot arrows for the agent's direction with larger size
        for i in range(1, len(traj_x)):
            dx = traj_x[i] - traj_x[i - 1]
            dy = traj_y[i] - traj_y[i - 1]
            plt.arrow(
                traj_x[i - 1] * self.cell_width + self.cell_width / 2,
                traj_y[i - 1] * self.cell_height + self.cell_height / 2,
                dx * self.cell_width,
                dy * self.cell_height,
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
