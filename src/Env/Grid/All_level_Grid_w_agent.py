import random
import math
import matplotlib.pyplot as plt
import numpy as np

from .Higher_Grid_w_agent import high_Grid_w_agent
from .Cont_Grid import lowest_Grid_w_agent


# for get the highest level (as the heesang's paper)
def largest_power_gcd(a, b, RS=2):
    # Step 1: Calculate the GCD of a and b
    gcd = math.gcd(a, b)

    # Step 2: Find the largest power of 2 that divides the GCD
    largest_power = 0
    while gcd % RS == 0:
        largest_power += 1
        gcd //= RS

    return largest_power


def check_both_power_of_RS(a, b, RS=2):
    largest_power = largest_power_gcd(a, b, RS)
    if (a == b) and (a == RS**largest_power):
        return largest_power
    else:
        return largest_power + 1


class All_level_Grids_w_agent:
    def __init__(
        self,
        l1_rows: int,
        l1_cols: int,
        l1_width,  # not have to be int, but recommended
        l1_height,  # not have to be int, but recommended
        destination_radius: float = 2,
        barrier_find_segment: int = 101,
        highest_level=None,
        A_space={(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)},
    ):
        if highest_level is None:
            self.highest_level = check_both_power_of_RS(l1_rows, l1_cols)
        else:
            self.highest_level = highest_level

        self.Grid_list = [
            lowest_Grid_w_agent(
                l1_rows,
                l1_cols,
                l1_width,
                l1_height,
                level=0,
                destination_radius=destination_radius,
                barrier_find_segment=barrier_find_segment,
            )
        ]
        for i in range(self.highest_level):
            rows, cols = l1_rows / (2**i), l1_cols / (2**i)
            cell_width = l1_width * (2**i)
            cell_height = l1_height * (2**i)

            higher_Grid = high_Grid_w_agent(
                rows,
                cols,
                cell_width,
                cell_height,
                self.Grid_list[0].start_x,
                self.Grid_list[0].start_y,
                self.Grid_list[0].dest_x,
                self.Grid_list[0].dest_y,
                level=i+1,
                A_space=A_space,
            )
            self.Grid_list.append(higher_Grid)

    def __str__(self):
        return (
            f" Highest level: {self.highest_level} \n"
            f"start: ({self.Grid_list[0].start_x}, {self.Grid_list[0].start_y}), \n"
            f"destination: ({self.Grid_list[0].dest_x}, {self.Grid_list[0].dest_y}), \n"
        )

    def check_Root_pos(self, level, x, y):
        if level == 0:
            return self.Grid_list[level].check_Root_pos(x, y)
        else:  # level > 0
            return self.Grid_list[level].check_Root_pos(x, y)

    def check_terminal_pos(self, level, x, y):
        if level == 0:
            return self.Grid_list[level].check_termination_pos(x, y, False)
        else:  # level > 0
            return self.Grid_list[level].check_termination_pos(x, y, False)
        
        
class high_level_Grids_w_agent:
    def __init__(
        self,
        l1_rows: int,
        l1_cols: int,
        l1_width,  # not have to be int, but recommended
        l1_height,  # not have to be int, but recommended
        destination_radius: float = 2,
        barrier_find_segment: int = 101,
        highest_level=None,
        A_space={(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)},
    ):
        if highest_level is None:
            self.highest_level = check_both_power_of_RS(l1_rows, l1_cols)
        else:
            self.highest_level = highest_level
            
        self.l1_grid = lowest_Grid_w_agent(
            l1_rows,
            l1_cols,
            l1_width,
            l1_height,
            level=0,
            destination_radius=destination_radius,
            barrier_find_segment=barrier_find_segment,
        )

        self.Grid_list = []
        
        for i in range(self.highest_level):
            rows, cols = l1_rows / (2**i), l1_cols / (2**i)
            cell_width = l1_width * (2**i)
            cell_height = l1_height * (2**i)

            higher_Grid = high_Grid_w_agent(
                rows,
                cols,
                cell_width,
                cell_height,
                self.l1_grid.start_x,
                self.l1_grid.start_y,
                self.l1_grid.dest_x,
                self.l1_grid.dest_y,
                level=i + 1,
                A_space=A_space,
            )
            self.Grid_list.append(higher_Grid)

    def __str__(self):
        return (
            f" Highest level: {self.highest_level} \n"
            f"start: ({self.Grid_list[0].start_x}, {self.Grid_list[0].start_y}), \n"
            f"destination: ({self.Grid_list[0].dest_x}, {self.Grid_list[0].dest_y}), \n"
        )

    def check_Root_pos(self, level, x, y):
        return self.Grid_list[level].check_Root_pos(x, y)

    def check_terminal_pos(self, level, x, y):
        return self.Grid_list[level].check_termination_pos(x, y, False)