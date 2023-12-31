# Hand designed grids

class HD_Grid:
    def __init__(self, barriers, start_goal):
        self.barriers = barriers
        self.start_goal = start_goal

HD_GRID_1 = HD_Grid(
    barriers=[(1, 0)],
    start_goal=[(0, 0), (2, 0)]
)

HD_GRID_2 = HD_Grid(
    barriers=[(1, 0), (1, 1)],
    start_goal=[(0, 0), (2, 0)]
)

HD_GRID_3 = HD_Grid(
    barriers=[(0, 2), (1, 1), (2, 1)],
    start_goal=[(0, 0), (2, 2)]
)


HD_GRID_4 = HD_Grid(
    barriers=[(0, 1), (1, 1), (2, 1)],
    start_goal=[(0, 0), (1, 3)]
)


HD_GRID_5 = HD_Grid(
    barriers=[(0, 2), (1, 2), (2, 2)],
    start_goal=[(0, 0), (1, 3)]
)


HD_GRID_6 = HD_Grid(
    barriers=[(0, 1), (1, 1), (2, 1), (3, 1)],
    start_goal=[(0, 0), (1, 3)]
)


HD_GRID_7 = HD_Grid(
    barriers=[(2, 3), (2, 4), (2, 5), (3, 3), (4, 3), (5, 3), (5, 2), (5, 4)],
    start_goal=[(1, 2), (6, 5)]
)


# HD_GRID_8 = HD_Grid(
#     barriers=[(0, 1), (1, 1), (2, 1), (3, 1)],
#     start_goal=[(0, 0), (1, 3)]
# )


# HD_GRID_9 = HD_Grid(
#     barriers=[(0, 1), (1, 1), (2, 1), (3, 1)],
#     start_goal=[(0, 0), (1, 3)]
# )




hd_grids = [HD_GRID_1, HD_GRID_2, HD_GRID_3, HD_GRID_4, HD_GRID_5, HD_GRID_6, HD_GRID_7]

num_grid_1 = HD_Grid(
    barriers=[(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)],
    start_goal=[(2, 6), (0, 0)]
)

num_grid_2 = HD_Grid(
    barriers=[(2, 2), (2, 3), (2, 4), (2, 6), (3, 2), (3, 4), (3, 6), (4, 2), (4, 4), (4, 6), (5, 2), (5, 4), (5, 5), (5, 6)],
    start_goal=[(6, 6), (2, 5)]
)

num_grid_3 = HD_Grid(
    barriers=[(2, 2), (2, 4), (2, 6), (3, 2), (3, 4), (3, 6), (4, 2), (4, 4), (4, 6), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)],
    start_goal=[(4, 3), (3, 5)]
)

num_grid_4 = HD_Grid(
    barriers=[(2, 5), (2, 4), (2, 6), (3, 4), (4, 4), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)],
    start_goal=[(4, 5), (5, 1)]
)

num_grid_5 = HD_Grid(
    barriers=[(2, 2), (2, 4), (2, 5), (2, 6), (3, 2), (3, 4), (3, 6), (4, 2), (4, 4), (4, 6), (5, 2), (5, 3), (5, 4), (5, 6)],
    start_goal=[(4, 5), (5, 1)]
)

num_grid_6 = HD_Grid(
    barriers=[(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 2), (3, 4), (3, 6), (4, 2), (4, 4), (4, 6), (5, 2), (5, 3), (5, 4), (5, 6)],
    start_goal=[(6, 7), (2, 1)]
)

num_grid_7 = HD_Grid(
    barriers=[(2, 6), (3, 6), (4, 6), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)],
    start_goal=[(4, 5), (3, 7)]
)

num_grid_8 = HD_Grid(
    barriers=[(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 2), (3, 4), (3, 6), (4, 2), (4, 4), (4, 6), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)],
    start_goal=[(4, 1), (3, 7)]
)

num_grid_9 = HD_Grid(
    barriers=[(2, 2), (2, 4), (2, 5), (2, 6), (3, 2), (3, 4), (3, 6), (4, 2), (4, 4), (4, 6), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)],
    start_goal=[(2, 3), (6, 7)]
)


num_grid_10 = HD_Grid(
    barriers=[(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (4, 2), (4, 6), (5, 2), (5, 3), (5, 4), (5, 6)],
    start_goal=[(2, 6), (4, 4)]
)


number_grids = [num_grid_1, num_grid_2, num_grid_3, num_grid_4, num_grid_5, num_grid_6, num_grid_7, num_grid_8, num_grid_9, num_grid_10]


maze_grid_1 = HD_Grid(
    barriers=[(0, 3), (0, 4), (1, 3), (1, 4), (3, 0), (3, 1), (3, 3), (3, 4), (3, 6), (3, 7), (4, 0), (4, 1), (4, 3), (4, 4), (4, 6), (4, 7), (6, 3), (6, 4), (7, 3), (7, 4)],
    start_goal=[(0, 0), (7, 7)]
)

maze_grid_2 = HD_Grid(
    barriers=[(0, 3), (0, 4), (1, 3), (1, 4), (3, 0), (3, 1), (3, 3), (3, 4), (3, 6), (3, 7), (4, 0), (4, 1), (4, 3), (4, 4), (4, 6), (4, 7), (6, 3), (6, 4), (7, 3), (7, 4)],
    start_goal=[(0, 0), (7, 0)]
)

maze_grid_3 = HD_Grid(
    barriers=[(0, 3), (0, 4), (1, 3), (1, 4), (3, 0), (3, 1), (3, 3), (3, 4), (3, 6), (3, 7), (4, 0), (4, 1), (4, 3), (4, 4), (4, 6), (4, 7), (6, 3), (6, 4), (7, 3), (7, 4)],
    start_goal=[(0, 2), (6, 5)]
)

maze_grid_4 = HD_Grid(
    barriers=[(0, 3), (0, 4), (1, 3), (1, 4), (3, 0), (3, 1), (3, 3), (3, 4), (3, 6), (3, 7), (4, 0), (4, 1), (4, 3), (4, 4), (4, 6), (4, 7), (6, 3), (6, 4), (7, 3), (7, 4)],
    start_goal=[(2, 1), (6, 0)]
)


maze_grid_5 = HD_Grid(
    barriers=[(0, 3), (1, 3), (1, 6), (2, 5), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 7), (5, 0), (5, 3), (5, 6), (6, 3), (6, 5), (6, 6), (7, 3)],
    start_goal=[(0, 0), (7, 7)]
)

maze_grid_6 = HD_Grid(
    barriers=[(0, 3), (1, 3), (1, 6), (2, 5), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 7), (5, 0), (5, 3), (5, 6), (6, 3), (6, 5), (6, 6), (7, 3)],
    start_goal=[(0, 0), (7, 0)]
)

maze_grid_7 = HD_Grid(
    barriers=[(0, 3), (1, 3), (1, 6), (2, 5), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 7), (5, 0), (5, 3), (5, 6), (6, 3), (6, 5), (6, 6), (7, 3)],
    start_goal=[(0, 7), (7, 0)]
)

maze_grid_8 = HD_Grid(
    barriers=[(0, 3), (1, 3), (1, 6), (2, 5), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 7), (5, 0), (5, 3), (5, 6), (6, 3), (6, 5), (6, 6), (7, 3)],
    start_goal=[(7, 7), (7, 0)]
)

maze_grid_9 = HD_Grid(
    barriers=[(0, 3), (1, 3), (1, 6), (2, 5), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 7), (5, 0), (5, 3), (5, 6), (6, 3), (6, 5), (6, 6), (7, 3)],
    start_goal=[(0, 7), (0, 0)]
)

maze_grid_10 = HD_Grid(
    barriers=[(0, 3), (1, 3), (1, 6), (2, 5), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 7), (5, 0), (5, 3), (5, 6), (6, 3), (6, 5), (6, 6), (7, 3)],
    start_goal=[(5, 7), (5, 4)]
)


maze_grids = [maze_grid_1, maze_grid_2, maze_grid_3, maze_grid_4, maze_grid_5, maze_grid_6, maze_grid_7, maze_grid_8, maze_grid_9, maze_grid_10]
