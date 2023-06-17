# only check the position is same as Root(starting)
def check_Root_pos(Grid, level, x, y):
    return Grid.check_Root_pos(level=level, x=x, y=y)


# only check the position is same as terminal(destination)
def check_terminal_pos(Grid, level, x, y):
    return Grid.check_terminal_pos(level=level, x=x, y=y)


# level i's location -> level i + 1's location
def hierarchy_a_map(level, x, y, cell_width, cell_height, RS=2):
    # continuous action space
    if level == 1:
        hier_x, hier_y = int(x / cell_width), int(y / cell_height)

    # discrete action space
    else:  # level > 1
        hier_x, hier_y = int(x / RS), int(y / RS)

    return level + 1, hier_x, hier_y


def hierarchy_map(level_current, level2move, pos, RS=2):
    # continuous Action space
    # if level_current == 0:
    #     level_up = level_to_move - level_current - 1
    #     if level_up < 0:
    #         raise ValueError('wrong level input')
        
    #     hier_x = int(int(x / cell_width) / (RS ** level_up))
    #     hier_y = int(int(y / cell_height) / (RS ** level_up))
        
    # else:
    x, y = pos
    level_up = level2move - level_current
    if level_up < 0:
        raise ValueError('wrong level input')
    hier_x = int(x / (RS ** level_up))
    hier_y = int(y / (RS ** level_up))

    return hier_x, hier_y