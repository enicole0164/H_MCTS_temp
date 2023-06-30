import math

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


def hierarchy_map(level_curr, level2move, pos, RS=2):
    # continuous Action space
    # if level_curr == 0:
    #     level_up = level_to_move - level_curr - 1
    #     if level_up < 0:
    #         raise ValueError('wrong level input')
        
    #     hier_x = int(int(x / cell_width) / (RS ** level_up))
    #     hier_y = int(int(y / cell_height) / (RS ** level_up))
        
    # else:
    x, y = pos
    level_up = level2move - level_curr
    if level_up < 0:
        raise ValueError('wrong level input')
    hier_x = int(x / (RS ** level_up))
    hier_y = int(y / (RS ** level_up))

    return hier_x, hier_y

def hierarchy_map_cont(level_curr, level2move, pos, RS=2):
    x, y = pos
    
    # continuous Action space
    if level_curr == 0:
        if level2move == 0:
            hier_x = x
            hier_y = y
        else:
            
            level_up = level2move - level_curr - 1
            if level_up < 0:
                raise ValueError('wrong level input')
            
            hier_x = int(int(x / RS) / (RS ** level_up))
            hier_y = int(int(y / RS) / (RS ** level_up))
    # discrete Action space
    else:
        level_up = level2move - level_curr
        if level_up < 0:
            raise ValueError('wrong level input')
        hier_x = int(x / (RS ** level_up))
        hier_y = int(y / (RS ** level_up))

    return hier_x, hier_y

def check_both_power_of_RS(a, b, RS=2):
    largest_power = largest_power_gcd(a, b, RS)
    if (a == b) and (a == RS**largest_power):
        return largest_power
    else:
        return largest_power + 1
    
def largest_power_gcd(a, b, RS=2):
    # Step 1: Calculate the GCD of a and b
    gcd = math.gcd(a, b)

    # Step 2: Find the largest power of 2 that divides the GCD
    largest_power = 0
    while gcd % RS == 0:
        largest_power += 1
        gcd //= RS

    return largest_power