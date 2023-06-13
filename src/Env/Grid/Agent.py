import math
import random

class lowest_Agent:  # continuous action
    def __init__(self, start_x, start_y, level=0, A_radius=5):
        self.level = level
        self.A_radius=A_radius
        self.x = start_x
        self.y = start_y
        self.trajectory = [(start_x, start_y)]
        self.A_trajectory = []
        self.is_Arrived = False
        
    def __str__(self):
        return (
            f"Agent Level: {self.level} \n"
            f"Start location: ({self.trajectory[0][0]}, {self.trajectory[0][1]}), \n"
            f"current location: ({self.x}, {self.y}) \n"
            f"is_Arrived: {self.is_Arrived}"
        )
        
    def move_plan(self, radius, radians):
        radius = self.check_A_radius(radius)
        dx = radius * math.cos(radians)
        dy = radius * math.sin(radians)
        self.A_trajectory.append((dx, dy))
        return dx, dy
    
    def check_A_radius(self, radius):
        if radius <= 0:
            return 0
        elif radius > self.A_radius:
            return self.A_radius
        else:
            return radius
    
    def move(self, next_x, next_y):
        self.x, self.y = next_x, next_y
        self.trajectory.append((next_x, next_y))
        
    def random_action(self):
        radius = random.uniform(0, self.A_radius)
        radians = random.uniform(0, 2 * math.pi)
        return radius, radians        

        
class highlevel_Agent:  # discrete action
    def __init__(self, start_x, start_y, level, A_space={(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)}):
        self.level = level
        self.x = start_x
        self.y = start_y
        self.trajectory = [(start_x, start_y)]
        self.A_trajectory = []
        self.A_space = A_space
        self.is_Arrived = False
        
    def __str__(self):
        return (
            f"Agent Level: {self.level} \n"
            f"Start location: ({self.trajectory[0][0]}, {self.trajectory[0][1]}), \n"
            f"current location: ({self.x}, {self.y}) \n"
            f"is_Arrived: {self.is_Arrived}"
        )
        
    def move_plan(self, dx, dy):
        self.A_trajectory.append((dx, dy))
        return dx, dy
    
    def move(self, next_x, next_y):
        self.x, self.y = next_x, next_y
        self.trajectory.append((self.x, self.y))
        
    def random_action(self, action_space: list):
        return random.choice(action_space)
    
    def get_possible_A(self, rows, cols, x, y):  # only possible at higher (discrete action space)
        possible_A = []
        
        # Check the neighboring cells in all directions
        directions = tuple(self.A_space)
        
        for dx, dy in directions:
            new_x = x + dx
            new_y = y + dy
            
            # Check if the new position is within the grid boundaries
            if 0 <= new_x < cols and 0 <= new_y < rows:
                possible_A.append((dx, dy))
        
        return possible_A