from collections import defaultdict

class STRIPS:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.actions = []

    def solve(self):
        self.actions = []
        state = self.initial_state.copy()
        while not self.goal_test(state):
            applicable_actions = self.get_applicable_actions(state)
            action = self.select_action(applicable_actions)
            self.actions.append(action)
            self.apply_action(state, action)
        return self.actions

    def goal_test(self, state):
        return state == self.goal_state

    def get_applicable_actions(self, state):
        actions = []
        x, y = state
        if x > 0:
            actions.append("LEFT")
        if x < len(grid) - 1:
            actions.append("RIGHT")
        if y > 0:
            actions.append("UP")
        if y < len(grid[0]) - 1:
            actions.append("DOWN")
        return actions

    def select_action(self, actions):
        return actions[0]  # Select the first available action

    def apply_action(self, state, action):
        x, y = state
        if action == "LEFT":
            state[0] = x - 1
        elif action == "RIGHT":
            state[0] = x + 1
        elif action == "UP":
            state[1] = y - 1
        elif action == "DOWN":
            state[1] = y + 1


# Example usage
# grid = [
#     [0, 0, 0, 0],
#     [0, 1, 0, 1],
#     [0, 0, 0, 0],
#     [0, 1, 0, 0]
# ]
# start_state = [0, 0]
# goal_state = [3, 3]

# strips = STRIPS(start_state, goal_state)
# actions = strips.solve()
# print(actions)
